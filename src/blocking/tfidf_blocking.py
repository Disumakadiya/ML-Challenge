"""Sparse TF-IDF candidate retrieval for business entity resolution (M2_E5).

Source 2 and Source 3 are indexed as one candidate universe. Retrieval keeps
their original entity IDs and source identity, so a candidate ID is always the
original ``S2-``/``S3-`` value from the provided TSVs.

The module is deliberately out-of-core friendly. Target records are streamed
once into a columnar store of memory-mapped files (byte blobs for text, fixed
width vectors for codes and lengths) together with two CSR TF-IDF matrices, one
over normalized ``business_name`` and one over normalized ``business_address``.
Retrieval then runs in two nested loops over batches of Source 1 queries and
blocks of target rows, so peak memory depends on the batch and block sizes
rather than on the corpus size. A dense all-pairs similarity matrix is never
materialized.

Text normalization is *not* reimplemented here: the store is always built from
frames returned by :func:`src.preprocessing.prepare_source`, and the same
normalized columns (``name_norm``/``address_norm``/``country_norm``) drive both
the index and the pair features.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sparse

from src.preprocessing import FilePath, iter_source_chunks, prepare_source

FIELDS: tuple[str, str] = ("name", "address")
FIELD_COLUMNS: dict[str, str] = {"name": "name_norm", "address": "address_norm"}
SOURCE_CODES: dict[str, int] = {"S2": 0, "S3": 1}
UNKNOWN_SOURCE_CODE = 2


@dataclass(frozen=True)
class TfidfConfig:
    """Deterministic TF-IDF and retrieval controls for the E5 blocker.

    ``name_features``/``address_features`` are the sizes of the hashed term
    spaces and must be powers of two. Hashing keeps the index build free of a
    multi-million entry Python vocabulary, which is what makes a single
    streaming pass over the corpus affordable.
    """

    name_features: int = 1 << 22
    address_features: int = 1 << 22
    min_token_length: int = 1
    min_df: int = 1
    max_df_ratio: float = 1.0
    sublinear_tf: bool = True
    source_chunk_rows: int = 100_000
    target_block_rows: int = 8_192
    query_batch_rows: int = 1_024
    max_target_rows: int | None = None

    def __post_init__(self) -> None:
        for name in ("name_features", "address_features"):
            size = getattr(self, name)
            if size < 1024 or size & (size - 1):
                raise ValueError(f"{name} must be a power of two of at least 1024.")
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be positive.")
        if self.min_df < 0:
            raise ValueError("min_df must not be negative.")
        if not 0.0 < self.max_df_ratio <= 1.0:
            raise ValueError("max_df_ratio must be in (0, 1].")
        if self.source_chunk_rows < 1 or self.target_block_rows < 1:
            raise ValueError("source_chunk_rows and target_block_rows must be positive.")
        if self.query_batch_rows < 1:
            raise ValueError("query_batch_rows must be positive.")
        if self.max_target_rows is not None and self.max_target_rows < 1:
            raise ValueError("max_target_rows must be positive when provided.")

    def features_for(self, field: str) -> int:
        """Return the hashed term-space size used for ``field``."""
        return self.name_features if field == "name" else self.address_features


def hash_term(token: str, modulus: int) -> int:
    """Map a token to a term index with a process-independent 32-bit hash."""
    return zlib.crc32(token.encode("utf-8")) % modulus


def _encode_texts(
    texts: Sequence[str],
    modulus: int,
    min_token_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CSR raw term counts, indptr, and per-term document frequency.

    Rows whose tokens are all shorter than ``min_token_length`` produce an empty
    row, which is recorded as a valid indptr step so row indices stay aligned
    with the input frame.
    """
    indptr = np.zeros(len(texts) + 1, dtype=np.int64)
    document_frequency = np.zeros(modulus, dtype=np.int32)
    term_indices: list[int] = []
    term_counts: list[float] = []
    cursor = 0
    for row, text in enumerate(texts):
        counts: dict[int, int] = {}
        for token in text.split():
            if len(token) < min_token_length:
                continue
            term_index = hash_term(token, modulus)
            counts[term_index] = counts.get(term_index, 0) + 1
        if counts:
            keys = sorted(counts)
            term_indices.extend(keys)
            term_counts.extend(float(counts[key]) for key in keys)
            document_frequency[keys] += 1
            cursor += len(keys)
        indptr[row + 1] = cursor
    return (
        indptr,
        np.asarray(term_indices, dtype=np.int32),
        np.asarray(term_counts, dtype=np.float32),
        document_frequency,
    )


def _row_normalize(
    values: np.ndarray,
    indptr: np.ndarray,
    idf: np.ndarray,
    indices: np.ndarray,
    sublinear_tf: bool,
) -> np.ndarray:
    """Apply sublinear TF, IDF weighting, and L2 row normalization in place."""
    if values.size:
        if sublinear_tf:
            values[:] = 1.0 + np.log(values)
        values *= idf[indices]
        squared = values * values
        cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
        row_sums = cumulative[indptr[1:]] - cumulative[indptr[:-1]]
        inverse = np.zeros(row_sums.shape, dtype=np.float32)
        np.divide(1.0, np.sqrt(row_sums), out=inverse, where=row_sums > 0.0)
        values *= np.repeat(inverse, np.diff(indptr)).astype(values.dtype)
        values[~np.isfinite(values)] = 0.0
    return values


class _BlobWriter:
    """Appends variable-length byte values with an int64 offset index."""

    def __init__(self, store_dir: Path, name: str) -> None:
        self._values = (store_dir / f"{name}.bin").open("wb")
        self._offsets = (store_dir / f"{name}.off").open("wb")
        self._offset = 0
        self._closed = False

    def append(self, values: Sequence[bytes]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed column.")
        if not values:
            return
        lengths = np.fromiter(
            (len(value) for value in values), dtype=np.int64, count=len(values)
        )
        # One start offset per value; ``close`` appends the final end offset.
        offsets = np.empty(len(values), dtype=np.int64)
        offsets[0] = self._offset
        np.cumsum(lengths[:-1], out=offsets[1:])
        offsets[1:] += self._offset
        self._offsets.write(offsets.tobytes())
        for value in values:
            self._values.write(value)
        self._offset += int(lengths.sum())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._offsets.write(np.array([self._offset], dtype=np.int64).tobytes())
        self._values.close()
        self._offsets.close()


class _VectorWriter:
    """Appends fixed-width values to a raw binary file."""

    def __init__(self, store_dir: Path, name: str, dtype: str) -> None:
        self._handle = (store_dir / f"{name}.bin").open("wb")
        self._dtype = np.dtype(dtype)
        self._closed = False

    def append(self, values: np.ndarray | Sequence[int]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed column.")
        self._handle.write(np.asarray(values, dtype=self._dtype).tobytes())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._handle.close()


def _map_int64(path: Path) -> np.ndarray:
    """Memory-map an int64 column, tolerating a zero-length file."""
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.memmap(path, dtype=np.int64, mode="r")


def _map_dtype(path: Path, dtype: np.dtype) -> np.ndarray:
    """Memory-map a typed column, tolerating a zero-length file."""
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")


class _BlobReader:
    """Random access to a memory-mapped byte blob by row index."""

    def __init__(self, store_dir: Path, name: str) -> None:
        self._values = _map_dtype(store_dir / f"{name}.bin", np.dtype(np.uint8))
        self._offsets = _map_int64(store_dir / f"{name}.off")

    def __len__(self) -> int:
        return int(self._offsets.shape[0]) - 1

    def get(self, index: int) -> bytes:
        start = int(self._offsets[index])
        return self._values[start : int(self._offsets[index + 1])].tobytes()

    def length(self, indices: np.ndarray) -> np.ndarray:
        offsets = np.asarray(self._offsets[indices])
        following = np.asarray(self._offsets[indices + 1])
        return (following - offsets).astype(np.int32)

    def starts(self, indices: np.ndarray) -> np.ndarray:
        """Return the byte offset of each requested row, in the given order."""
        return np.asarray(self._offsets[indices], dtype=np.int64)

    def read(self, start: int, total: int) -> bytes:
        """Return ``total`` bytes of the value file beginning at ``start``."""
        if total <= 0:
            return b""
        return self._values[start : start + total].tobytes()

    def close(self) -> None:
        for array in (self._values, self._offsets):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


class _VectorReader:
    """Random access to a memory-mapped fixed-width column."""

    def __init__(self, store_dir: Path, name: str, dtype: str) -> None:
        self._values = _map_dtype(store_dir / f"{name}.bin", np.dtype(dtype))

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def __getitem__(self, index: int) -> int:
        return int(self._values[index])

    def gather(self, indices: np.ndarray, dtype: str = "float32") -> np.ndarray:
        return np.asarray(self._values[indices], dtype=dtype)

    def close(self) -> None:
        mmap = getattr(self._values, "_mmap", None)
        if mmap is not None:
            mmap.close()


class TargetStoreWriter:
    """Streams prepared S2/S3 chunks into a columnar target store."""

    def __init__(self, store_dir: FilePath, config: TfidfConfig) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.row_count = 0
        self.country_codes: dict[str, int] = {}
        self.source_counts: dict[str, int] = {}
        self._nonzeros = {field: 0 for field in FIELDS}
        self._document_frequency = {
            field: np.zeros(config.features_for(field), dtype=np.int32)
            for field in FIELDS
        }
        self._blobs = {
            name: _BlobWriter(self.store_dir, name)
            for name in ("entity_id", "name", "address")
        }
        self._vectors = {
            "country_code": _VectorWriter(self.store_dir, "country_code", "int16"),
            "source_code": _VectorWriter(self.store_dir, "source_code", "int8"),
            "name_missing": _VectorWriter(self.store_dir, "name_missing", "int8"),
            "address_missing": _VectorWriter(self.store_dir, "address_missing", "int8"),
        }
        self._csr: dict[str, dict[str, _VectorWriter]] = {}
        for field in FIELDS:
            self._csr[field] = {
                "indptr": _VectorWriter(self.store_dir, f"{field}_indptr", "int64"),
                "indices": _VectorWriter(self.store_dir, f"{field}_indices", "int32"),
                "data": _VectorWriter(self.store_dir, f"{field}_data", "float32"),
            }
            self._csr[field]["indptr"].append(np.zeros(1, dtype=np.int64))

    def add_prepared_frame(self, frame) -> int:
        """Append one prepared S2/S3 chunk and return the number of rows kept."""
        limit = len(frame)
        if self.config.max_target_rows is not None:
            remaining = self.config.max_target_rows - self.row_count
            if remaining <= 0:
                return 0
            limit = min(limit, remaining)
            frame = frame.iloc[:limit]
        if limit == 0:
            return 0

        self._blobs["entity_id"].append(
            [str(value).encode("utf-8") for value in frame["entity_id"]]
        )
        for field in FIELDS:
            column = frame[FIELD_COLUMNS[field]]
            self._blobs[field].append(
                [str(value).encode("utf-8") for value in column]
            )
            modulus = self.config.features_for(field)
            indptr, indices, counts, frequency = _encode_texts(
                [str(value) for value in column], modulus, self.config.min_token_length
            )
            self._document_frequency[field] += frequency
            offset = self._nonzeros[field]
            self._csr[field]["indptr"].append(indptr[1:] + offset)
            self._csr[field]["indices"].append(indices)
            self._csr[field]["data"].append(counts)
            self._nonzeros[field] = offset + int(indices.shape[0])

        country_values = [str(value) for value in frame["country_norm"]]
        encoded_countries: list[int] = []
        for country in country_values:
            code = self.country_codes.get(country)
            if code is None:
                code = len(self.country_codes)
                self.country_codes[country] = code
            encoded_countries.append(code)
        self._vectors["country_code"].append(np.asarray(encoded_countries, dtype="int16"))
        self._vectors["name_missing"].append(
            np.asarray(frame["name_missing"], dtype="int8")
        )
        self._vectors["address_missing"].append(
            np.asarray(frame["address_missing"], dtype="int8")
        )

        source_values = [str(value) for value in frame["source"]]
        encoded_sources = [
            SOURCE_CODES.get(source, UNKNOWN_SOURCE_CODE) for source in source_values
        ]
        self._vectors["source_code"].append(np.asarray(encoded_sources, dtype="int8"))
        for source in source_values:
            self.source_counts[source] = self.source_counts.get(source, 0) + 1

        self.row_count += limit
        return limit

    def finalize(self) -> None:
        """Close writers, compute IDF, and L2-normalize the stored TF-IDF values."""
        for name, blob in self._blobs.items():
            blob.close()
        for vector in self._vectors.values():
            vector.close()
        for field, writers in self._csr.items():
            for writer in writers.values():
                writer.close()

        row_count = self.row_count
        idf: dict[str, np.ndarray] = {}
        for field in FIELDS:
            frequency = self._document_frequency[field]
            field_idf = (
                np.log((1.0 + row_count) / (1.0 + frequency.astype(np.float64))) + 1.0
            ).astype(np.float32)
            if self.config.max_df_ratio < 1.0:
                field_idf[frequency > self.config.max_df_ratio * row_count] = 0.0
            if self.config.min_df > 1:
                field_idf[frequency < self.config.min_df] = 0.0
            field_idf[~np.isfinite(field_idf)] = 0.0
            idf[field] = field_idf
            self._document_frequency[field] = np.zeros(0, dtype=np.int32)
            self._normalize_stored_matrix(field, field_idf)
            np.asarray(field_idf).tofile(self.store_dir / f"{field}_idf.bin")

        (self.store_dir / "country_vocab.json").write_text(
            json.dumps(self.country_codes, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.store_dir / "meta.json").write_text(
            json.dumps(
                {
                    "row_count": row_count,
                    "source_counts": dict(sorted(self.source_counts.items())),
                    "country_count": len(self.country_codes),
                    "fields": list(FIELDS),
                    "features": {field: self.config.features_for(field) for field in FIELDS},
                    "min_token_length": self.config.min_token_length,
                    "min_df": self.config.min_df,
                    "max_df_ratio": self.config.max_df_ratio,
                    "sublinear_tf": self.config.sublinear_tf,
                    "max_target_rows": self.config.max_target_rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _normalize_stored_matrix(self, field: str, field_idf: np.ndarray) -> None:
        indptr = np.memmap(
            self.store_dir / f"{field}_indptr.bin", dtype=np.int64, mode="r+"
        )
        indices = np.memmap(
            self.store_dir / f"{field}_indices.bin", dtype=np.int32, mode="r+"
        )
        data = np.memmap(
            self.store_dir / f"{field}_data.bin", dtype=np.float32, mode="r+"
        )
        for start in range(0, self.row_count, self.config.target_block_rows):
            stop = min(start + self.config.target_block_rows, self.row_count)
            begin, end = int(indptr[start]), int(indptr[stop])
            if end == begin:
                continue
            local = np.asarray(indptr[start : stop + 1], dtype=np.int64) - begin
            _row_normalize(
                data[begin:end],
                local,
                field_idf,
                np.asarray(indices[begin:end]),
                self.config.sublinear_tf,
            )
        for array in (indptr, indices, data):
            array.flush()
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def close(self) -> None:
        """Release any still-open writer handles."""
        for blob in self._blobs.values():
            blob.close()
        for vector in self._vectors.values():
            vector.close()
        for writers in self._csr.values():
            for writer in writers.values():
                writer.close()


class TargetStore:
    """Read-only, memory-mapped view of a built S2+S3 candidate universe."""

    def __init__(self, store_dir: FilePath) -> None:
        self.store_dir = Path(store_dir)
        metadata_path = self.store_dir / "meta.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Target store metadata not found: {metadata_path}. Build the store "
                "first with build_target_store()."
            )
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.n_rows = int(self.meta["row_count"])
        self._blobs = {
            name: _BlobReader(self.store_dir, name)
            for name in ("entity_id", "name", "address")
        }
        self._country_code = _VectorReader(self.store_dir, "country_code", "int16")
        self._source_code = _VectorReader(self.store_dir, "source_code", "int8")
        self._name_missing = _VectorReader(self.store_dir, "name_missing", "int8")
        self._address_missing = _VectorReader(
            self.store_dir, "address_missing", "int8"
        )
        self._indptr = {
            field: _VectorReader(self.store_dir, f"{field}_indptr", "int64")
            for field in FIELDS
        }
        self._indices = {
            field: _VectorReader(self.store_dir, f"{field}_indices", "int32")
            for field in FIELDS
        }
        self._data = {
            field: _VectorReader(self.store_dir, f"{field}_data", "float32")
            for field in FIELDS
        }
        self._idf = {
            field: _map_dtype(
                self.store_dir / f"{field}_idf.bin", np.dtype(np.float32)
            )
            for field in FIELDS
        }
        self._country_vocab = {
            code: country
            for country, code in json.loads(
                (self.store_dir / "country_vocab.json").read_text(encoding="utf-8")
            ).items()
        }
        self._sublinear_tf = bool(self.meta.get("sublinear_tf", True))
        self._min_token_length = int(self.meta.get("min_token_length", 1))

    def features_for(self, field: str) -> int:
        """Return the hashed term-space size stored for ``field``."""
        return int(self.meta["features"][field])

    def entity_id(self, index: int) -> str:
        """Return the original S2/S3 entity ID for a target row."""
        return self._blobs["entity_id"].get(index).decode("utf-8")

    def text(self, field: str, index: int) -> bytes:
        """Return the normalized text bytes for a target row and field."""
        return self._blobs[field].get(index)

    def text_length(self, field: str, indices: np.ndarray) -> np.ndarray:
        """Return normalized text lengths for a vector of target rows."""
        return self._blobs[field].length(indices)

    def country_code(self, indices: np.ndarray) -> np.ndarray:
        """Return country codes for a vector of target rows."""
        return self._country_code.gather(indices, dtype="int32")

    def source_code(self, indices: np.ndarray) -> np.ndarray:
        """Return Source 2/Source 3 codes for a vector of target rows."""
        return self._source_code.gather(indices, dtype="int32")

    def country(self, code: int) -> str:
        """Return the open-set country string for a stored country code."""
        return self._country_vocab.get(int(code), "")

    def text_missing(self, field: str, indices: np.ndarray) -> np.ndarray:
        """Return the raw-missing flags for a vector of target rows.

        These are stored rather than recomputed from the normalized text so the
        pair features use the same "raw value was empty" rule on both sides of
        a pair.
        """
        reader = self._name_missing if field == "name" else self._address_missing
        return reader.gather(indices, dtype="int8")

    def _gather_texts(self, field: str, indices: np.ndarray) -> list[str]:
        """Return normalized text for many rows.

        Rows are split into maximal runs of consecutive indices so each run costs
        one contiguous byte gather instead of one lookup per value. Retrieval
        results are naturally clustered, so the runs are short, but correctness
        never depends on them being adjacent.
        """
        rows = np.asarray(indices, dtype=np.int64)
        if rows.size == 0:
            return []
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        blob = self._blobs[field]
        breaks = np.flatnonzero(np.diff(sorted_rows) != 1) + 1
        run_starts = np.concatenate(([0], breaks)).tolist()
        run_stops = np.concatenate((breaks, [sorted_rows.size])).tolist()
        texts = ["" for _ in range(rows.size)]
        for run_start, run_stop in zip(run_starts, run_stops):
            run = sorted_rows[run_start:run_stop]
            lengths = blob.length(run).tolist()
            buffer = blob.read(int(blob.starts(run)[0]), int(sum(lengths)))
            cursor = 0
            for position, length in zip(
                order[run_start:run_stop].tolist(), lengths
            ):
                texts[position] = buffer[cursor : cursor + length].decode("utf-8")
                cursor += length
        return texts

    def entity_ids(self, indices: np.ndarray) -> list[str]:
        """Return the original S2/S3 entity IDs for a vector of target rows."""
        return self._gather_texts("entity_id", indices)

    def texts(self, field: str, indices: np.ndarray) -> list[str]:
        """Return the normalized text for a vector of target rows and ``field``."""
        return self._gather_texts(field, indices)

    def block(self, start: int, stop: int, field: str) -> sparse.csr_matrix:
        """Return a writable CSR slice of target rows for ``field``.

        The returned matrix owns its value/index arrays. That matters because
        SciPy updates sparse products in place, so handing out the raw
        memory-mapped buffers would fault the process.
        """
        if start < 0 or stop > self.n_rows or start > stop:
            raise ValueError(f"Invalid target block [{start}, {stop}).")
        features = self.features_for(field)
        indptr = np.asarray(
            self._indptr[field].gather(np.arange(start, stop + 1), "int64")
        )
        begin, end = int(indptr[0]), int(indptr[-1])
        local = (indptr - begin).astype(np.int64)
        indices = np.array(
            self._indices[field].gather(np.arange(begin, end), "int32"), copy=True
        )
        data = np.array(
            self._data[field].gather(np.arange(begin, end), "float32"), copy=True
        )
        return sparse.csr_matrix((data, indices, local), shape=(stop - start, features))

    def encode_queries(self, texts: Sequence[str], field: str) -> sparse.csr_matrix:
        """Project query texts into the stored TF-IDF space for ``field``."""
        modulus = self.features_for(field)
        indptr, indices, counts, _ = _encode_texts(
            list(texts), modulus, self._min_token_length
        )
        values = _row_normalize(
            counts,
            indptr,
            np.asarray(self._idf[field]),
            indices,
            self._sublinear_tf,
        )
        return sparse.csr_matrix(
            (values, indices, indptr), shape=(len(texts), modulus), copy=False
        )

    def close(self) -> None:
        """Release every memory mapping held by the store."""
        for blob in self._blobs.values():
            blob.close()
        for reader in (
            self._country_code,
            self._source_code,
            self._name_missing,
            self._address_missing,
        ):
            reader.close()
        for group in (self._indptr, self._indices, self._data):
            for reader in group.values():
                reader.close()
        for array in self._idf.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def build_target_store(
    source_paths: Sequence[tuple[FilePath, str]],
    store_dir: FilePath,
    config: TfidfConfig,
    progress: Callable[[str], None] | None = None,
) -> TargetStore:
    """Stream S2/S3 TSVs into a TF-IDF target store and return it opened for reads."""
    writer = TargetStoreWriter(store_dir, config)
    report = progress or (lambda message: None)
    try:
        for path, expected_source in source_paths:
            kept = 0
            for chunk in iter_source_chunks(
                path,
                chunksize=config.source_chunk_rows,
                expected_source=expected_source,
            ):
                kept += writer.add_prepared_frame(
                    prepare_source(chunk, expected_source=expected_source)
                )
                report(
                    f"    {Path(path).name}: {kept:,} target rows indexed "
                    f"({writer.row_count:,} total)"
                )
                if (
                    config.max_target_rows is not None
                    and writer.row_count >= config.max_target_rows
                ):
                    break
            if kept == 0 and writer.row_count == 0:
                raise ValueError(
                    f"No rows were read from {Path(path).name} before the "
                    f"max_target_rows cap of {config.max_target_rows:,} was reached."
                )
        writer.finalize()
    except BaseException:
        writer.close()
        raise
    report(
        f"  target store ready: {writer.row_count:,} rows, "
        f"{len(writer.country_codes):,} distinct country values"
    )
    return TargetStore(store_dir)


@dataclass(frozen=True)
class CandidateBatch:
    """Flattened Top-K retrieval output for a batch of Source 1 records.

    ``source1_index`` maps a flat pair back to its row in the batch, and
    ``pair_offsets`` slices the flat arrays into one contiguous run per Source 1
    record. ``name_rank``/``address_rank`` are 1-based retrieval ranks and are
    zero when a candidate was not retrieved by that field, which is what lets a
    smaller K be derived from a single larger-K retrieval pass.
    """

    source1_entity_ids: tuple[str, ...]
    source1_index: np.ndarray
    pair_offsets: np.ndarray
    target_index: np.ndarray
    name_score: np.ndarray
    address_score: np.ndarray
    name_rank: np.ndarray
    address_rank: np.ndarray

    @property
    def pair_count(self) -> int:
        """Total number of deduplicated candidate pairs in the batch."""
        return int(self.target_index.shape[0])

    def candidate_counts(self) -> np.ndarray:
        """Return the deduplicated candidate count for every Source 1 record."""
        return np.diff(np.asarray(self.pair_offsets, dtype=np.int64))

    def restrict_to_k(self, k: int) -> "CandidateBatch":
        """Return the union of the Top-K name and Top-K address retrievals."""
        if k < 1:
            raise ValueError("k must be positive.")
        keep = (
            ((self.name_rank > 0) & (self.name_rank <= k))
            | ((self.address_rank > 0) & (self.address_rank <= k))
        )
        positions = np.flatnonzero(keep)
        source1_index = np.asarray(self.source1_index)[positions]
        counts = np.bincount(
            source1_index, minlength=len(self.source1_entity_ids)
        ).astype(np.int64)
        pair_offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=pair_offsets[1:])
        return CandidateBatch(
            source1_entity_ids=self.source1_entity_ids,
            source1_index=source1_index.astype(np.int32),
            pair_offsets=pair_offsets,
            target_index=np.asarray(self.target_index)[positions].astype(np.int32),
            name_score=np.asarray(self.name_score)[positions],
            address_score=np.asarray(self.address_score)[positions],
            name_rank=np.asarray(self.name_rank)[positions].astype(np.int32),
            address_rank=np.asarray(self.address_rank)[positions].astype(np.int32),
        )


def _top_k_indices(
    store: TargetStore,
    queries: sparse.csr_matrix,
    k: int,
    field: str,
    target_block_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Top-K target indices and scores per query row without dense all-pairs.

    Target rows are consumed in blocks, so the dense similarity block is at most
    ``queries.shape[0] x target_block_rows`` wide and is released immediately
    after a running Top-K merge. A block whose largest score cannot beat the
    current K-th best score is skipped outright.
    """
    row_count = int(queries.shape[0])
    if store.n_rows == 0 or row_count == 0:
        return (
            np.zeros((row_count, 0), dtype=np.int32),
            np.zeros((row_count, 0), dtype=np.float32),
        )
    width = int(min(k, store.n_rows))
    best_scores = np.full((row_count, width), -np.inf, dtype=np.float32)
    best_indices = np.full((row_count, width), -1, dtype=np.int32)
    for start in range(0, store.n_rows, target_block_rows):
        stop = min(start + target_block_rows, store.n_rows)
        scores = (queries @ store.block(start, stop, field).T.tocsr()).toarray()
        if float(scores.max()) <= float(best_scores[:, -1].min()):
            continue
        block_width = int(min(width, scores.shape[1]))
        partition = np.argpartition(-scores, block_width - 1, axis=1)[:, :block_width]
        block_scores = np.take_along_axis(scores, partition, axis=1)
        order = np.argsort(-block_scores, axis=1, kind="stable")
        block_scores = np.take_along_axis(block_scores, order, axis=1)
        block_indices = np.take_along_axis(partition, order, axis=1) + start
        positive = block_scores > 0.0
        block_scores = np.where(positive, block_scores, -np.inf).astype(np.float32)
        block_indices = np.where(positive, block_indices, -1).astype(np.int32)
        merged_scores = np.concatenate([best_scores, block_scores], axis=1)
        merged_indices = np.concatenate([best_indices, block_indices], axis=1)
        keep = np.argsort(-merged_scores, axis=1, kind="stable")[:, :width]
        best_scores = np.ascontiguousarray(
            np.take_along_axis(merged_scores, keep, axis=1)
        )
        best_indices = np.ascontiguousarray(
            np.take_along_axis(merged_indices, keep, axis=1)
        )
    return best_indices, best_scores


_WORKER_STORE: TargetStore | None = None


def _worker_init_store(store_dir: str) -> None:
    """Open the read-only memory-mapped store once per worker process."""
    global _WORKER_STORE
    _WORKER_STORE = TargetStore(store_dir)


def _worker_top_k(
    payload: tuple[list[str], str, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Score one query batch against the whole target store inside a worker."""
    texts, field, k_max, target_block_rows = payload
    store = _WORKER_STORE
    if store is None:  # pragma: no cover - defensive
        raise RuntimeError("worker store was not initialized")
    return _top_k_indices(
        store,
        store.encode_queries(texts, field),
        k_max,
        field,
        target_block_rows,
    )


class TfidfBlockingIndex:
    """Union of Top-K name TF-IDF and Top-K address TF-IDF retrievals."""

    def __init__(
        self,
        store: TargetStore,
        k_max: int,
        target_block_rows: int = 8_192,
        query_batch_rows: int = 1_024,
        workers: int = 1,
    ) -> None:
        if k_max < 1:
            raise ValueError("k_max must be positive.")
        if target_block_rows < 1 or query_batch_rows < 1:
            raise ValueError("target_block_rows and query_batch_rows must be positive.")
        if workers < 1:
            raise ValueError("workers must be positive.")
        self.store = store
        self.k_max = k_max
        self.target_block_rows = target_block_rows
        self.query_batch_rows = query_batch_rows
        self.workers = int(workers)

    def _top_k_for_field(
        self, texts: Sequence[str], field: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run batched Top-K retrieval for one field, chunking the query side.

        Retrieval is embarrassingly parallel over query batches: each batch needs
        the full target store but nothing from its siblings. When ``workers`` is
        greater than one the batches are farmed out to a process pool, and every
        worker re-opens the read-only memory-mapped store in its own address
        space, so the target side is never copied or pickled. The store files are
        mapped read-only, which makes concurrent readers safe.
        """
        total = len(texts)
        indices = np.zeros((total, self.k_max), dtype=np.int32)
        scores = np.zeros((total, self.k_max), dtype=np.float32)
        starts = list(range(0, total, self.query_batch_rows))
        if self.workers > 1 and len(starts) > 1:
            payloads = [
                (
                    list(texts[start : min(start + self.query_batch_rows, total)]),
                    field,
                    self.k_max,
                    self.target_block_rows,
                )
                for start in starts
            ]
            context = multiprocessing.get_context(
                "spawn" if os.name == "nt" else "fork"
            )
            with context.Pool(
                processes=min(self.workers, len(starts)),
                initializer=_worker_init_store,
                initargs=(str(self.store.store_dir),),
            ) as pool:
                results = pool.imap(_worker_top_k, payloads, chunksize=1)
                for start, (batch_indices, batch_scores) in zip(starts, results):
                    stop = min(start + self.query_batch_rows, total)
                    indices[start:stop] = batch_indices
                    scores[start:stop] = batch_scores
            return indices, scores
        for start in starts:
            stop = min(start + self.query_batch_rows, total)
            batch_indices, batch_scores = _top_k_indices(
                self.store,
                self.store.encode_queries(texts[start:stop], field),
                self.k_max,
                field,
                self.target_block_rows,
            )
            indices[start:stop] = batch_indices
            scores[start:stop] = batch_scores
        return indices, scores

    def retrieve(self, source1_frame) -> CandidateBatch:
        """Retrieve the deduplicated Top-K union for a prepared Source 1 frame."""
        source1_ids = tuple(str(value) for value in source1_frame["entity_id"])
        name_indices, name_scores = self._top_k_for_field(
            [str(value) for value in source1_frame[FIELD_COLUMNS["name"]]], "name"
        )
        address_indices, address_scores = self._top_k_for_field(
            [str(value) for value in source1_frame[FIELD_COLUMNS["address"]]], "address"
        )
        return merge_channel_retrievals(
            source1_ids,
            name_indices,
            name_scores,
            address_indices,
            address_scores,
            universe_size=self.store.n_rows,
        )


def merge_channel_retrievals(
    source1_entity_ids: Sequence[str],
    name_indices: np.ndarray,
    name_scores: np.ndarray,
    address_indices: np.ndarray,
    address_scores: np.ndarray,
    universe_size: int,
) -> CandidateBatch:
    """Deduplicate the name and address Top-K lists into one candidate set.

    Ranks are 1-based within each channel and zero where a channel did not
    retrieve the candidate, so :meth:`CandidateBatch.restrict_to_k` can rebuild
    the union of Top-K retrievals for any smaller K without re-querying the index.
    """
    row_count = len(source1_entity_ids)
    if universe_size <= 0:
        return _empty_batch(source1_entity_ids)
    name_ranks = _rank_positions(name_indices)
    address_ranks = _rank_positions(address_indices)
    flat_indices = np.concatenate([name_indices.ravel(), address_indices.ravel()])
    flat_rows = np.concatenate(
        [
            np.repeat(np.arange(row_count, dtype=np.int64), name_indices.shape[1]),
            np.repeat(np.arange(row_count, dtype=np.int64), address_indices.shape[1]),
        ]
    )
    flat_scores = np.concatenate([name_scores.ravel(), address_scores.ravel()])
    flat_ranks = np.concatenate([name_ranks.ravel(), address_ranks.ravel()])
    from_name = np.concatenate(
        [
            np.ones(name_indices.size, dtype=bool),
            np.zeros(address_indices.size, dtype=bool),
        ]
    )
    valid = (flat_indices >= 0) & (flat_scores > 0.0)
    flat_indices = flat_indices[valid].astype(np.int64)
    flat_rows = flat_rows[valid]
    flat_scores = flat_scores[valid]
    flat_ranks = flat_ranks[valid]
    from_name = from_name[valid]
    if flat_indices.size == 0:
        return _empty_batch(source1_entity_ids)

    keys = flat_rows * np.int64(universe_size) + flat_indices
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
    )
    group_sizes = np.diff(np.concatenate((boundaries, [sorted_keys.shape[0]])))
    entry_group = np.empty(sorted_keys.shape[0], dtype=np.int64)
    entry_group[order] = np.repeat(np.arange(boundaries.shape[0]), group_sizes)
    unique_keys = sorted_keys[boundaries]
    group_count = boundaries.shape[0]

    name_score = np.zeros(group_count, dtype=np.float32)
    address_score = np.zeros(group_count, dtype=np.float32)
    name_rank = np.zeros(group_count, dtype=np.int32)
    address_rank = np.zeros(group_count, dtype=np.int32)
    name_entries = np.flatnonzero(from_name)
    address_entries = np.flatnonzero(~from_name)
    name_score[entry_group[name_entries]] = flat_scores[name_entries]
    name_rank[entry_group[name_entries]] = flat_ranks[name_entries]
    address_score[entry_group[address_entries]] = flat_scores[address_entries]
    address_rank[entry_group[address_entries]] = flat_ranks[address_entries]

    source1_index = (unique_keys // np.int64(universe_size)).astype(np.int32)
    counts = np.bincount(source1_index, minlength=row_count).astype(np.int64)
    pair_offsets = np.zeros(row_count + 1, dtype=np.int64)
    np.cumsum(counts, out=pair_offsets[1:])
    return CandidateBatch(
        source1_entity_ids=tuple(source1_entity_ids),
        source1_index=source1_index,
        pair_offsets=pair_offsets,
        target_index=(unique_keys % np.int64(universe_size)).astype(np.int32),
        name_score=name_score,
        address_score=address_score,
        name_rank=name_rank,
        address_rank=address_rank,
    )


def _empty_batch(source1_entity_ids: Sequence[str]) -> CandidateBatch:
    """Return a batch with no candidate pairs but one slot per Source 1 record."""
    row_count = len(source1_entity_ids)
    return CandidateBatch(
        source1_entity_ids=tuple(source1_entity_ids),
        source1_index=np.zeros(0, dtype=np.int32),
        pair_offsets=np.zeros(row_count + 1, dtype=np.int64),
        target_index=np.zeros(0, dtype=np.int32),
        name_score=np.zeros(0, dtype=np.float32),
        address_score=np.zeros(0, dtype=np.float32),
        name_rank=np.zeros(0, dtype=np.int32),
        address_rank=np.zeros(0, dtype=np.int32),
    )


def _rank_positions(indices: np.ndarray) -> np.ndarray:
    """Return 1-based ranks for retrieval positions, zero for empty slots."""
    ranks = np.zeros(indices.shape, dtype=np.int32)
    valid = indices >= 0
    if valid.any():
        positions = np.arange(1, indices.shape[1] + 1, dtype=np.int32)
        ranks[valid] = np.tile(positions, (indices.shape[0], 1))[valid]
    return ranks


def summarize_candidate_counts(counts: Sequence[int] | np.ndarray) -> dict[str, float]:
    """Summarize per-Source-1 candidate counts and the resulting reduction ratio.

    ``candidate_reduction_ratio`` is the fraction of the dense all-pairs space
    removed by blocking; it needs the target universe size, supplied separately
    by the caller through :func:`candidate_reduction_ratio`.
    """
    values = np.asarray(counts, dtype=np.int64)
    if values.size == 0:
        return {
            "candidate_pair_count": 0,
            "candidate_count_average": 0.0,
            "candidate_count_median": 0.0,
            "candidate_count_p95": 0.0,
            "candidate_count_max": 0,
            "candidate_count_min": 0,
        }
    return {
        "candidate_pair_count": int(values.sum()),
        "candidate_count_average": float(values.mean()),
        "candidate_count_median": float(np.median(values)),
        "candidate_count_p95": float(np.percentile(values, 95, method="nearest")),
        "candidate_count_max": int(values.max()),
        "candidate_count_min": int(values.min()),
    }


def candidate_reduction_ratio(
    source1_count: int, target_count: int, candidate_pair_count: int
) -> float:
    """Return the share of the all-pairs space removed by candidate retrieval."""
    possible_pairs = int(source1_count) * int(target_count)
    if possible_pairs <= 0:
        return 0.0
    return float(1.0 - candidate_pair_count / possible_pairs)


def iter_source1_batches(
    path: FilePath,
    query_batch_rows: int,
    chunk_size: int,
    limit: int | None = None,
) -> Iterable[object]:
    """Yield prepared Source 1 frames in retrieval-sized batches.

    Used by the experiment runner so that the S1 side also streams instead of
    being loaded whole. ``limit`` bounds the number of Source 1 records and is
    what the smoke-test mode uses.
    """
    if query_batch_rows < 1:
        raise ValueError("query_batch_rows must be positive.")
    emitted = 0
    for chunk in iter_source_chunks(path, chunksize=chunk_size, expected_source="S1"):
        prepared = prepare_source(chunk, expected_source="S1")
        if limit is not None:
            remaining = limit - emitted
            if remaining <= 0:
                break
            if len(prepared) > remaining:
                prepared = prepared.iloc[:remaining]
        for start in range(0, len(prepared), query_batch_rows):
            yield prepared.iloc[start : start + query_batch_rows]
        emitted += len(prepared)


def _p95_nearest_rank(values: np.ndarray) -> int:
    """Return the nearest-rank 95th percentile of an integer array."""
    if values.size == 0:
        return 0
    ordered = np.sort(values)
    return int(ordered[max(0, math.ceil(0.95 * ordered.shape[0]) - 1)])
