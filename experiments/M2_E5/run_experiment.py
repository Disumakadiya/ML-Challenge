"""Sparse TF-IDF name/address blocking plus LightGBM pair matching for M2_E5.

The pipeline has three clearly separated stages:

1. **Build** - ``src.blocking.tfidf_blocking`` streams every Source 2 and Source 3
   record of a split into one memory-mapped TF-IDF target store, treating S2 and
   S3 as a single candidate universe while keeping the original entity IDs.
2. **Retrieve** - the store answers batched Top-K queries over the normalized
   ``business_name`` and ``business_address`` columns. The name and address
   rankings are unioned and deduplicated once at ``K = max(k_values)``; every
   smaller K is then derived from the stored per-channel ranks, so the K sweep
   costs a single retrieval pass.
3. **Match** - LightGBM scores the deduplicated candidates. Pair features reuse
   the M2_E4 feature builder for the shared similarity columns and add the E5
   TF-IDF similarities and retrieval ranks.

Model and K selection use the training split only: a deterministic,
match-count-stratified Source 1 validation fold is held out, the K that
maximizes validation macro F0.5 is selected, and the test labels are never
read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.M2_E4.run_experiment import FEATURE_NAMES as E4_FEATURE_NAMES
from experiments.M2_E4.run_experiment import pair_features as e4_pair_features
from src.blocking.tfidf_blocking import (
    SOURCE_CODES,
    UNKNOWN_SOURCE_CODE,
    TfidfBlockingIndex,
    TfidfConfig,
    TargetStore,
    build_target_store,
    candidate_reduction_ratio,
    iter_source1_batches,
    summarize_candidate_counts,
)
from src.preprocessing import (
    _prepare_ground_truth,
    iter_source_chunks,
    iter_tsv_chunks,
    prepare_source,
    write_tsv,
)

EXPERIMENT_ID = "M2_E5"
DEFAULT_K_VALUES = (5, 10, 20, 30, 50)
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
_DIGITS = re.compile(r"\d+")
_SOURCE_NAMES = {code: source for source, code in SOURCE_CODES.items()}
_SOURCE_NAMES[UNKNOWN_SOURCE_CODE] = ""

# The M2_E4 feature builder is reused as-is. Its ``pass_*`` columns describe the
# E4 SQLite blocking keys, which do not exist in E5, so they are dropped.
_SHARED_FEATURES = tuple(
    name for name in E4_FEATURE_NAMES if not name.startswith("pass_")
)
_SHARED_POSITIONS = tuple(E4_FEATURE_NAMES.index(name) for name in _SHARED_FEATURES)
FEATURE_NAMES = _SHARED_FEATURES + (
    "tfidf_name_similarity",
    "tfidf_address_similarity",
    "retrieval_rank",
    "retrieval_rank_name",
    "retrieval_rank_address",
    "retrieval_channel_name",
    "retrieval_channel_address",
)
E5_FEATURES = FEATURE_NAMES[len(_SHARED_FEATURES) :]


def _stable_hash64(value: str) -> int:
    """Return a stable 64-bit hash so splits never depend on the interpreter."""
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible K sweep, validation split, and resource controls."""

    train_dir: Path
    output_dir: Path
    test_dir: Path = Path("student_resource/dataset/test")
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    validation_fraction: float = 0.2
    random_seed: int = 42
    max_train_entities: int = 25_000
    negative_samples_per_entity: int = 30
    chunk_size: int = 20_000
    name_features: int = 1 << 22
    address_features: int = 1 << 22
    min_token_length: int = 1
    source_chunk_rows: int = 100_000
    target_block_rows: int = 8_192
    query_batch_rows: int = 1_024
    workers: int = 1
    max_target_rows: int | None = None
    prediction_threshold: float = 0.5
    num_boost_round: int = 300
    smoke_test: bool = False
    smoke_source1_rows: int = 1_500
    smoke_filler_rows: int = 3_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_values", tuple(sorted(set(self.k_values))))
        if not self.k_values or min(self.k_values) < 1:
            raise ValueError("k_values must contain positive integers.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.max_train_entities < 1 or self.negative_samples_per_entity < 1:
            raise ValueError("Sampling limits must be positive.")
        if self.chunk_size < 1 or self.target_block_rows < 1 or self.query_batch_rows < 1:
            raise ValueError("Batch and block sizes must be positive.")
        if self.workers < 1:
            raise ValueError("workers must be positive.")
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must be between 0 and 1.")
        if self.num_boost_round < 1:
            raise ValueError("num_boost_round must be positive.")

    @property
    def k_max(self) -> int:
        """Largest K in the sweep; the single retrieval pass uses this K."""
        return max(self.k_values)

    @property
    def gather_fanout(self) -> int:
        """How many query batches a single ``retrieve`` call receives.

        ``TfidfBlockingIndex.retrieve`` splits whatever frame it is given back into
        ``query_batch_rows`` batches, and only then hands them to the worker pool.
        A frame exactly one batch wide would therefore leave every worker idle, so
        Source 1 frames are gathered this many batches wide. The fanout is capped
        at 8 so one gather cannot produce an oversized candidate frame: it is
        bounded by both the pool size and the read chunk.
        """
        return max(1, min(self.workers, 8))

    @property
    def gather_rows(self) -> int:
        """Source 1 rows per retrieval call, clamped to the read chunk size."""
        return min(self.query_batch_rows * self.gather_fanout, self.chunk_size)

    def tfidf_config(self) -> TfidfConfig:
        """Return the blocking configuration for the store build and retrieval."""
        return TfidfConfig(
            name_features=self.name_features,
            address_features=self.address_features,
            min_token_length=self.min_token_length,
            source_chunk_rows=self.source_chunk_rows,
            target_block_rows=self.target_block_rows,
            query_batch_rows=self.query_batch_rows,
            max_target_rows=self.max_target_rows,
        )


class GroundTruthIndex:
    """Columnar, memory-bounded Source 1 ground-truth index with fixed folds.

    A Python ``dict`` of 2.2M rows and their match lists costs several hundred
    megabytes, so the labels are kept as parallel NumPy arrays plus one packed
    byte blob. Lookups go through a sorted 64-bit hash of the Source 1 ID, which
    is a pure NumPy ``searchsorted`` and therefore cheap enough to call once per
    retrieved batch.
    """

    def __init__(
        self,
        path: Path,
        config: ExperimentConfig,
        create_validation_split: bool = True,
    ) -> None:
        source1_ids: list[bytes] = []
        strata: list[int] = []
        split_hashes: list[int] = []
        match_offsets: list[int] = [0]
        id_offsets: list[int] = [0]
        blob = bytearray()
        for chunk in iter_tsv_chunks(path, chunksize=config.chunk_size):
            prepared = _prepare_ground_truth(chunk)
            for source1_id, match_ids in prepared[
                ["source1_entity_id", "matched_entity_ids"]
            ].itertuples(index=False, name=None):
                entity_id = str(source1_id)
                encoded = entity_id.encode("utf-8")
                source1_ids.append(encoded)
                match_list = list(match_ids)
                strata.append(
                    0 if not match_list else 1 if len(match_list) == 1 else 2
                )
                split_hashes.append(
                    _stable_hash64(f"{config.random_seed}:{entity_id}")
                )
                for match_id in match_list:
                    blob += match_id.encode("utf-8")
                    id_offsets.append(len(blob))
                match_offsets.append(len(id_offsets) - 1)

        self._ids = np.array(source1_ids, dtype="S24")
        self._strata = np.asarray(strata, dtype=np.int8)
        self._split_hashes = np.asarray(split_hashes, dtype=np.uint64)
        self._match_offsets = np.asarray(match_offsets, dtype=np.int64)
        self._id_offsets = np.asarray(id_offsets, dtype=np.int64)
        self._blob = bytes(blob)
        self.row_count = int(self._ids.shape[0])
        hashes = np.array(
            [_stable_hash64(value.decode("utf-8")) for value in self._ids],
            dtype=np.uint64,
        )
        self._order = np.argsort(hashes, kind="stable")
        self._sorted_hashes = hashes[self._order]
        self._is_validation = np.zeros(self.row_count, dtype=bool)
        if create_validation_split:
            self._assign_folds(config.validation_fraction)

    def _assign_folds(self, validation_fraction: float) -> None:
        """Hold out a deterministic, match-count-stratified validation fold."""
        for stratum in np.unique(self._strata):
            rows = np.flatnonzero(self._strata == stratum)
            holdout = min(
                len(rows), int(math.floor(len(rows) * validation_fraction + 0.5))
            )
            if holdout == 0:
                continue
            ordered = rows[np.argsort(self._split_hashes[rows], kind="stable")]
            self._is_validation[ordered[:holdout]] = True

    def source1_id(self, row: int) -> str:
        """Return the Source 1 entity ID of a ground-truth row."""
        return self._ids[row].decode("utf-8")

    @property
    def stratum_counts(self) -> dict[str, int]:
        """Return the singleton / one-match / multi-match row counts."""
        return {
            "singletons": int((self._strata == 0).sum()),
            "one_match": int((self._strata == 1).sum()),
            "multi_match": int((self._strata == 2).sum()),
        }

    def validation_rows(self) -> np.ndarray:
        """Return the ground-truth rows assigned to the validation fold."""
        return np.flatnonzero(self._is_validation)

    def training_rows(self, limit: int) -> np.ndarray:
        """Return a deterministic, stratum-balanced sample of fit-fold rows.

        Within a stratum the rows with the largest split hash are taken, which is
        the complement of the smallest-hash validation selection, so the two sets
        are disjoint by construction.
        """
        available = np.flatnonzero(~self._is_validation)
        if available.size == 0:
            return available
        if limit >= available.size:
            return available
        strata = self._strata[available]
        counts = {
            int(stratum): int((strata == stratum).sum())
            for stratum in np.unique(strata)
        }
        total = sum(counts.values())
        quotas = {
            stratum: int(limit * count / total) for stratum, count in counts.items()
        }
        remainder = limit - sum(quotas.values())
        for stratum in sorted(
            counts,
            key=lambda key: (-(limit * counts[key] / total - quotas[key]), key),
        )[:remainder]:
            quotas[stratum] += 1
        chosen: list[np.ndarray] = []
        for stratum, quota in sorted(quotas.items()):
            if quota == 0:
                continue
            rows = available[strata == stratum]
            ordered = rows[
                np.argsort(-self._split_hashes[rows].astype(np.int64), kind="stable")
            ]
            chosen.append(ordered[:quota])
        return np.sort(np.concatenate(chosen)) if chosen else available[:0]

    def _rows_for(self, source1_ids: Sequence[str]) -> np.ndarray:
        """Return ground-truth row indices for the given Source 1 IDs."""
        if self.row_count == 0:
            return np.zeros(len(source1_ids), dtype=np.int64)
        hashes = np.array(
            [_stable_hash64(value) for value in source1_ids], dtype=np.uint64
        )
        positions = np.searchsorted(self._sorted_hashes, hashes)
        clipped = np.clip(positions, 0, self.row_count - 1)
        found = self._sorted_hashes[clipped] == hashes
        rows = self._order[clipped]
        if not bool(found.all()):
            missing = [
                value
                for value, ok in zip(source1_ids, found.tolist())
                if not ok
            ][:5]
            raise ValueError(
                "Ground truth is missing Source 1 ID(s) present in the source "
                f"file, e.g. {missing}"
            )
        return rows.astype(np.int64)

    def matches(self, source1_ids: Sequence[str]) -> list[frozenset[str]]:
        """Return the true match ID set for each Source 1 ID in the batch."""
        rows = self._rows_for(source1_ids)
        results: list[frozenset[str]] = []
        for row in rows.tolist():
            first = int(self._match_offsets[row])
            last = int(self._match_offsets[row + 1])
            if last == first:
                results.append(frozenset())
                continue
            packed = self._blob[
                int(self._id_offsets[first]) : int(self._id_offsets[last])
            ]
            results.append(frozenset(packed.decode("utf-8").split(",")))
        return results

    def matches_for_rows(self, rows: np.ndarray) -> list[frozenset[str]]:
        """Return the true match ID set for explicit ground-truth row indices."""
        return self.matches([self.source1_id(int(row)) for row in rows])


@dataclass(frozen=True)
class EncodedBatch:
    """One retrieved batch reduced to a flat feature matrix for every K sweep.

    Features are built once at ``K = k_max``. :meth:`pair_mask` then reproduces
    the deduplicated Top-K union for any smaller K using the per-channel ranks,
    which is what makes the sweep a single pass over the target store.
    """

    source1_entity_ids: tuple[str, ...]
    pair_offsets: np.ndarray
    features: np.ndarray
    target_entity_ids: tuple[str, ...]
    truth: tuple[frozenset[str], ...]
    name_rank: np.ndarray
    address_rank: np.ndarray

    def pair_mask(self, k: int) -> np.ndarray:
        """Return the pair mask for the union of the Top-K name/address lists."""
        name_hit = (self.name_rank > 0) & (self.name_rank <= k)
        address_hit = (self.address_rank > 0) & (self.address_rank <= k)
        return name_hit | address_hit

    def row_of_pair(self) -> np.ndarray:
        """Return the Source 1 row index of every pair in the flat arrays."""
        return np.repeat(
            np.arange(len(self.source1_entity_ids), dtype=np.int64),
            np.diff(np.asarray(self.pair_offsets, dtype=np.int64)),
        )

    def masked_offsets(self, k: int) -> np.ndarray:
        """Return the per-row offsets into the masked pair arrays for ``k``.

        This is what lets the writers and the scorecards walk a K-restricted batch
        in one linear pass instead of rescanning the pairs for every row.
        """
        row_of_pair = self.row_of_pair()
        counts = np.bincount(
            row_of_pair[self.pair_mask(k)],
            minlength=len(self.source1_entity_ids),
        )
        offsets = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return offsets


def _target_record(
    name_norm: str,
    address_norm: str,
    country_norm: str,
    source: str,
    name_missing: bool,
    address_missing: bool,
) -> dict[str, object]:
    """Return the M2_E4 feature-builder view of one stored target record.

    ``name_sorted`` and ``address_number_signature`` are derived here from the
    stored normalized text so the shared E4 columns are computed identically on
    both sides of a pair.
    """
    return {
        "name_norm": name_norm,
        "address_norm": address_norm,
        "country_norm": country_norm,
        "name_sorted": " ".join(sorted(name_norm.split())),
        "address_number_signature": "-".join(_DIGITS.findall(address_norm)),
        "name_missing": name_missing,
        "address_missing": address_missing,
        "source": source,
    }


def pair_feature_row(
    source1: dict[str, object],
    target: dict[str, object],
    name_score: float,
    address_score: float,
    name_rank: int,
    address_rank: int,
) -> list[float]:
    """Return one E5 feature row, reusing the shared M2_E4 similarity columns."""
    shared = e4_pair_features(source1, target, ())
    ranks = [rank for rank in (name_rank, address_rank) if rank > 0]
    retrieval_rank = min(ranks) if ranks else 0
    values = [shared[position] for position in _SHARED_POSITIONS]
    values.extend(
        (
            float(name_score),
            float(address_score),
            float(retrieval_rank),
            float(name_rank),
            float(address_rank),
            float(name_rank > 0),
            float(address_rank > 0),
        )
    )
    return values


def _group_features(
    store: TargetStore,
    source1: dict[str, object],
    batch,
    row: int,
) -> tuple[np.ndarray, list[float], list[list[float]]]:
    """Return the candidate rows, their TF-IDF scores, and the feature rows."""
    first = int(batch.pair_offsets[row])
    last = int(batch.pair_offsets[row + 1])
    targets = np.asarray(batch.target_index[first:last], dtype=np.int64)
    if targets.size == 0:
        return targets, [], []
    names = store.texts("name", targets)
    addresses = store.texts("address", targets)
    countries = [store.country(code) for code in store.country_code(targets).tolist()]
    sources = [_SOURCE_NAMES[code] for code in store.source_code(targets).tolist()]
    name_missing = store.text_missing("name", targets).astype(bool)
    address_missing = store.text_missing("address", targets).astype(bool)
    name_scores = np.asarray(batch.name_score[first:last], dtype=np.float64)
    address_scores = np.asarray(batch.address_score[first:last], dtype=np.float64)
    name_ranks = np.asarray(batch.name_rank[first:last], dtype=np.int64)
    address_ranks = np.asarray(batch.address_rank[first:last], dtype=np.int64)
    features: list[list[float]] = []
    for position in range(targets.size):
        record = _target_record(
            names[position],
            addresses[position],
            countries[position],
            sources[position],
            bool(name_missing[position]),
            bool(address_missing[position]),
        )
        features.append(
            pair_feature_row(
                source1,
                record,
                name_scores[position],
                address_scores[position],
                int(name_ranks[position]),
                int(address_ranks[position]),
            )
        )
    return targets, name_scores.tolist(), features


def encode_batch(
    store: TargetStore,
    index: TfidfBlockingIndex,
    frame,
    truth: Sequence[frozenset[str]] | None,
) -> EncodedBatch:
    """Retrieve one prepared Source 1 batch and build its K-max feature matrix."""
    candidates = index.retrieve(frame)
    row_count = len(frame)
    features: list[list[float]] = []
    target_ids: list[str] = []
    for row in range(row_count):
        source1 = frame.iloc[row].to_dict()
        targets, _, rows = _group_features(store, source1, candidates, row)
        if targets.size:
            target_ids.extend(store.entity_ids(targets))
        features.extend(rows)
    matrix = (
        np.asarray(features, dtype=np.float32)
        if features
        else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    return EncodedBatch(
        source1_entity_ids=tuple(candidates.source1_entity_ids),
        pair_offsets=np.asarray(candidates.pair_offsets, dtype=np.int64),
        features=matrix,
        target_entity_ids=tuple(target_ids),
        truth=tuple(truth) if truth is not None else tuple(
            frozenset() for _ in range(row_count)
        ),
        name_rank=np.asarray(candidates.name_rank, dtype=np.int64),
        address_rank=np.asarray(candidates.address_rank, dtype=np.int64),
    )


def _iter_source1_frames(path: Path, config: ExperimentConfig, limit: int | None = None):
    """Yield prepared Source 1 frames of ``gather_rows`` rows."""
    return iter_source1_batches(
        path,
        query_batch_rows=config.gather_rows,
        chunk_size=config.chunk_size,
        limit=limit,
    )


def _select_frames(
    path: Path,
    config: ExperimentConfig,
    selected: set[str],
    limit: int | None = None,
):
    """Yield only the prepared Source 1 frames whose IDs are in ``selected``."""
    for frame in _iter_source1_frames(path, config, limit=limit):
        mask = frame["entity_id"].isin(selected)
        if not bool(mask.any()):
            continue
        filtered = frame.loc[mask]
        for start in range(0, len(filtered), config.query_batch_rows):
            yield filtered.iloc[start : start + config.query_batch_rows]


def _train_booster(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, config: ExperimentConfig):
    """Fit one deterministic LightGBM binary booster on E5 candidate features."""
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is required for M2_E5. Install requirements.txt first."
        ) from error
    dataset = lgb.Dataset(
        features,
        label=labels,
        weight=weights,
        feature_name=list(FEATURE_NAMES),
    )
    return lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "seed": config.random_seed,
            "feature_fraction_seed": config.random_seed,
            "bagging_seed": config.random_seed,
            "data_random_seed": config.random_seed,
            "num_threads": max(1, os.cpu_count() or 1),
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=config.num_boost_round,
    )


def _sample_training_pairs(
    batch: EncodedBatch,
    k: int,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return capped training features, labels, and weights for one K.

    Every retrieved positive is kept. Negatives are capped per Source 1 record
    in a deterministic hash order and weighted by the inverse sampling fraction,
    so the capped sample stays an unbiased estimate of the full candidate set.
    """
    mask = batch.pair_mask(k)
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return (
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.float32),
        )
    target_ids = np.asarray(batch.target_entity_ids, dtype=object)
    positives = np.array(
        [
            position
            for position, row in zip(
                positions.tolist(), batch.row_of_pair()[positions].tolist()
            )
            if str(target_ids[position]) in batch.truth[row]
        ],
        dtype=np.int64,
    )
    positive_set = set(positives.tolist())
    keep_offsets: list[int] = []
    keep_weights: list[float] = []
    for row, run in enumerate(_runs(batch, k)):
        negatives = [
            offset
            for offset in range(run[0], run[1])
            if int(positions[offset]) not in positive_set
        ]
        if len(negatives) > config.negative_samples_per_entity:
            negatives = sorted(
                negatives,
                key=lambda offset: _stable_hash64(
                    f"{config.random_seed}:{batch.source1_entity_ids[row]}:"
                    f"{target_ids[positions[offset]]}"
                ),
            )[: config.negative_samples_per_entity]
        weight = (
            (run[1] - run[0] - len(negatives)) / len(negatives) if negatives else 1.0
        )
        keep_offsets.extend(negatives)
        keep_weights.extend([max(1.0, weight)] * len(negatives))
    negative_offsets = np.asarray(keep_offsets, dtype=np.int64)
    selected = np.concatenate([positives, positions[negative_offsets]])
    labels = np.concatenate(
        [
            np.ones(positives.size, dtype=np.int8),
            np.zeros(negative_offsets.size, dtype=np.int8),
        ]
    )
    weights = np.concatenate(
        [
            np.ones(positives.size, dtype=np.float32),
            np.asarray(keep_weights, dtype=np.float32),
        ]
    )
    return batch.features[selected], labels, weights


def _runs(batch: EncodedBatch, k: int) -> list[tuple[int, int]]:
    """Return the ``[start, stop)`` run of masked pairs for each Source 1 row."""
    offsets = batch.masked_offsets(k)
    return [
        (int(offsets[row]), int(offsets[row + 1]))
        for row in range(len(batch.source1_entity_ids))
    ]


def _train_models_per_k(
    store: TargetStore,
    source1_path: Path,
    config: ExperimentConfig,
    truth: GroundTruthIndex,
    selected_ids: set[str],
    index: TfidfBlockingIndex,
) -> tuple[dict[int, object], dict[int, dict[str, object]]]:
    """Retrieve the sampled fit rows once and fit one booster per K."""
    per_batch: list[EncodedBatch] = []
    matched_rows = 0
    for frame in _select_frames(source1_path, config, selected_ids):
        source1_ids = [str(value) for value in frame["entity_id"]]
        batch = encode_batch(store, index, frame, truth.matches(source1_ids))
        per_batch.append(batch)
        matched_rows += len(batch.source1_entity_ids)
    report = {
        "training_source1_rows": matched_rows,
        "training_candidate_pairs_k_max": sum(
            int(batch.pair_offsets[-1]) for batch in per_batch
        ),
    }
    models: dict[int, object] = {}
    stats: dict[int, dict[str, object]] = {}
    for k in config.k_values:
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for batch in per_batch:
            block_features, block_labels, block_weights = _sample_training_pairs(
                batch, k, config
            )
            if block_labels.size:
                features.append(block_features)
                labels.append(block_labels)
                weights.append(block_weights)
        if not labels:
            raise ValueError(
                f"No training candidates were retrieved for K={k}. Increase "
                "max_train_entities or review the TF-IDF configuration."
            )
        matrix = np.concatenate(features)
        label_array = np.concatenate(labels)
        weight_array = np.concatenate(weights)
        if int(np.unique(label_array).size) < 2:
            raise ValueError(
                f"The K={k} training candidates do not contain both positive and "
                "negative labels; increase max_train_entities or lower K."
            )
        models[k] = _train_booster(matrix, label_array, weight_array, config)
        stats[k] = {
            "training_candidate_pairs": int(label_array.size),
            "training_positive_pairs": int(label_array.sum()),
        }
    report.update(
        {
            "training_sample_rows": len(selected_ids),
            "per_k": {str(k): stats[k] for k in config.k_values},
        }
    )
    return models, report


def _f05(true_ids: frozenset[str], predicted_ids: frozenset[str]) -> float:
    """Return the F0.5 score (precision weighted twice as heavily as recall)."""
    if not true_ids and not predicted_ids:
        return 1.0
    if not true_ids or not predicted_ids:
        return 0.0
    hits = len(true_ids & predicted_ids)
    precision = hits / len(predicted_ids)
    recall = hits / len(true_ids)
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


class _KScorecard:
    """Streaming accumulator of blocking and matching metrics for one K."""

    def __init__(self, k: int, source1_count: int, target_count: int) -> None:
        self.k = k
        self.source1_count = source1_count
        self.target_count = target_count
        self.counts: list[int] = []
        self.true_matches = 0
        self.recalled_matches = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.f05_values: list[float] = []
        self.precision_values: list[float] = []
        self.recall_values: list[float] = []
        self.singletons = 0
        self.singleton_correct = 0
        self.predicted_pairs = 0

    def add(
        self,
        truth: frozenset[str],
        candidates: frozenset[str],
        predicted: frozenset[str],
    ) -> None:
        """Fold one Source 1 record into the scorecard."""
        self.counts.append(len(candidates))
        self.true_matches += len(truth)
        self.recalled_matches += len(truth & candidates)
        hits = len(truth & predicted)
        self.true_positive += hits
        self.false_positive += len(predicted - truth)
        self.false_negative += len(truth - predicted)
        self.predicted_pairs += len(predicted)
        self.f05_values.append(_f05(truth, predicted))
        self.precision_values.append(
            hits / len(predicted) if predicted else float(not truth)
        )
        self.recall_values.append(
            hits / len(truth) if truth else float(not predicted)
        )
        if not truth:
            self.singletons += 1
            self.singleton_correct += int(not predicted)

    def summary(self) -> dict[str, object]:
        """Return the JSON-ready metric block for this K."""
        counts = summarize_candidate_counts(self.counts)
        possible_pairs = self.source1_count * self.target_count
        return {
            "k": self.k,
            "source1_rows": self.source1_count,
            "target_rows": self.target_count,
            "possible_pair_count": possible_pairs,
            "candidate_pair_count": counts["candidate_pair_count"],
            "candidate_count_average": counts["candidate_count_average"],
            "candidate_count_median": counts["candidate_count_median"],
            "candidate_count_p95": counts["candidate_count_p95"],
            "candidate_count_max": counts["candidate_count_max"],
            "candidate_count_min": counts["candidate_count_min"],
            "candidate_reduction_ratio": candidate_reduction_ratio(
                self.source1_count, self.target_count, counts["candidate_pair_count"]
            ),
            "blocking_recall": (
                self.recalled_matches / self.true_matches
                if self.true_matches
                else 0.0
            ),
            "blocking_recalled_match_ids": self.recalled_matches,
            "blocking_total_match_ids": self.true_matches,
            "matching_precision_micro": (
                self.true_positive / (self.true_positive + self.false_positive)
                if self.true_positive + self.false_positive
                else 0.0
            ),
            "matching_recall_micro": (
                self.true_positive / (self.true_positive + self.false_negative)
                if self.true_positive + self.false_negative
                else 0.0
            ),
            "matching_precision_macro": (
                statistics.fmean(self.precision_values) if self.precision_values else 0.0
            ),
            "matching_recall_macro": (
                statistics.fmean(self.recall_values) if self.recall_values else 0.0
            ),
            "macro_f0_5": (
                statistics.fmean(self.f05_values) if self.f05_values else 0.0
            ),
            "true_positive_pairs": self.true_positive,
            "false_positive_pairs": self.false_positive,
            "false_negative_pairs": self.false_negative,
            "predicted_pair_count": self.predicted_pairs,
            "validation_singletons": self.singletons,
            "singleton_accuracy": (
                self.singleton_correct / self.singletons if self.singletons else 0.0
            ),
        }


def _predict(boster, features: np.ndarray) -> np.ndarray:
    """Return predicted match probabilities, or an empty array for no pairs."""
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(booster.predict(features), dtype=np.float64)


def _select_best_k(summaries: dict[int, dict[str, object]]) -> int:
    """Pick K by validation macro F0.5 with deterministic tie-breaks."""
    return max(
        summaries,
        key=lambda k: (
            float(summaries[k]["macro_f0_5"]),
            float(summaries[k]["blocking_recall"]),
            float(summaries[k]["matching_recall_macro"]),
            k,
        ),
    )


def _write_config(config: ExperimentConfig) -> None:
    """Persist the resolved run configuration next to the metrics."""
    payload = asdict(config)
    for key in ("train_dir", "output_dir", "test_dir"):
        payload[key] = str(getattr(config, key))
    payload["k_values"] = list(config.k_values)
    (config.output_dir / "config.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required data file(s): " + ", ".join(missing))


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Sweep K on the training split, select K, and write validation artifacts."""
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source1_path = config.train_dir / "train_source1.tsv"
    source2_path = config.train_dir / "train_source2.tsv"
    source3_path = config.train_dir / "train_source3.tsv"
    ground_truth_path = config.train_dir / "train_ground_truth.tsv"
    _require_files((source1_path, source2_path, source3_path, ground_truth_path))
    _write_config(config)
    report = lambda message: print(message, flush=True)
    tfidf_config = config.tfidf_config()

    report("[1/5] indexing training ground truth")
    truth = GroundTruthIndex(ground_truth_path, config)
    selected_rows = truth.training_rows(config.max_train_entities)
    selected_ids = {truth.source1_id(int(row)) for row in selected_rows}
    report(
        f"      {truth.row_count:,} labelled rows; "
        f"{len(selected_ids):,} sampled for the fit"
    )

    store_dir = Path(
        tempfile.mkdtemp(prefix="m2e5-train-store-", dir=config.output_dir.parent)
    ) / "store"
    report("[2/5] building the training TF-IDF target store (S2 + S3)")
    build_started = time.perf_counter()
    store = build_target_store(
        [(source2_path, "S2"), (source3_path, "S3")],
        store_dir,
        tfidf_config,
        progress=report,
    )
    build_seconds = time.perf_counter() - build_started
    report(
        f"      target store: {store.n_rows:,} rows, "
        f"country values: {int(store.meta['country_count']):,}"
    )

    try:
        index = TfidfBlockingIndex(
            store,
            k_max=config.k_max,
            target_block_rows=config.target_block_rows,
            query_batch_rows=config.query_batch_rows,
            workers=config.workers,
        )
        report(f"[3/5] retrieving candidates and fitting LightGBM for K in {list(config.k_values)}")
        train_started = time.perf_counter()
        models, train_stats = _train_models_per_k(
            store, source1_path, config, truth, selected_ids, index
        )
        train_seconds = time.perf_counter() - train_started
        report(f"      fitted {len(models)} models in {train_seconds:,.1f}s")

        report("[4/5] evaluating the validation fold for every K")
        eval_started = time.perf_counter()
        summaries, temp_outputs = _evaluate_validation(
            store, index, source1_path, config, truth, models
        )
        eval_seconds = time.perf_counter() - eval_started
        best_k = _select_best_k(summaries)
        report(f"      selected K={best_k} (macro F0.5 = {summaries[best_k]['macro_f0_5']:.4f})")
        candidate_path = config.output_dir / "candidate_pairs.tsv"
        results_path = config.output_dir / "matching_results.tsv"
        _promote_outputs(temp_outputs, best_k, candidate_path, results_path)
    finally:
        store.close()
        shutil.rmtree(store_dir.parent, ignore_errors=True)

    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "split": "train_validation",
        "selected_k": best_k,
        "k_values": list(config.k_values),
        "validation_fraction": config.validation_fraction,
        "validation_seed": config.random_seed,
        "validation_stratum_counts": truth.stratum_counts,
        "training_sample_rows": len(selected_ids),
        "training_candidate_pairs_k_max": train_stats[
            "training_candidate_pairs_k_max"
        ],
        "training_per_k": train_stats["per_k"],
        "negative_samples_per_entity": config.negative_samples_per_entity,
        "target_row_count": int(store.n_rows),
        "target_source_counts": store.meta.get("source_counts", {}),
        "target_country_count": int(store.meta["country_count"]),
        "tfidf": {
            "name_features": config.name_features,
            "address_features": config.address_features,
            "min_token_length": config.min_token_length,
            "sublinear_tf": True,
            "max_target_rows": config.max_target_rows,
        },
        "retrieval": {
            "k_max": config.k_max,
            "target_block_rows": config.target_block_rows,
            "query_batch_rows": config.query_batch_rows,
        },
        "model": (
            f"LightGBM native binary booster; {config.num_boost_round} trees; "
            f"fixed threshold {config.prediction_threshold}"
        ),
        "features": list(FEATURE_NAMES),
        "shared_features_with_m2_e4": list(_SHARED_FEATURES),
        "new_features": list(E5_FEATURES),
        "per_k": {str(k): summaries[k] for k in config.k_values},
        "prediction_threshold": config.prediction_threshold,
        "smoke_test": config.smoke_test,
        "split_note": (
            "Source 1 rows are assigned to a match-count-stratified validation "
            "fold by blake2b(seed, entity_id); the smallest split hashes per "
            "stratum form validation and the fit sample takes the largest, so the "
            "two sets are disjoint. K is selected on validation macro F0.5 only. "
            "No test data is read."
        ),
        "store_build_runtime_seconds": build_seconds,
        "model_fit_runtime_seconds": train_seconds,
        "validation_runtime_seconds": eval_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "candidate_pairs_path": str(candidate_path),
        "matching_results_path": str(results_path),
        "artifact_note": (
            "candidate_pairs.tsv and matching_results.tsv are validation-fold "
            "artifacts at the selected K, not test submissions."
        ),
    }
    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def _open_tsv(path: Path, header: Sequence[str]):
    """Open a TSV for streaming writes with the given header."""
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(list(header))
    return handle, writer


def _evaluate_validation(
    store: TargetStore,
    index: TfidfBlockingIndex,
    source1_path: Path,
    config: ExperimentConfig,
    truth: GroundTruthIndex,
    models: dict[int, object],
) -> tuple[dict[int, dict[str, object]], dict[int, tuple[Path, Path]]]:
    """Score the validation fold at every K in one retrieval pass.

    Each K gets its own temporary TSV pair so the selected K can be promoted
    without a second full pass over the target store.
    """
    temporary = Path(tempfile.mkdtemp(prefix="m2e5-sweep-", dir=config.output_dir.parent))
    handles: dict[int, tuple] = {}
    outputs: dict[int, tuple[Path, Path]] = {}
    scorecards: dict[int, _KScorecard] = {}
    validation_ids = {
        truth.source1_id(int(row)) for row in truth.validation_rows()
    }
    for k in config.k_values:
        candidate_path = temporary / f"candidates_k{k}.tsv"
        results_path = temporary / f"results_k{k}.tsv"
        outputs[k] = (candidate_path, results_path)
        handles[k] = (
            _open_tsv(candidate_path, CANDIDATE_HEADER),
            _open_tsv(results_path, MATCHING_HEADER),
        )
    processed = 0
    for frame in _select_frames(source1_path, config, validation_ids):
        source1_ids = [str(value) for value in frame["entity_id"]]
        truth_sets = truth.matches(source1_ids)
        batch = encode_batch(store, index, frame, truth_sets)
        for k in config.k_values:
            mask = batch.pair_mask(k)
            features = batch.features[mask]
            probabilities = _predict(models[k], features)
            target_ids = np.asarray(batch.target_entity_ids, dtype=object)[mask]
            runs = _runs(batch, k)
            scorecard = scorecards.setdefault(k, _KScorecard(k, 0, store.n_rows))
            candidate_writer = handles[k][0][1]
            result_writer = handles[k][1][1]
            for row, (first, last) in enumerate(runs):
                candidate_ids = [str(value) for value in target_ids[first:last]]
                predicted = [
                    candidate_id
                    for candidate_id, score in zip(
                        candidate_ids, probabilities[first:last]
                    )
                    if float(score) >= config.prediction_threshold
                ]
                candidate_writer.writerow(
                    [batch.source1_entity_ids[row], ",".join(candidate_ids)]
                )
                result_writer.writerow(
                    [batch.source1_entity_ids[row], ",".join(predicted)]
                )
                scorecard.add(
                    truth_sets[row],
                    frozenset(candidate_ids),
                    frozenset(predicted),
                )
        processed += len(batch.source1_entity_ids)
    for k in config.k_values:
        (candidate_handle, _), (results_handle, _) = handles[k]
        for handle in (candidate_handle, results_handle):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        scorecards[k].source1_count = processed
    summaries = {k: scorecards[k].summary() for k in config.k_values}
    return summaries, outputs


def _promote_outputs(
    outputs: dict[int, tuple[Path, Path]],
    best_k: int,
    candidate_path: Path,
    results_path: Path,
) -> None:
    """Keep the selected K's TSVs, rename them, and delete the other K's files."""
    for k, (source_candidates, source_results) in outputs.items():
        if k == best_k:
            os.replace(source_candidates, candidate_path)
            os.replace(source_results, results_path)
        else:
            source_candidates.unlink(missing_ok=True)
            source_results.unlink(missing_ok=True)
    shutil.rmtree(
        next(iter(outputs.values()))[0].parent, ignore_errors=True
    )


def run_final_inference(
    config: ExperimentConfig,
    test_dir: Path,
    submission_dir: Path,
    k: int | None = None,
) -> dict[str, object]:
    """Fit on the full training labels and write the test submission TSVs."""
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    source1_path = config.train_dir / "train_source1.tsv"
    source2_path = config.train_dir / "train_source2.tsv"
    source3_path = config.train_dir / "train_source3.tsv"
    ground_truth_path = config.train_dir / "train_ground_truth.tsv"
    test_source1 = test_dir / "test_source1.tsv"
    _require_files(
        (
            source1_path,
            source2_path,
            source3_path,
            ground_truth_path,
            test_source1,
            test_dir / "test_source2.tsv",
            test_dir / "test_source3.tsv",
        )
    )
    report = lambda message: print(message, flush=True)
    tfidf_config = config.tfidf_config()
    if k is None:
        k = config.k_max
    if k not in config.k_values:
        k = config.k_max

    report("[1/4] indexing training ground truth (no validation split)")
    truth = GroundTruthIndex(ground_truth_path, config, create_validation_split=False)
    selected_rows = truth.training_rows(config.max_train_entities)
    selected_ids = {truth.source1_id(int(row)) for row in selected_rows}
    report(f"      fitting on {len(selected_ids):,} Source 1 rows")

    train_store_dir = Path(
        tempfile.mkdtemp(prefix="m2e5-fit-store-", dir=submission_dir.parent)
    ) / "store"
    report("[2/4] building the training TF-IDF target store and fitting LightGBM")
    fit_started = time.perf_counter()
    train_store = build_target_store(
        [(source2_path, "S2"), (source3_path, "S3")],
        train_store_dir,
        tfidf_config,
        progress=report,
    )
    try:
        train_index = TfidfBlockingIndex(
            train_store,
            k_max=config.k_max,
            target_block_rows=config.target_block_rows,
            query_batch_rows=config.query_batch_rows,
            workers=config.workers,
        )
        models, train_stats = _train_models_per_k(
            train_store, source1_path, config, truth, selected_ids, train_index
        )
    finally:
        train_store.close()
        shutil.rmtree(train_store_dir.parent, ignore_errors=True)
    fit_seconds = time.perf_counter() - fit_started
    model = models[k]
    report(f"      model ready in {fit_seconds:,.1f}s; releasing the training store")

    test_store_dir = Path(
        tempfile.mkdtemp(prefix="m2e5-test-store-", dir=submission_dir.parent)
    ) / "store"
    report("[3/4] building the test TF-IDF target store (S2 + S3)")
    test_started = time.perf_counter()
    test_store = build_target_store(
        [(test_dir / "test_source2.tsv", "S2"), (test_dir / "test_source3.tsv", "S3")],
        test_store_dir,
        tfidf_config,
        progress=report,
    )
    candidate_path = submission_dir / "candidate_pairs.tsv"
    results_path = submission_dir / "matching_results.tsv"
    candidate_temp = candidate_path.with_suffix(".tsv.tmp")
    results_temp = results_path.with_suffix(".tsv.tmp")
    scorecard = _KScorecard(k, 0, test_store.n_rows)
    source1_count = 0
    try:
        test_index = TfidfBlockingIndex(
            test_store,
            k_max=k,
            target_block_rows=config.target_block_rows,
            query_batch_rows=config.query_batch_rows,
            workers=config.workers,
        )
        report(f"[4/4] retrieving K={k} candidates and scoring every test Source 1 row")
        with (
            _open_tsv(candidate_temp, CANDIDATE_HEADER) as (candidate_file, candidate_writer),
            _open_tsv(results_temp, MATCHING_HEADER) as (results_file, results_writer),
        ):
            for frame in _iter_source1_frames(test_source1, config):
                batch = encode_batch(test_store, test_index, frame, None)
                mask = batch.pair_mask(k)
                features = batch.features[mask]
                probabilities = _predict(model, features)
                target_ids = np.asarray(batch.target_entity_ids, dtype=object)[mask]
                for row, (first, last) in enumerate(_runs(batch, k)):
                    candidate_ids = [str(value) for value in target_ids[first:last]]
                    predicted = [
                        candidate_id
                        for candidate_id, score in zip(
                            candidate_ids, probabilities[first:last]
                        )
                        if float(score) >= config.prediction_threshold
                    ]
                    candidate_writer.writerow(
                        [batch.source1_entity_ids[row], ",".join(candidate_ids)]
                    )
                    results_writer.writerow(
                        [batch.source1_entity_ids[row], ",".join(predicted)]
                    )
                    scorecard.add(
                        frozenset(),
                        frozenset(candidate_ids),
                        frozenset(predicted),
                    )
                source1_count += len(batch.source1_entity_ids)
                if source1_count % (config.gather_rows * 200) < config.gather_rows:
                    report(
                        f"      {source1_count:,} test Source 1 rows scored"
                    )
            for handle in (candidate_file, results_file):
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        test_store.close()
        shutil.rmtree(test_store_dir.parent, ignore_errors=True)
    os.replace(candidate_temp, candidate_path)
    os.replace(results_temp, results_path)
    scorecard.source1_count = source1_count
    inference = scorecard.summary()
    inference.pop("blocking_recall", None)
    inference.pop("blocking_total_match_ids", None)
    metrics = {
        "experiment_id": EXPERIMENT_ID,
        "split": "test",
        "selected_k": k,
        "k_values": list(config.k_values),
        "smoke_test": config.smoke_test,
        "features": list(FEATURE_NAMES),
        "shared_features_with_m2_e4": list(_SHARED_FEATURES),
        "new_features": list(E5_FEATURES),
        "prediction_threshold": config.prediction_threshold,
        "source1_count": source1_count,
        "target_row_count": int(test_store.n_rows),
        "training_source1_sample_rows": len(selected_ids),
        "training_candidate_pairs_k_max": train_stats[
            "training_candidate_pairs_k_max"
        ],
        "training_per_k": train_stats["per_k"],
        "inference": inference,
        "model_fit_runtime_seconds": fit_seconds,
        "test_inference_runtime_seconds": time.perf_counter() - test_started,
        "runtime_seconds": time.perf_counter() - started,
        "candidate_pairs_path": str(candidate_path),
        "matching_results_path": str(results_path),
        "submission_note": (
            "candidate_pairs.tsv and matching_results.tsv are generated from the "
            "test split; every predicted ID is a retrieved candidate and every "
            "test Source 1 ID gets exactly one row. metrics.json is run metadata "
            "and is not a submission TSV."
        ),
    }
    (submission_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def _first_rows(path: Path, source: str, count: int, chunk_size: int) -> pd.DataFrame:
    """Return the first ``count`` rows of a source TSV."""
    collected: list[pd.DataFrame] = []
    total = 0
    for chunk in iter_source_chunks(path, chunksize=chunk_size, expected_source=source):
        collected.append(chunk)
        total += len(chunk)
        if total >= count:
            break
    return pd.concat(collected, ignore_index=True).iloc[:count]


def _matching_rows(
    ground_truth_path: Path, source1_ids: set[str], chunk_size: int
) -> pd.DataFrame:
    """Return the ground-truth rows for the given Source 1 IDs.

    The ground truth file is streamed rather than indexed so smoke mode does not
    pay for the full 2.2M-row label index.
    """
    collected: list[pd.DataFrame] = []
    for chunk in iter_tsv_chunks(ground_truth_path, chunksize=chunk_size):
        prepared = _prepare_ground_truth(chunk)
        selected = prepared[prepared["source1_entity_id"].isin(source1_ids)]
        if not selected.empty:
            collected.append(selected)
    if not collected:
        return pd.DataFrame(
            {
                "source1_entity_id": pd.Series(dtype="string"),
                "matched_entity_ids": pd.Series(dtype="string"),
            }
        )
    truth = pd.concat(collected, ignore_index=True)
    truth["matched_entity_ids"] = truth["matched_entity_ids"].map(
        lambda match_ids: ",".join(match_ids)
    )
    return truth


def _build_smoke_split(
    source_dir: Path,
    destination: Path,
    config: ExperimentConfig,
    split: str,
) -> None:
    """Write a small but self-consistent copy of one real split.

    Simply prefix-capping the target store would make the smoke run useless: an
    S1 entity's true S2/S3 match sits at an arbitrary offset in the file, so a
    prefix keeps almost no positives and no model can be fitted. Instead one
    streaming pass keeps the rows that are actually referenced by the sampled
    ground truth, plus ``smoke_filler_rows`` other rows per target file to give
    retrieval realistic distractors.
    """
    source1 = _first_rows(
        source_dir / f"{split}_source1.tsv", "S1", config.smoke_source1_rows, config.chunk_size
    )
    write_tsv(source1, destination / f"{split}_source1.tsv")
    keep: set[str] = set()
    if split == "train":
        truth = _matching_rows(
            source_dir / "train_ground_truth.tsv",
            set(source1["entity_id"].astype(str)),
            config.chunk_size,
        )
        write_tsv(
            truth[["source1_entity_id", "matched_entity_ids"]],
            destination / "train_ground_truth.tsv",
        )
        for match_ids in truth["matched_entity_ids"]:
            keep.update(match_ids.split(","))
    for source, suffix in (("S2", "2"), ("S3", "3")):
        path = source_dir / f"{split}_source{suffix}.tsv"
        filler = _first_rows(path, source, config.smoke_filler_rows, config.chunk_size)
        keep.update(filler["entity_id"].astype(str))
        rows = [filler]
        for chunk in iter_source_chunks(
            path, chunksize=config.chunk_size, expected_source=source
        ):
            selected = chunk[chunk["entity_id"].isin(keep)]
            if not selected.empty:
                rows.append(selected)
        combined = pd.concat(rows, ignore_index=True).drop_duplicates(
            subset=["entity_id"], keep="first"
        )
        write_tsv(combined, destination / f"{split}_source{suffix}.tsv")


def _build_smoke_dataset(
    source_dir: Path, destination: Path, config: ExperimentConfig, splits: Sequence[str]
) -> dict[str, int]:
    """Build the smoke dataset and return the row count of every written file."""
    counts: dict[str, int] = {}
    for split in splits:
        target = destination / split
        target.mkdir(parents=True, exist_ok=True)
        _build_smoke_split(source_dir / split, target, config, split)
        for path in sorted(target.glob("*.tsv")):
            counts[path.name] = sum(
                1 for _ in path.open(encoding="utf-8")
            ) - 1
    return counts


def _parse_k_values(text: str) -> tuple[int, ...]:
    """Parse a comma-separated K list such as ``5,10,20``."""
    values = tuple(int(part) for part in text.replace(" ", "").split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("--k-values must list at least one integer.")
    return values


TRAIN_FILENAMES = (
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
)
TEST_FILENAMES = (
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
)


def _is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def _download_s3_object(uri: str, destination: Path) -> None:
    """Stream one S3 object to ``destination`` with a resumable partial file.

    ``download_fileobj`` performs the multipart transfer internally, which matters
    because the Source 2/3 TSVs are hundreds of megabytes each. The download lands
    on a ``.part`` file first so an interrupted run never leaves a truncated TSV
    that later stages would happily reuse.
    """
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Reading the dataset from S3 needs boto3. Install it with "
            "`pip install boto3`, or pass a local --train-dir/--test-dir."
        ) from error
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI: {uri!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    client = boto3.client("s3")
    with partial.open("wb") as handle:
        client.download_fileobj(bucket, key, handle)
    os.replace(partial, destination)


def stage_dataset_dir(base: str, stage_root: Path, filenames: Sequence[str]) -> Path:
    """Return a local directory holding ``filenames``, downloading from S3 if needed.

    Everything downstream reads plain local files: the TF-IDF store builder seeks
    through the TSVs, and the ground-truth index memory-maps offsets it computes
    from byte positions. Both need a real seekable path, so S3 inputs are staged
    onto the instance's local disk once and then reused by every later stage.
    Existing complete files are kept, which makes repeated calls cheap.
    """
    if not _is_s3_uri(base):
        local = Path(base)
        if not local.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {local}")
        return local
    # Hash the URI so two buckets with the same trailing path do not collide.
    import hashlib

    token = hashlib.blake2b(base.encode("utf-8"), digest_size=8).hexdigest()
    local = stage_root / f"s3-{token}"
    local.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        destination = local / name
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        uri = f"{base.rstrip('/')}/{name}"
        print(f"staging {uri} -> {destination}", flush=True)
        _download_s3_object(uri, destination)
    return local


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the M2_E5 TF-IDF blocking K sweep, or generate the final test "
            "submission files."
        )
    )
    parser.add_argument(
        "--final-inference",
        action="store_true",
        help="fit from training data and write the test submission TSVs",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help=(
            "project root, or an s3:// prefix; train/ and test/ default to "
            "<data-root>/student_resource/dataset"
        ),
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="train directory holding the S1/S2/S3 TSVs and ground truth; local or s3://",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="test directory holding the S1/S2/S3 TSVs; local or s3://",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--submission-dir", type=str, default=None)
    parser.add_argument(
        "--stage-dir",
        type=str,
        default=None,
        help=(
            "local directory used to stage s3:// inputs; defaults to a "
            "m2e5-s3-stage folder next to the output directory"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="fixed K for final inference; defaults to max(--k-values)",
    )
    parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=DEFAULT_K_VALUES,
        help=f"comma-separated K sweep (default: {','.join(str(k) for k in DEFAULT_K_VALUES)})",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-entities", type=int, default=25_000)
    parser.add_argument("--negative-samples-per-entity", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--target-block-rows", type=int, default=8_192)
    parser.add_argument("--query-batch-rows", type=int, default=1_024)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "processes used to score query batches against the target store; "
            "0 (default) uses every available core, 1 disables multiprocessing"
        ),
    )
    parser.add_argument(
        "--max-target-rows",
        type=int,
        default=None,
        help="cap the indexed S2+S3 rows; useful for quick runs",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "fast end-to-end run over a small dataset derived from the real "
            "train/test files; writes to a temporary directory and deletes it"
        ),
    )
    parser.add_argument(
        "--smoke-source1-rows",
        type=int,
        default=1_500,
        help="Source 1 rows per split in smoke mode (default: %(default)s)",
    )
    parser.add_argument(
        "--smoke-filler-rows",
        type=int,
        default=3_000,
        help="non-match S2/S3 rows per target file in smoke mode (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    dataset_dir = "/".join(
        [args.data_root.rstrip("/"), "student_resource", "dataset"]
    )
    train_base = args.train_dir or f"{dataset_dir}/train"
    test_base = args.test_dir or f"{dataset_dir}/test"
    output_dir = Path(args.output_dir or f"{args.data_root.rstrip('/')}/experiments/{EXPERIMENT_ID}")
    submission_dir = Path(args.submission_dir or f"{args.data_root.rstrip('/')}/output")
    stage_root = Path(args.stage_dir or (output_dir.parent / "m2e5-s3-stage"))
    stage_root.mkdir(parents=True, exist_ok=True)
    # S3 inputs are staged onto local disk first: the store builder seeks through
    # the TSVs and the ground-truth index memory-maps byte offsets derived from
    # them, so both need real seekable files rather than a streaming object.
    train_dir = stage_dataset_dir(train_base, stage_root, TRAIN_FILENAMES)
    test_dir = stage_dataset_dir(test_base, stage_root, TEST_FILENAMES)
    dataset_dir = str(train_dir.parent)
    for name in TRAIN_FILENAMES:
        if not (train_dir / name).is_file():
            raise FileNotFoundError(f"Missing training file: {train_dir / name}")

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(
        f"configuration: workers={workers} of {os.cpu_count()} cores, "
        f"query_batch_rows={args.query_batch_rows}, target_block_rows={args.target_block_rows}",
        flush=True,
    )
    if workers > 1:
        print(
            "retrieval runs a process pool: each worker memory-maps the target "
            "store read-only, so expect peak memory near "
            f"{workers * args.query_batch_rows * args.target_block_rows * 4 / 1e9:.1f} GB "
            "for the in-flight similarity blocks",
            flush=True,
        )

    k_values = tuple(sorted(set(args.k_values) | ({args.k} if args.k else set())))
    chunk_size = args.chunk_size
    source_chunk_rows = 100_000
    if args.smoke_test:
        k_values = (args.k,) if args.k else (5, 20)
        chunk_size = 2_000
        source_chunk_rows = 2_000

    config = ExperimentConfig(
        train_dir=train_dir,
        output_dir=output_dir,
        test_dir=test_dir,
        k_values=k_values,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
        max_train_entities=1_000 if args.smoke_test else args.max_train_entities,
        negative_samples_per_entity=args.negative_samples_per_entity,
        chunk_size=chunk_size,
        source_chunk_rows=source_chunk_rows,
        target_block_rows=args.target_block_rows,
        query_batch_rows=args.query_batch_rows,
        workers=1 if args.smoke_test else workers,
        max_target_rows=None if args.smoke_test else args.max_target_rows,
        prediction_threshold=args.threshold,
        num_boost_round=30 if args.smoke_test else args.num_boost_round,
        smoke_test=args.smoke_test,
        smoke_source1_rows=args.smoke_source1_rows,
        smoke_filler_rows=args.smoke_filler_rows,
    )
    if not args.smoke_test:
        metrics = (
            run_final_inference(config, test_dir, submission_dir, k=args.k)
            if args.final_inference
            else run_experiment(config)
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0

    return _run_smoke(args, config, dataset_dir, output_dir, submission_dir, test_dir)


def _run_smoke(
    args: argparse.Namespace,
    config: ExperimentConfig,
    dataset_dir: Path,
    output_dir: Path,
    submission_dir: Path,
    test_dir: Path,
) -> int:
    """Run the whole pipeline on a temporary dataset derived from the real files."""
    splits = ("train", "test") if args.final_inference else ("train",)
    scratch = Path(tempfile.mkdtemp(prefix="m2e5-smoke-"))
    try:
        smoke_dataset = scratch / "dataset"
        print("smoke mode: building a small dataset from the real files", flush=True)
        counts = _build_smoke_dataset(dataset_dir, smoke_dataset, config, splits)
        for name, rows in counts.items():
            print(f"      {name}: {rows:,} rows", flush=True)
        config = replace(
            config,
            train_dir=smoke_dataset / "train",
            test_dir=smoke_dataset / "test",
            output_dir=scratch / "validation_output",
        )
        if args.final_inference:
            metrics = run_final_inference(
                config, config.test_dir, scratch / "submission", k=args.k
            )
        else:
            metrics = run_experiment(config)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
