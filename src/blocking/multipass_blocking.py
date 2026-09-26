"""Deterministic multi-pass blocking for business-record entity resolution."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from src.preprocessing import prepare_source, validate_source

BLOCKING_PASSES = (
    "exact_name_country",
    "sorted_name_country",
    "name_token_country",
    "name_prefix_country",
    "address_number_country",
)
_NUMBER_PATTERN = re.compile(r"\d+")
_TARGET_COLUMNS = (
    "entity_id",
    "source",
    "name_norm",
    "name_sorted",
    "address_norm",
    "address_number_signature",
    "country_norm",
    "name_missing",
    "address_missing",
)


@dataclass(frozen=True)
class BlockingConfig:
    """Controls deterministic key generation and frequent-key pruning."""

    min_name_length: int = 4
    min_token_length: int = 4
    min_address_digits: int = 2
    max_bucket_size: int = 100
    use_country: bool = True

    def __post_init__(self) -> None:
        if self.min_name_length < 1 or self.min_token_length < 1:
            raise ValueError("Name and token minimum lengths must be positive.")
        if self.min_address_digits < 1 or self.max_bucket_size < 1:
            raise ValueError("Address digit and bucket limits must be positive.")


@dataclass
class CandidateResult:
    """Candidate pairs and counts from each pass, retaining singleton S1 rows."""

    pairs: pd.DataFrame
    candidate_pairs: pd.DataFrame
    diagnostics: dict[str, object]


def _value(record: Mapping[str, object], primary: str, legacy: str) -> str:
    value = record.get(primary, record.get(legacy, ""))
    if value is None or pd.isna(value):
        return ""
    return str(value)


def make_blocking_keys(
    record: Mapping[str, object], config: BlockingConfig | None = None
) -> list[tuple[str, str]]:
    """Return unique `(pass_name, key)` values for a prepared source row."""
    settings = config or BlockingConfig()
    name = _value(record, "name_norm", "norm_name")
    address = _value(record, "address_norm", "norm_address")
    country = _value(record, "country_norm", "norm_country")
    country_part = country if settings.use_country else "*"
    keys: set[tuple[str, str]] = set()

    if len(name) >= settings.min_name_length:
        keys.add(("exact_name_country", f"{country_part}\x1f{name}"))
        sorted_name = _value(record, "name_sorted", "name_norm")
        if sorted_name:
            keys.add(("sorted_name_country", f"{country_part}\x1f{sorted_name}"))
        else:
            keys.add(
                ("sorted_name_country", f"{country_part}\x1f{' '.join(sorted(name.split()))}")
            )
        tokens = {
            token
            for token in name.split()
            if len(token) >= settings.min_token_length
        }
        for token in tokens:
            keys.add(("name_token_country", f"{country_part}\x1f{token}"))
        first_token = name.split()[0]
        if len(first_token) >= settings.min_token_length:
            keys.add(
                ("name_prefix_country", f"{country_part}\x1f{first_token[:4]}")
            )

    number_signature = _value(
        record, "address_number_signature", "address_number_signature"
    )
    if not number_signature:
        number_signature = "-".join(_NUMBER_PATTERN.findall(address))
    digit_count = sum(character.isdigit() for character in number_signature)
    if digit_count >= settings.min_address_digits:
        keys.add(
            ("address_number_country", f"{country_part}\x1f{number_signature}")
        )
    return sorted(keys)


class InMemoryBlockingIndex:
    """In-memory inverted index for experiments whose target rows fit in RAM."""

    def __init__(
        self, targets: pd.DataFrame, config: BlockingConfig | None = None
    ) -> None:
        self.config = config or BlockingConfig()
        validate_source(targets)
        self.targets = prepare_source(targets) if "name_norm" not in targets else targets
        self.target_records = {
            str(row["entity_id"]): row
            for _, row in self.targets.iterrows()
        }
        key_records: dict[tuple[str, str], set[str]] = defaultdict(set)
        for entity_id, row in self.target_records.items():
            for pass_name, block_key in make_blocking_keys(row, self.config):
                key_records[(pass_name, block_key)].add(entity_id)
        self.key_records = dict(key_records)
        self.skipped_keys = {
            key
            for key, ids in self.key_records.items()
            if len(ids) > self.config.max_bucket_size
        }

    def candidates(
        self, source1_record: Mapping[str, object]
    ) -> dict[str, set[str]]:
        """Return candidate ID to responsible blocking passes for one S1 row."""
        candidates: dict[str, set[str]] = defaultdict(set)
        for key in make_blocking_keys(source1_record, self.config):
            if key in self.skipped_keys:
                continue
            for entity_id in self.key_records.get(key, ()):
                candidates[entity_id].add(key[0])
        return dict(candidates)


class SQLiteBlockingIndex:
    """Disk-backed inverted index for large target corpora."""

    def __init__(
        self,
        database_path: str | Path,
        config: BlockingConfig | None = None,
    ) -> None:
        self.config = config or BlockingConfig()
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.executescript(
            """
            CREATE TABLE targets (
                entity_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                name_norm TEXT NOT NULL,
                name_sorted TEXT NOT NULL,
                address_norm TEXT NOT NULL,
                address_number_signature TEXT NOT NULL,
                country_norm TEXT NOT NULL,
                name_missing INTEGER NOT NULL,
                address_missing INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE blocking_keys (
                pass_name TEXT NOT NULL,
                block_key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (pass_name, block_key, entity_id)
            ) WITHOUT ROWID;
            """
        )
        self._finished = False

    def add_targets(self, targets: pd.DataFrame) -> None:
        """Add a prepared S2 or S3 chunk and its keys to the index."""
        if self._finished:
            raise RuntimeError("Cannot add targets after finish() has been called.")
        validate_source(targets)
        if "name_norm" not in targets:
            targets = prepare_source(targets)
        missing = [column for column in _TARGET_COLUMNS if column not in targets]
        if missing:
            raise ValueError(f"Prepared targets lack required fields: {missing}")

        columns = list(_TARGET_COLUMNS)
        record_rows: list[tuple[object, ...]] = []
        key_rows: set[tuple[str, str, str]] = set()
        for values in targets.loc[:, columns].itertuples(index=False, name=None):
            record = dict(zip(columns, values))
            entity_id = str(record["entity_id"])
            record_rows.append(
                (
                    entity_id,
                    str(record["source"]),
                    str(record["name_norm"]),
                    str(record["name_sorted"]),
                    str(record["address_norm"]),
                    str(record["address_number_signature"]),
                    str(record["country_norm"]),
                    int(bool(record["name_missing"])),
                    int(bool(record["address_missing"])),
                )
            )
            key_rows.update(
                (pass_name, block_key, entity_id)
                for pass_name, block_key in make_blocking_keys(record, self.config)
            )
        self.connection.executemany(
            "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            record_rows,
        )
        self.connection.executemany(
            "INSERT INTO blocking_keys VALUES (?, ?, ?)", key_rows
        )

    def finish(self) -> None:
        """Finalize frequency counts used to skip overly broad keys."""
        if self._finished:
            return
        self.connection.execute(
            """
            CREATE TABLE key_sizes (
                pass_name TEXT NOT NULL,
                block_key TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                PRIMARY KEY (pass_name, block_key)
            ) WITHOUT ROWID
            """
        )
        self.connection.execute(
            """
            INSERT INTO key_sizes
            SELECT pass_name, block_key, COUNT(*)
            FROM blocking_keys
            GROUP BY pass_name, block_key
            """
        )
        self.connection.commit()
        self._finished = True

    def candidates(
        self, source1_record: Mapping[str, object]
    ) -> dict[str, set[str]]:
        """Lookup one S1 row and return target IDs with their contributing passes."""
        if not self._finished:
            raise RuntimeError("Call finish() before looking up candidates.")
        keys = make_blocking_keys(source1_record, self.config)
        if not keys:
            return {}
        key_predicate = ",".join("(?, ?)" for _ in keys)
        parameters: list[object] = [self.config.max_bucket_size]
        for pass_name, block_key in keys:
            parameters.extend((pass_name, block_key))
        rows = self.connection.execute(
            f"""
            SELECT k.entity_id, k.pass_name
            FROM blocking_keys AS k
            JOIN key_sizes AS s USING (pass_name, block_key)
            WHERE s.record_count <= ?
              AND (k.pass_name, k.block_key) IN ({key_predicate})
            """,
            parameters,
        )
        candidates: dict[str, set[str]] = defaultdict(set)
        for entity_id, pass_name in rows:
            candidates[entity_id].add(pass_name)
        return dict(candidates)

    def get_targets(self, entity_ids: Iterable[str]) -> dict[str, dict[str, object]]:
        """Fetch prepared target text/features for candidate IDs."""
        ids = list(dict.fromkeys(entity_ids))
        found: dict[str, dict[str, object]] = {}
        for start in range(0, len(ids), 500):
            subset = ids[start : start + 500]
            placeholders = ",".join("?" for _ in subset)
            rows = self.connection.execute(
                f"SELECT * FROM targets WHERE entity_id IN ({placeholders})",
                subset,
            )
            found.update(
                (str(row[0]), dict(zip(_TARGET_COLUMNS, row)))
                for row in rows
            )
        return found

    def target_counts(self) -> dict[str, int]:
        """Return target-row counts by source."""
        return {
            str(source): int(count)
            for source, count in self.connection.execute(
                "SELECT source, COUNT(*) FROM targets GROUP BY source"
            )
        }

    def close(self) -> None:
        """Close the SQLite database connection."""
        self.connection.close()

    def __enter__(self) -> "SQLiteBlockingIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> CandidateResult:
    """Generate the deterministic union of pass candidates for every S1 row."""
    validate_source(source1, expected_source="S1")
    validate_source(source2, expected_source="S2")
    validate_source(source3, expected_source="S3")
    settings = config or BlockingConfig()
    prepared_s1 = prepare_source(source1, expected_source="S1")
    targets = pd.concat(
        [
            prepare_source(source2, expected_source="S2"),
            prepare_source(source3, expected_source="S3"),
        ],
        ignore_index=True,
    )
    index = InMemoryBlockingIndex(targets, settings)

    pair_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    pass_counts: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for row in prepared_s1.to_dict(orient="records"):
        entity_id = str(row["entity_id"])
        candidates = index.candidates(row)
        candidate_ids = sorted(candidates)
        candidate_counts.append(len(candidate_ids))
        for candidate_id in candidate_ids:
            passes = sorted(candidates[candidate_id])
            pair_rows.append(
                {
                    "source1_entity_id": entity_id,
                    "candidate_entity_id": candidate_id,
                    "blocking_passes": passes,
                }
            )
            pass_counts.update(passes)
        candidate_rows.append(
            {
                "source1_entity_id": entity_id,
                "candidate_entity_ids": ",".join(candidate_ids),
            }
        )

    possible_pairs = len(prepared_s1) * len(targets)
    counts = pd.Series(candidate_counts, dtype="int64")
    diagnostics: dict[str, object] = {
        "source1_count": len(prepared_s1),
        "target_count": len(targets),
        "candidate_pair_count": len(pair_rows),
        "possible_pair_count": possible_pairs,
        "candidate_reduction_ratio": (
            1.0 - len(pair_rows) / possible_pairs if possible_pairs else 0.0
        ),
        "candidate_count_mean": float(counts.mean()) if len(counts) else 0.0,
        "candidate_count_median": float(counts.median()) if len(counts) else 0.0,
        "candidate_count_p95": float(counts.quantile(0.95)) if len(counts) else 0.0,
        "candidate_count_max": int(counts.max()) if len(counts) else 0,
        "blocking_pass_pair_counts": dict(sorted(pass_counts.items())),
        "skipped_frequent_key_count": len(index.skipped_keys),
    }
    return CandidateResult(
        pairs=pd.DataFrame(
            pair_rows,
            columns=(
                "source1_entity_id",
                "candidate_entity_id",
                "blocking_passes",
            ),
        ),
        candidate_pairs=pd.DataFrame(
            candidate_rows,
            columns=("source1_entity_id", "candidate_entity_ids"),
        ),
        diagnostics=diagnostics,
    )
