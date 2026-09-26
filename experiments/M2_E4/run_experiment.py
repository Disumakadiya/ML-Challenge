"""Full-target, train-only validation run for M2_E4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from src.blocking.multipass_blocking import (
    BLOCKING_PASSES,
    BlockingConfig,
    SQLiteBlockingIndex,
)
from src.preprocessing import (
    _prepare_ground_truth,
    iter_source_chunks,
    iter_tsv_chunks,
    prepare_source,
)

FEATURE_NAMES = (
    "country_equal",
    "name_exact",
    "name_sorted_equal",
    "name_first_token_equal",
    "name_token_jaccard",
    "address_exact",
    "address_token_jaccard",
    "address_number_equal",
    "name_length_difference",
    "address_length_difference",
    "name_missing_either",
    "address_missing_either",
    "pass_exact_name",
    "pass_sorted_name",
    "pass_name_token",
    "pass_name_prefix",
    "pass_address_number",
    "source_is_s2",
    "source_is_s3",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible train-only validation and resource controls."""

    train_dir: Path
    output_dir: Path
    validation_fraction: float = 0.2
    random_seed: int = 42
    max_train_entities: int = 25_000
    negative_samples_per_entity: int = 30
    chunk_size: int = 20_000
    max_bucket_size: int = 100
    prediction_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.max_train_entities < 1:
            raise ValueError("max_train_entities must be positive.")
        if self.negative_samples_per_entity < 1:
            raise ValueError("negative_samples_per_entity must be positive.")
        if self.chunk_size < 1 or self.max_bucket_size < 1:
            raise ValueError("chunk_size and max_bucket_size must be positive.")
        if not 0 <= self.prediction_threshold <= 1:
            raise ValueError("prediction_threshold must be between 0 and 1.")


def _hash_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _open_ground_truth_store(
    connection: sqlite3.Connection,
    ground_truth_path: Path,
    config: ExperimentConfig,
) -> dict[int, int]:
    connection.executescript(
        """
        CREATE TABLE ground_truth (
            source1_entity_id TEXT PRIMARY KEY,
            matched_json TEXT NOT NULL,
            match_count INTEGER NOT NULL,
            stratum INTEGER NOT NULL,
            split_hash TEXT NOT NULL,
            fold TEXT NOT NULL DEFAULT 'train'
        ) WITHOUT ROWID;
        CREATE INDEX ground_truth_stratum_hash
            ON ground_truth (stratum, fold, split_hash);
        CREATE TABLE selected_train (
            source1_entity_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        """
    )
    for chunk in iter_tsv_chunks(ground_truth_path, chunksize=config.chunk_size):
        prepared = _prepare_ground_truth(chunk)
        rows = []
        for source1_id, match_ids in prepared[
            ["source1_entity_id", "matched_entity_ids"]
        ].itertuples(index=False, name=None):
            entity_id = str(source1_id)
            matches = list(match_ids)
            stratum = 0 if not matches else 1 if len(matches) == 1 else 2
            rows.append(
                (
                    entity_id,
                    json.dumps(matches, ensure_ascii=False, separators=(",", ":")),
                    len(matches),
                    stratum,
                    _hash_key(config.random_seed, entity_id),
                )
            )
        connection.executemany(
            """
            INSERT INTO ground_truth
            (source1_entity_id, matched_json, match_count, stratum, split_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

    stratum_counts = {
        int(stratum): int(count)
        for stratum, count in connection.execute(
            "SELECT stratum, COUNT(*) FROM ground_truth GROUP BY stratum"
        )
    }
    for stratum, count in stratum_counts.items():
        validation_count = min(
            count,
            int(math.floor(count * config.validation_fraction + 0.5)),
        )
        connection.execute(
            """
            UPDATE ground_truth SET fold='validation'
            WHERE source1_entity_id IN (
                SELECT source1_entity_id FROM ground_truth
                WHERE stratum=? ORDER BY split_hash LIMIT ?
            )
            """,
            (stratum, validation_count),
        )

    train_counts = {
        int(stratum): int(count)
        for stratum, count in connection.execute(
            """
            SELECT stratum, COUNT(*) FROM ground_truth
            WHERE fold='train' GROUP BY stratum
            """
        )
    }
    total_train = sum(train_counts.values())
    sample_limit = min(config.max_train_entities, total_train)
    quotas = {
        stratum: int(sample_limit * count / total_train)
        for stratum, count in train_counts.items()
    }
    remainder = sample_limit - sum(quotas.values())
    for stratum in sorted(
        train_counts,
        key=lambda key: (
            -(sample_limit * train_counts[key] / total_train - quotas[key]),
            key,
        ),
    )[:remainder]:
        quotas[stratum] += 1
    for stratum, quota in quotas.items():
        connection.execute(
            """
            INSERT INTO selected_train
            SELECT source1_entity_id FROM ground_truth
            WHERE fold='train' AND stratum=?
            ORDER BY split_hash DESC LIMIT ?
            """,
            (stratum, quota),
        )
    connection.commit()
    return stratum_counts


def _selected_ids(
    connection: sqlite3.Connection, selection: str
) -> set[str]:
    if selection == "validation":
        query = (
            "SELECT source1_entity_id FROM ground_truth "
            "WHERE fold='validation'"
        )
    elif selection == "train":
        query = "SELECT source1_entity_id FROM selected_train"
    else:
        raise ValueError(f"Unknown Source 1 selection: {selection}")
    return {str(row[0]) for row in connection.execute(query)}


def _selected_source1_rows(
    path: Path,
    selected_ids: set[str],
    chunksize: int,
) -> Iterator[dict[str, object]]:
    seen = 0
    for chunk in iter_source_chunks(
        path, chunksize=chunksize, expected_source="S1"
    ):
        selected = chunk[chunk["entity_id"].isin(selected_ids)]
        if not selected.empty:
            prepared = prepare_source(selected, expected_source="S1")
            for row in prepared.to_dict(orient="records"):
                seen += 1
                yield row
    if seen != len(selected_ids):
        raise ValueError(
            f"Ground truth/source1 mismatch: found {seen} of "
            f"{len(selected_ids)} selected Source 1 IDs."
        )


def _build_target_index(
    index: SQLiteBlockingIndex,
    train_dir: Path,
    chunksize: int,
) -> dict[str, int]:
    for source, filename in (
        ("S2", "train_source2.tsv"),
        ("S3", "train_source3.tsv"),
    ):
        path = train_dir / filename
        for chunk in iter_source_chunks(
            path, chunksize=chunksize, expected_source=source
        ):
            index.add_targets(prepare_source(chunk, expected_source=source))
    index.finish()
    return index.target_counts()


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def pair_features(
    source1: dict[str, object],
    target: dict[str, object],
    passes: Iterable[str],
) -> list[float]:
    """Create fold-safe deterministic pair features for the M2_E4 model."""
    pass_set = set(passes)
    name1 = str(source1["name_norm"])
    name2 = str(target["name_norm"])
    address1 = str(source1["address_norm"])
    address2 = str(target["address_norm"])
    first1 = name1.split()[0] if name1.split() else ""
    first2 = name2.split()[0] if name2.split() else ""
    number1 = str(source1["address_number_signature"])
    number2 = str(target["address_number_signature"])
    values = {
        "country_equal": float(source1["country_norm"] == target["country_norm"]),
        "name_exact": float(name1 == name2 and bool(name1)),
        "name_sorted_equal": float(source1["name_sorted"] == target["name_sorted"]),
        "name_first_token_equal": float(bool(first1) and first1 == first2),
        "name_token_jaccard": _token_jaccard(name1, name2),
        "address_exact": float(address1 == address2 and bool(address1)),
        "address_token_jaccard": _token_jaccard(address1, address2),
        "address_number_equal": float(bool(number1) and number1 == number2),
        "name_length_difference": float(
            abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)
        ),
        "address_length_difference": float(
            abs(len(address1) - len(address2))
            / max(len(address1), len(address2), 1)
        ),
        "name_missing_either": float(
            bool(source1["name_missing"]) or bool(target["name_missing"])
        ),
        "address_missing_either": float(
            bool(source1["address_missing"]) or bool(target["address_missing"])
        ),
        "pass_exact_name": float("exact_name_country" in pass_set),
        "pass_sorted_name": float("sorted_name_country" in pass_set),
        "pass_name_token": float("name_token_country" in pass_set),
        "pass_name_prefix": float("name_prefix_country" in pass_set),
        "pass_address_number": float("address_number_country" in pass_set),
        "source_is_s2": float(target["source"] == "S2"),
        "source_is_s3": float(target["source"] == "S3"),
    }
    return [values[name] for name in FEATURE_NAMES]


def _matches_for(
    connection: sqlite3.Connection, source1_id: str
) -> set[str]:
    row = connection.execute(
        "SELECT matched_json FROM ground_truth WHERE source1_entity_id=?",
        (source1_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Missing ground truth for Source 1 entity {source1_id}.")
    return set(json.loads(row[0]))


def _train_lightgbm(
    features: list[list[float]],
    labels: list[int],
    weights: list[float],
    config: ExperimentConfig,
):
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is required for M2_E4. Install requirements.txt first."
        ) from error
    training_data = lgb.Dataset(
        np.asarray(features, dtype=np.float32),
        label=np.asarray(labels, dtype=np.int8),
        weight=np.asarray(weights, dtype=np.float32),
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
        training_data,
        num_boost_round=300,
    )


def _train_model(
    connection: sqlite3.Connection,
    index: SQLiteBlockingIndex,
    source1_path: Path,
    config: ExperimentConfig,
):
    train_ids = _selected_ids(connection, "train")
    features: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    for source1 in _selected_source1_rows(
        source1_path, train_ids, config.chunk_size
    ):
        source1_id = str(source1["entity_id"])
        truth = _matches_for(connection, source1_id)
        candidate_passes = index.candidates(source1)
        if not candidate_passes:
            continue
        all_candidates = sorted(candidate_passes)
        positive_ids = [entity_id for entity_id in all_candidates if entity_id in truth]
        negative_ids = [entity_id for entity_id in all_candidates if entity_id not in truth]
        if len(negative_ids) > config.negative_samples_per_entity:
            negative_ids = sorted(
                negative_ids,
                key=lambda entity_id: _hash_key(
                    config.random_seed, f"{source1_id}:{entity_id}"
                ),
            )[: config.negative_samples_per_entity]
        selected = positive_ids + negative_ids
        target_records = index.get_targets(selected)
        negative_weight = (
            max(1.0, (len(all_candidates) - len(positive_ids)) / len(negative_ids))
            if negative_ids
            else 1.0
        )
        for entity_id in positive_ids:
            features.append(
                pair_features(source1, target_records[entity_id], candidate_passes[entity_id])
            )
            labels.append(1)
            weights.append(1.0)
        for entity_id in negative_ids:
            features.append(
                pair_features(source1, target_records[entity_id], candidate_passes[entity_id])
            )
            labels.append(0)
            weights.append(negative_weight)

    if len(set(labels)) < 2:
        raise ValueError(
            "The selected training candidates do not contain both positive and "
            "negative labels; increase max_train_entities or review the blocking keys."
        )
    model = _train_lightgbm(features, labels, weights, config)
    return model, len(labels), int(sum(labels))


def _f05(true_ids: set[str], predicted_ids: set[str]) -> float:
    if not true_ids and not predicted_ids:
        return 1.0
    if not true_ids or not predicted_ids:
        return 0.0
    precision = len(true_ids & predicted_ids) / len(predicted_ids)
    recall = len(true_ids & predicted_ids) / len(true_ids)
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


def _nearest_rank_p95(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _evaluate_validation(
    connection: sqlite3.Connection,
    index: SQLiteBlockingIndex,
    source1_path: Path,
    validation_ids: set[str],
    model,
    config: ExperimentConfig,
    candidate_path: Path,
    results_path: Path,
) -> dict[str, object]:
    candidate_temp = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    results_temp = results_path.with_suffix(results_path.suffix + ".tmp")
    counts: list[int] = []
    pass_pair_counts: Counter[str] = Counter()
    total_true = total_candidate_hits = 0
    total_tp = total_fp = total_fn = 0
    macro_f05_values: list[float] = []
    macro_precision_values: list[float] = []
    macro_recall_values: list[float] = []
    singleton_total = singleton_correct = 0
    singleton_f05: list[float] = []
    row_count = 0
    pending_groups: list[
        tuple[str, list[str], set[str], int, int]
    ] = []
    pending_features: list[list[float]] = []

    def flush(candidate_writer, results_writer) -> None:
        nonlocal total_true, total_candidate_hits
        nonlocal total_tp, total_fp, total_fn, singleton_total
        nonlocal singleton_correct, row_count
        if not pending_groups:
            return
        if pending_features:
            probabilities = model.predict(
                np.asarray(pending_features, dtype=np.float32)
            )
        else:
            probabilities = np.empty(0, dtype=np.float32)
        for source1_id, candidate_ids, truth, start, end in pending_groups:
            scores = probabilities[start:end]
            predicted = {
                entity_id
                for entity_id, score in zip(candidate_ids, scores)
                if float(score) >= config.prediction_threshold
            }
            candidate_writer.writerow([source1_id, ",".join(candidate_ids)])
            results_writer.writerow([source1_id, ",".join(sorted(predicted))])
            hits = truth & set(candidate_ids)
            total_true += len(truth)
            total_candidate_hits += len(hits)
            true_positive = len(truth & predicted)
            total_tp += true_positive
            total_fp += len(predicted - truth)
            total_fn += len(truth - predicted)
            macro_f05_values.append(_f05(truth, predicted))
            macro_precision_values.append(
                true_positive / len(predicted) if predicted else float(not truth)
            )
            macro_recall_values.append(
                true_positive / len(truth) if truth else float(not predicted)
            )
            if not truth:
                singleton_total += 1
                is_correct = not predicted
                singleton_correct += int(is_correct)
                singleton_f05.append(float(is_correct))
            row_count += 1
        pending_groups.clear()
        pending_features.clear()

    with (
        candidate_temp.open("w", encoding="utf-8", newline="") as candidate_file,
        results_temp.open("w", encoding="utf-8", newline="") as results_file,
    ):
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        results_writer = csv.writer(results_file, delimiter="\t", lineterminator="\n")
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        results_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for source1 in _selected_source1_rows(
            source1_path, validation_ids, config.chunk_size
        ):
            source1_id = str(source1["entity_id"])
            passes_by_candidate = index.candidates(source1)
            candidate_ids = sorted(passes_by_candidate)
            target_records = index.get_targets(candidate_ids)
            start = len(pending_features)
            for entity_id in candidate_ids:
                pending_features.append(
                    pair_features(
                        source1,
                        target_records[entity_id],
                        passes_by_candidate[entity_id],
                    )
                )
                pass_pair_counts.update(passes_by_candidate[entity_id])
            end = len(pending_features)
            truth = _matches_for(connection, source1_id)
            counts.append(len(candidate_ids))
            pending_groups.append(
                (source1_id, candidate_ids, truth, start, end)
            )
            if len(pending_features) >= 50_000 or len(pending_groups) >= 2_000:
                flush(candidate_writer, results_writer)
        flush(candidate_writer, results_writer)
        candidate_file.flush()
        results_file.flush()
        os.fsync(candidate_file.fileno())
        os.fsync(results_file.fileno())
    os.replace(candidate_temp, candidate_path)
    os.replace(results_temp, results_path)

    validation_count = len(validation_ids)
    target_count = sum(index.target_counts().values())
    possible_pairs = validation_count * target_count
    non_singleton_rows = validation_count - singleton_total
    return {
        "validation_source1_rows": validation_count,
        "validation_rows_written": row_count,
        "validation_singletons": singleton_total,
        "singleton_accuracy": (
            singleton_correct / singleton_total if singleton_total else 0.0
        ),
        "singleton_macro_f0_5": (
            statistics.fmean(singleton_f05) if singleton_f05 else 0.0
        ),
        "candidate_pair_count": sum(counts),
        "candidate_count_average": (
            statistics.fmean(counts) if counts else 0.0
        ),
        "candidate_count_median": statistics.median(counts) if counts else 0.0,
        "candidate_count_p95": _nearest_rank_p95(counts),
        "candidate_count_max": max(counts, default=0),
        "possible_pair_count": possible_pairs,
        "candidate_reduction_ratio": (
            1.0 - sum(counts) / possible_pairs if possible_pairs else 0.0
        ),
        "blocking_recall": (
            total_candidate_hits / total_true if total_true else 0.0
        ),
        "blocking_recalled_match_ids": total_candidate_hits,
        "blocking_total_match_ids": total_true,
        "matching_precision_micro": (
            total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        ),
        "matching_recall_micro": (
            total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        ),
        "matching_precision_macro": (
            statistics.fmean(macro_precision_values)
            if macro_precision_values
            else 0.0
        ),
        "matching_recall_macro": (
            statistics.fmean(macro_recall_values)
            if macro_recall_values
            else 0.0
        ),
        "macro_f0_5": (
            statistics.fmean(macro_f05_values) if macro_f05_values else 0.0
        ),
        "true_positive_pairs": total_tp,
        "false_positive_pairs": total_fp,
        "false_negative_pairs": total_fn,
        "blocking_pass_candidate_pair_counts": dict(
            sorted(pass_pair_counts.items())
        ),
        "non_singleton_validation_rows": non_singleton_rows,
    }


def _append_results_csv(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "experiment_id",
        "validation_source1_rows",
        "candidate_pair_count",
        "candidate_count_average",
        "candidate_count_median",
        "candidate_count_p95",
        "candidate_count_max",
        "candidate_reduction_ratio",
        "blocking_recall",
        "matching_precision_micro",
        "matching_recall_micro",
        "matching_precision_macro",
        "matching_recall_macro",
        "macro_f0_5",
        "singleton_accuracy",
        "runtime_seconds",
        "metrics_path",
    )
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "M2_E4",
                **{field: metrics.get(field, "") for field in fields if field != "experiment_id"},
            }
        )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Build the full-target index, fit LightGBM, and evaluate the fixed holdout."""
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source1_path = config.train_dir / "train_source1.tsv"
    source2_path = config.train_dir / "train_source2.tsv"
    source3_path = config.train_dir / "train_source3.tsv"
    ground_truth_path = config.train_dir / "train_ground_truth.tsv"
    missing = [
        str(path)
        for path in (source1_path, source2_path, source3_path, ground_truth_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required training file(s): " + ", ".join(missing))

    config_path = config.output_dir / "config.json"
    config_payload = asdict(config)
    config_payload["train_dir"] = str(config.train_dir)
    config_payload["output_dir"] = str(config.output_dir)
    config_path.write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    blocking_config = BlockingConfig(max_bucket_size=config.max_bucket_size)
    phase_started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="m2e4-index-", dir=config.output_dir.parent
    ) as temporary_dir:
        database_path = Path(temporary_dir) / "m2e4.sqlite"
        with SQLiteBlockingIndex(database_path, blocking_config) as index:
            connection = index.connection
            stratum_counts = _open_ground_truth_store(
                connection, ground_truth_path, config
            )
            split_seconds = time.perf_counter() - phase_started
            target_counts = _build_target_index(
                index, config.train_dir, config.chunk_size
            )
            index_seconds = time.perf_counter() - phase_started - split_seconds
            validation_ids = _selected_ids(connection, "validation")
            train_sample_count = int(
                connection.execute("SELECT COUNT(*) FROM selected_train").fetchone()[0]
            )

            train_started = time.perf_counter()
            model, train_pair_count, train_positive_count = _train_model(
                connection,
                index,
                source1_path,
                config,
            )
            train_seconds = time.perf_counter() - train_started
            validation_started = time.perf_counter()
            metrics = _evaluate_validation(
                connection,
                index,
                source1_path,
                validation_ids,
                model,
                config,
                config.output_dir / "candidate_pairs.tsv",
                config.output_dir / "matching_results.tsv",
            )
            validation_seconds = time.perf_counter() - validation_started
            target_key_count = int(
                connection.execute("SELECT COUNT(*) FROM blocking_keys").fetchone()[0]
            )
            key_size_rows = int(
                connection.execute("SELECT COUNT(*) FROM key_sizes").fetchone()[0]
            )

    metrics.update(
        {
            "experiment_id": "M2_E4",
            "validation_fraction": config.validation_fraction,
            "validation_seed": config.random_seed,
            "validation_stratum_counts": {
                "singletons": stratum_counts.get(0, 0),
                "one_match": stratum_counts.get(1, 0),
                "multi_match": stratum_counts.get(2, 0),
            },
            "training_source1_sample_rows": train_sample_count,
            "training_candidate_pairs_sampled": train_pair_count,
            "training_positive_pairs": train_positive_count,
            "negative_samples_per_entity": config.negative_samples_per_entity,
            "target_record_counts": target_counts,
            "target_record_count": sum(target_counts.values()),
            "blocking_key_rows": target_key_count,
            "unique_blocking_keys": key_size_rows,
            "blocking_passes": list(BLOCKING_PASSES),
            "max_bucket_size": config.max_bucket_size,
            "prediction_threshold": config.prediction_threshold,
            "model": "LightGBM native binary booster; 300 trees; fixed threshold",
            "features": list(FEATURE_NAMES),
            "split_note": (
                "Source 1 entities are assigned by match-count stratum using "
                "SHA-256(seed, entity_id); validation and fit S1 IDs are disjoint. "
                "All S2/S3 training records form the candidate universe."
            ),
            "negative_sampling_note": (
                "Training negatives are deterministically capped per sampled "
                "training entity and weighted by the inverse sampling fraction."
            ),
            "split_runtime_seconds": split_seconds,
            "target_index_runtime_seconds": index_seconds,
            "model_fit_runtime_seconds": train_seconds,
            "validation_runtime_seconds": validation_seconds,
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _append_results_csv(config.output_dir.parent / "experiment_results.csv", metrics)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("student_resource/dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/M2_E4"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-entities", type=int, default=25_000)
    parser.add_argument("--negative-samples-per-entity", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--max-bucket-size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
        max_train_entities=args.max_train_entities,
        negative_samples_per_entity=args.negative_samples_per_entity,
        chunk_size=args.chunk_size,
        max_bucket_size=args.max_bucket_size,
        prediction_threshold=args.threshold,
    )
    metrics = run_experiment(config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
