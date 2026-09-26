"""Shared, deterministic preprocessing utilities for business entity records."""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Iterator
from os import PathLike
from typing import Sequence, Union

import pandas as pd

FilePath = Union[str, PathLike[str]]

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
_SOURCES = ("S1", "S2", "S3")
_NULL_TEXT = {"", "nan", "none", "null", "<na>", "n/a"}
_DERIVED_COLUMNS = {
    "norm_name",
    "norm_address",
    "norm_country",
    "name_norm",
    "name_compact",
    "name_alnum",
    "name_sorted",
    "address_norm",
    "address_compact",
    "address_alnum",
    "address_sorted",
    "address_number_signature",
    "country_norm",
    "source",
    "name_char_length",
    "address_char_length",
    "name_length",
    "address_length",
    "name_token_count",
    "address_token_count",
    "name_missing",
    "address_missing",
    "country_missing",
}


def _read_options() -> dict[str, object]:
    return {
        "sep": "\t",
        "dtype": "string",
        "keep_default_na": False,
        "encoding": "utf-8",
    }


def _validate_expected_source(expected_source: str | None) -> None:
    if expected_source is not None and expected_source not in _SOURCES:
        raise ValueError("expected_source must be one of 'S1', 'S2', or 'S3'.")


def _required_columns(columns: Sequence[str], required: Sequence[str]) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _text_value(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if not isinstance(value, str):
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        value = str(value)
    text = value.strip()
    null_marker = unicodedata.normalize("NFKC", text).casefold()
    return "" if null_marker in _NULL_TEXT else value


def normalize_text(value: object) -> str:
    """Normalize a scalar text value without altering its meaning or token order."""
    text = unicodedata.normalize("NFKC", _text_value(value)).casefold()
    characters: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character.isspace():
            characters.append(" ")
        elif category[0] in {"L", "M", "N"}:
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def _is_missing(value: object) -> bool:
    return not _text_value(value).strip()


def _id_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string")


def validate_source(
    frame: pd.DataFrame, expected_source: str | None = None
) -> None:
    """Validate the required source columns, IDs, uniqueness, and optional prefix."""
    _validate_expected_source(expected_source)
    _required_columns(list(frame.columns), SOURCE_COLUMNS)

    entity_ids = _id_series(frame, "entity_id")
    empty_ids = entity_ids.isna() | entity_ids.str.strip().eq("")
    if empty_ids.any():
        raise ValueError(f"Found {int(empty_ids.sum())} empty entity_id value(s).")

    duplicate_ids = entity_ids.duplicated(keep=False)
    if duplicate_ids.any():
        raise ValueError(
            f"Found {int(duplicate_ids.sum())} row(s) with duplicate entity_id values."
        )

    valid_prefix = entity_ids.str.startswith(
        tuple(f"{source}-" for source in _SOURCES), na=False
    )
    if (~valid_prefix).any():
        raise ValueError(
            f"Found {int((~valid_prefix).sum())} entity_id value(s) without "
            "an S1-, S2-, or S3- prefix."
        )

    if expected_source is not None:
        expected_prefix = f"{expected_source}-"
        wrong_prefix = ~entity_ids.str.startswith(expected_prefix, na=False)
        if wrong_prefix.any():
            raise ValueError(
                f"Found {int(wrong_prefix.sum())} entity_id value(s) without "
                f"the expected {expected_prefix!r} prefix."
            )


def load_source(
    path: FilePath, expected_source: str | None = None
) -> pd.DataFrame:
    """Load and validate a source TSV, preserving every field as a string."""
    _validate_expected_source(expected_source)
    frame = pd.read_csv(path, **_read_options())
    validate_source(frame, expected_source=expected_source)
    return frame


def _infer_source(entity_id: object) -> str:
    text = _text_value(entity_id)
    for source in _SOURCES:
        if text.startswith(f"{source}-"):
            return source
    return ""


def _compact(text: str) -> str:
    return "".join(text.split())


def _sorted_tokens(text: str) -> str:
    return " ".join(sorted(text.split()))


def _number_signature(text: str) -> str:
    return "-".join(re.findall(r"\d+", text))


def prepare_source(
    frame: pd.DataFrame, expected_source: str | None = None
) -> pd.DataFrame:
    """Copy source records and add normalized text and deterministic row features."""
    validate_source(frame, expected_source=expected_source)
    collisions = sorted(_DERIVED_COLUMNS.intersection(frame.columns))
    if collisions:
        raise ValueError(
            "Source data already contains reserved preprocessing column(s): "
            + ", ".join(collisions)
        )
    prepared = frame.copy()
    prepared["entity_id"] = _id_series(prepared, "entity_id")

    text_columns = {
        "business_name": "norm_name",
        "business_address": "norm_address",
        "country": "norm_country",
    }
    for raw_column, normalized_column in text_columns.items():
        prepared[normalized_column] = prepared[raw_column].map(normalize_text)

    prepared["name_norm"] = prepared["norm_name"]
    prepared["name_compact"] = prepared["norm_name"].map(_compact)
    prepared["name_alnum"] = prepared["name_compact"]
    prepared["name_sorted"] = prepared["norm_name"].map(_sorted_tokens)
    prepared["address_norm"] = prepared["norm_address"]
    prepared["address_compact"] = prepared["norm_address"].map(_compact)
    prepared["address_alnum"] = prepared["address_compact"]
    prepared["address_sorted"] = prepared["norm_address"].map(_sorted_tokens)
    prepared["address_number_signature"] = prepared["norm_address"].map(
        _number_signature
    )
    prepared["country_norm"] = prepared["norm_country"]
    prepared["source"] = prepared["entity_id"].map(_infer_source)
    prepared["name_char_length"] = prepared["norm_name"].str.len()
    prepared["address_char_length"] = prepared["norm_address"].str.len()
    prepared["name_length"] = prepared["name_char_length"]
    prepared["address_length"] = prepared["address_char_length"]
    prepared["name_token_count"] = prepared["norm_name"].map(
        lambda text: len(text.split())
    )
    prepared["address_token_count"] = prepared["norm_address"].map(
        lambda text: len(text.split())
    )
    prepared["name_missing"] = prepared["business_name"].map(_is_missing)
    prepared["address_missing"] = prepared["business_address"].map(_is_missing)
    prepared["country_missing"] = prepared["country"].map(_is_missing)
    return prepared


def _parse_match_ids(value: object) -> list[str]:
    text = _text_value(value)
    if not text:
        return []
    match_ids = [match_id.strip() for match_id in text.split(",")]
    if any(not match_id for match_id in match_ids):
        raise ValueError("matched_entity_ids contains an empty list item.")
    if any(
        not match_id.startswith(("S2-", "S3-")) for match_id in match_ids
    ):
        raise ValueError("matched_entity_ids must contain only S2- or S3- IDs.")
    if len(match_ids) != len(set(match_ids)):
        raise ValueError("matched_entity_ids contains duplicate IDs.")
    return match_ids


def _prepare_ground_truth(frame: pd.DataFrame) -> pd.DataFrame:
    _required_columns(list(frame.columns), GROUND_TRUTH_COLUMNS)
    result = frame.copy()
    source1_ids = _id_series(result, "source1_entity_id")
    empty_ids = source1_ids.isna() | source1_ids.str.strip().eq("")
    if empty_ids.any():
        raise ValueError(
            f"Found {int(empty_ids.sum())} empty source1_entity_id value(s)."
        )
    if (~source1_ids.str.startswith("S1-", na=False)).any():
        raise ValueError("source1_entity_id values must use the 'S1-' prefix.")
    if source1_ids.duplicated(keep=False).any():
        raise ValueError("source1_entity_id values must be unique.")

    result["source1_entity_id"] = source1_ids
    result["matched_entity_ids"] = result["matched_entity_ids"].map(
        _parse_match_ids
    )
    return result


def load_ground_truth(path: FilePath) -> pd.DataFrame:
    """Load ground truth and parse each comma-separated match list into a list."""
    frame = pd.read_csv(path, **_read_options())
    return _prepare_ground_truth(frame)


def iter_tsv_chunks(
    path: FilePath, chunksize: int = 100_000
) -> Iterator[pd.DataFrame]:
    """Yield string-preserving TSV chunks using an explicit tab separator."""
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")
    reader = pd.read_csv(path, chunksize=chunksize, **_read_options())
    yield from reader


def iter_source_chunks(
    path: FilePath,
    chunksize: int = 100_000,
    expected_source: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield validated source chunks and detect duplicate IDs across chunks."""
    _validate_expected_source(expected_source)
    seen_ids: set[str] = set()
    for chunk in iter_tsv_chunks(path, chunksize=chunksize):
        validate_source(chunk, expected_source=expected_source)
        entity_ids = _id_series(chunk, "entity_id")
        repeated = entity_ids.isin(seen_ids)
        if repeated.any():
            raise ValueError(
                f"Found {int(repeated.sum())} entity_id value(s) duplicated "
                "across TSV chunks."
            )
        seen_ids.update(entity_ids.tolist())
        yield chunk


def iter_prepared_tsv(
    path: FilePath,
    chunksize: int = 100_000,
    expected_source: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield validated, prepared source chunks without loading the full TSV."""
    for chunk in iter_source_chunks(
        path, chunksize=chunksize, expected_source=expected_source
    ):
        yield prepare_source(chunk, expected_source=expected_source)


def write_tsv(frame: pd.DataFrame, path: FilePath) -> None:
    """Write a DataFrame as UTF-8 TSV without an index."""
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8")


def validate_source_file(
    path: FilePath,
    expected_source: str | None = None,
    chunksize: int = 100_000,
) -> tuple[int, list[str]]:
    """Validate a source TSV in chunks and return its row count and columns."""
    _validate_expected_source(expected_source)
    header = pd.read_csv(path, nrows=0, **_read_options())
    columns = list(header.columns)
    _required_columns(columns, SOURCE_COLUMNS)
    row_count = sum(
        len(chunk)
        for chunk in iter_source_chunks(
            path, chunksize=chunksize, expected_source=expected_source
        )
    )
    return row_count, columns


def _run_self_tests() -> None:
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text("ＮＵＬＬ") == ""
    assert normalize_text("  ＡＣＭＥ—東京, Café!  ") == "acme 東京 café"

    valid = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "business_name": ["  ＡＣＭＥ—東京, Café!  "],
            "business_address": [pd.NA],
            "country": ["France"],
        }
    )
    prepared = prepare_source(valid, expected_source="S1")
    assert prepared.loc[0, "entity_id"] == "S1-1"
    assert prepared.loc[0, "norm_name"] == "acme 東京 café"
    assert prepared.loc[0, "norm_country"] == "france"
    assert prepared.loc[0, "country_norm"] == "france"
    assert prepared.loc[0, "address_missing"]
    assert prepared.loc[0, "name_char_length"] == len("acme 東京 café")
    assert prepared.loc[0, "name_length"] == len("acme 東京 café")
    assert prepared.loc[0, "name_token_count"] == 3
    assert prepared.loc[0, "business_name"] == valid.loc[0, "business_name"]

    try:
        validate_source(pd.concat([valid, valid], ignore_index=True))
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Duplicate entity IDs were not rejected.")

    try:
        validate_source(valid, expected_source="S2")
    except ValueError as error:
        assert "expected 'S2-'" in str(error)
    else:
        raise AssertionError("An incorrect source prefix was not rejected.")

    invalid_prefix = valid.copy()
    invalid_prefix["entity_id"] = "OTHER-1"
    try:
        validate_source(invalid_prefix)
    except ValueError as error:
        assert "S1-, S2-, or S3-" in str(error)
    else:
        raise AssertionError("An unsupported source prefix was not rejected.")

    truth = _prepare_ground_truth(
        pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2"],
                "matched_entity_ids": ["", "S2-7, S3-9"],
            }
        )
    )
    assert truth["matched_entity_ids"].tolist() == [[], ["S2-7", "S3-9"]]


def main(argv: Sequence[str] | None = None) -> int:
    """Run a built-in self-test or validate a source TSV without loading it whole."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--validate", metavar="TSV_PATH")
    parser.add_argument("--expected-source", choices=_SOURCES)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args(argv)

    if args.self_test:
        _run_self_tests()
        print("PASS: preprocessing self-tests")
        return 0

    rows, columns = validate_source_file(
        args.validate,
        expected_source=args.expected_source,
        chunksize=args.chunksize,
    )
    print(f"PASS: {rows} rows, {len(columns)} columns")
    print(f"Columns: {', '.join(columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())