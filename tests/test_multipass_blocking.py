import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.blocking.multipass_blocking import (
    BlockingConfig,
    InMemoryBlockingIndex,
    SQLiteBlockingIndex,
    generate_candidate_pairs,
)
from src.preprocessing import prepare_source


def _source_rows(values: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        columns=("entity_id", "business_name", "business_address", "country"),
        dtype="string",
    )


class MultiPassBlockingTests(unittest.TestCase):
    def test_union_deduplicates_candidates_and_keeps_zero_candidate_rows(self) -> None:
        source1 = _source_rows(
            [
                ("S1-1", "North Star Cafe", "12 Baker Street", "US"),
                ("S1-2", "Blue River Shop", "44 Oak Road", "France"),
                ("S1-3", "No Match Business", "", "India"),
            ],
        )
        source2 = _source_rows(
            [
                ("S2-1", "North Star Cafe", "12 Baker Street", "US"),
                ("S2-2", "River Blue Shop", "44 Oak Road", "France"),
            ],
        )
        source3 = _source_rows(
            [("S3-1", "Unrelated Company", "99 Other Lane", "US")],
        )

        result = generate_candidate_pairs(source1, source2, source3)
        by_s1 = result.candidate_pairs.set_index("source1_entity_id")[
            "candidate_entity_ids"
        ]
        self.assertEqual(by_s1["S1-1"], "S2-1")
        self.assertEqual(by_s1["S1-2"], "S2-2")
        self.assertEqual(by_s1["S1-3"], "")
        self.assertEqual(len(result.pairs), 2)
        self.assertTrue(
            set(result.pairs["candidate_entity_id"]).issubset({"S2-1", "S2-2"})
        )
        self.assertLess(
            result.diagnostics["candidate_pair_count"],
            result.diagnostics["possible_pair_count"],
        )

    def test_frequent_keys_are_skipped_deterministically(self) -> None:
        targets = _source_rows(
            [
                ("S2-1", "Common Store Alpha", "1 Main Road", "US"),
                ("S2-2", "Common Store Beta", "2 Main Road", "US"),
            ],
        )
        prepared = prepare_source(targets)
        index = InMemoryBlockingIndex(
            prepared,
            BlockingConfig(max_bucket_size=1),
        )
        source1 = prepare_source(
            _source_rows(
                [("S1-1", "Common Store Gamma", "3 Main Road", "US")],
            )
        )
        self.assertEqual(index.candidates(source1.iloc[0].to_dict()), {})
        self.assertGreater(len(index.skipped_keys), 0)

    def test_sqlite_index_matches_in_memory_candidates(self) -> None:
        targets = prepare_source(
            _source_rows(
                [
                    ("S2-1", "North Star Cafe", "12 Baker Street", "US"),
                    ("S2-2", "River Blue Shop", "44 Oak Road", "France"),
                ],
            )
        )
        source1 = prepare_source(
            _source_rows(
                [
                    ("S1-1", "Star North Cafe", "12 Baker St", "US"),
                    ("S1-2", "Unseen Business", "", "India"),
                ],
            )
        )
        config = BlockingConfig()
        memory_index = InMemoryBlockingIndex(targets, config)

        with tempfile.TemporaryDirectory() as directory:
            with SQLiteBlockingIndex(Path(directory) / "blocking.sqlite", config) as index:
                index.add_targets(targets)
                index.finish()
                for row in source1.to_dict(orient="records"):
                    expected = memory_index.candidates(row)
                    actual = index.candidates(row)
                    self.assertEqual(actual, expected)
                self.assertEqual(index.target_counts(), {"S2": 2})


if __name__ == "__main__":
    unittest.main()
