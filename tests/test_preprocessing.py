import unittest

import pandas as pd

from src.preprocessing import (
    _prepare_ground_truth,
    normalize_text,
    prepare_source,
    validate_source,
)


class PreprocessingTests(unittest.TestCase):
    def test_normalization_preserves_unicode_letters_and_cleans_spacing(self) -> None:
        self.assertEqual(
            normalize_text("  ＡＣＭＥ—東京, Café!\t"),
            "acme 東京 café",
        )
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(pd.NA), "")
        self.assertEqual(normalize_text("NULL"), "")

    def test_prepares_all_sources_without_changing_raw_values_or_ids(self) -> None:
        for source, country in (("S1", "France"), ("S2", "US"), ("S3", "India")):
            frame = pd.DataFrame(
                {
                    "entity_id": [f"{source}-00001"],
                    "business_name": ["The Café, 東京"],
                    "business_address": ["12-B Baker St., Floor 2"],
                    "country": [country],
                },
                dtype="string",
            )
            prepared = prepare_source(frame, expected_source=source)

            self.assertEqual(prepared.loc[0, "entity_id"], frame.loc[0, "entity_id"])
            self.assertEqual(prepared.loc[0, "source"], source)
            for column in (
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ):
                self.assertTrue(prepared[column].equals(frame[column]))
            self.assertEqual(prepared.loc[0, "country_norm"], country.casefold())
            self.assertEqual(prepared.loc[0, "name_norm"], "the café 東京")
            self.assertEqual(prepared.loc[0, "name_compact"], "thecafé東京")
            self.assertEqual(prepared.loc[0, "name_sorted"], "café the 東京")
            self.assertEqual(prepared.loc[0, "address_number_signature"], "12-2")

    def test_country_normalization_accepts_unseen_country(self) -> None:
        frame = pd.DataFrame(
            {
                "entity_id": ["S1-1"],
                "business_name": ["Example"],
                "business_address": [""],
                "country": ["A country not in the training set"],
            },
            dtype="string",
        )
        prepared = prepare_source(frame)
        self.assertEqual(
            prepared.loc[0, "country_norm"], "a country not in the training set"
        )

    def test_duplicate_and_wrong_prefix_are_rejected(self) -> None:
        base = pd.DataFrame(
            {
                "entity_id": ["S1-1"],
                "business_name": ["Name"],
                "business_address": ["Address"],
                "country": ["Country"],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_source(pd.concat([base, base], ignore_index=True))

        with self.assertRaisesRegex(ValueError, "expected 'S2-'"):
            validate_source(base, expected_source="S2")

        invalid = base.copy()
        invalid["entity_id"] = "X1-1"
        with self.assertRaisesRegex(ValueError, "S1-, S2-, or S3-"):
            validate_source(invalid)

    def test_missing_features_and_ground_truth_lists(self) -> None:
        frame = pd.DataFrame(
            {
                "entity_id": ["S2-1"],
                "business_name": [pd.NA],
                "business_address": [None],
                "country": ["France"],
            }
        )
        prepared = prepare_source(frame)
        self.assertTrue(prepared.loc[0, "name_missing"])
        self.assertTrue(prepared.loc[0, "address_missing"])
        self.assertFalse(prepared.loc[0, "country_missing"])
        self.assertEqual(prepared.loc[0, "name_token_count"], 0)

        truth = _prepare_ground_truth(
            pd.DataFrame(
                {
                    "source1_entity_id": ["S1-1", "S1-2"],
                    "matched_entity_ids": ["", "S2-3,S3-4"],
                }
            )
        )
        self.assertEqual(
            truth["matched_entity_ids"].tolist(), [[], ["S2-3", "S3-4"]]
        )


if __name__ == "__main__":
    unittest.main()
