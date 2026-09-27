import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.M2_E4.run_experiment import (
    ExperimentConfig,
    run_experiment,
    run_final_inference,
)


class M2E4ExperimentTests(unittest.TestCase):
    def test_train_only_end_to_end_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / "train"
            output_dir = root / "experiments" / "M2_E4"
            train_dir.mkdir()

            source1_rows = []
            source2_rows = []
            source3_rows = []
            truth_rows = []
            for index in range(90):
                source1_id = f"S1-{index:05d}"
                name = f"Company {index} Cafe"
                address = f"{index + 100} Main Road"
                country = ("US", "India", "France")[index % 3]
                source1_rows.append((source1_id, name, address, country))

                matched_ids = []
                source2_id = f"S2-{index:05d}"
                if index % 9:
                    source2_rows.append((source2_id, name, address, country))
                    matched_ids.append(source2_id)
                else:
                    source2_rows.append(
                        (source2_id, f"Different Business {index}", address, country)
                    )

                if index % 4 == 0 and index % 9:
                    source3_id = f"S3-{index:05d}"
                    source3_rows.append(
                        (
                            source3_id,
                            f"Cafe {index} Company",
                            address,
                            country,
                        )
                    )
                    matched_ids.append(source3_id)
                truth_rows.append((source1_id, ",".join(matched_ids)))

            columns = ("entity_id", "business_name", "business_address", "country")
            for filename, rows in (
                ("train_source1.tsv", source1_rows),
                ("train_source2.tsv", source2_rows),
                ("train_source3.tsv", source3_rows),
            ):
                pd.DataFrame(rows, columns=columns).to_csv(
                    train_dir / filename,
                    sep="\t",
                    index=False,
                    encoding="utf-8",
                )
            pd.DataFrame(
                truth_rows,
                columns=("source1_entity_id", "matched_entity_ids"),
            ).to_csv(
                train_dir / "train_ground_truth.tsv",
                sep="\t",
                index=False,
                encoding="utf-8",
            )

            metrics = run_experiment(
                ExperimentConfig(
                    train_dir=train_dir,
                    output_dir=output_dir,
                    max_train_entities=60,
                    negative_samples_per_entity=10,
                    chunk_size=25,
                    max_bucket_size=200,
                )
            )
            self.assertEqual(metrics["validation_source1_rows"], metrics["validation_rows_written"])
            self.assertGreater(metrics["candidate_pair_count"], 0)
            self.assertGreaterEqual(metrics["blocking_recall"], 0.0)
            self.assertLessEqual(metrics["blocking_recall"], 1.0)
            self.assertEqual(
                sum(metrics["validation_stratum_counts"].values()), 90
            )
            self.assertIn("matching_precision_macro", metrics)
            self.assertIn("matching_recall_macro", metrics)
            self.assertTrue((output_dir / "metrics.json").is_file())

            with (output_dir / "candidate_pairs.tsv").open(
                encoding="utf-8", newline=""
            ) as candidate_file, (output_dir / "matching_results.tsv").open(
                encoding="utf-8", newline=""
            ) as results_file:
                candidates = list(csv.DictReader(candidate_file, delimiter="\t"))
                results = list(csv.DictReader(results_file, delimiter="\t"))
            self.assertEqual(len(candidates), metrics["validation_source1_rows"])
            self.assertEqual(len(results), metrics["validation_source1_rows"])
            candidate_ids = {
                row["source1_entity_id"]: set(
                    filter(None, row["candidate_entity_ids"].split(","))
                )
                for row in candidates
            }
            for row in results:
                predicted_ids = set(filter(None, row["matched_entity_ids"].split(",")))
                self.assertTrue(
                    predicted_ids.issubset(candidate_ids[row["source1_entity_id"]])
                )
            self.assertTrue((output_dir.parent / "experiment_results.csv").is_file())

    def test_final_inference_writes_test_submission_files_for_every_source1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_dir = root / "train"
            test_dir = root / "test"
            experiment_dir = root / "experiments" / "M2_E4"
            submission_dir = root / "output"
            train_dir.mkdir()
            test_dir.mkdir()

            source_columns = (
                "entity_id",
                "business_name",
                "business_address",
                "country",
            )
            train_source1 = []
            train_source2 = []
            train_source3 = []
            truth_rows = []
            for index in range(60):
                source1_id = f"S1-{index:05d}"
                name = f"Training Company {index} Cafe"
                address = f"{index + 100} Main Road"
                country = ("US", "India", "France")[index % 3]
                train_source1.append((source1_id, name, address, country))
                source2_id = f"S2-{index:05d}"
                train_source2.append((source2_id, name, address, country))
                train_source3.append(
                    (
                        f"S3-{index:05d}",
                        f"Cafe {index} Company Training",
                        address,
                        country,
                    )
                )
                truth_rows.append((source1_id, source2_id))

            train_tables = (
                ("train_source1.tsv", train_source1),
                ("train_source2.tsv", train_source2),
                ("train_source3.tsv", train_source3),
            )
            for filename, rows in train_tables:
                pd.DataFrame(rows, columns=source_columns).to_csv(
                    train_dir / filename, sep="\t", index=False
                )
            pd.DataFrame(
                truth_rows,
                columns=("source1_entity_id", "matched_entity_ids"),
            ).to_csv(
                train_dir / "train_ground_truth.tsv", sep="\t", index=False
            )

            test_source1 = [
                ("S1-T0001", "Test Cedar Market", "21 Cedar Street", "France"),
                ("S1-T0002", "Test Lonely Business", "", "A new country"),
            ]
            test_source2 = [
                ("S2-T0001", "Cedar Test Market", "21 Cedar St", "France"),
                ("S2-T0002", "Unrelated Target", "99 Other Road", "US"),
            ]
            test_source3 = [
                ("S3-T0001", "Completely Separate", "80 Birch Lane", "India")
            ]
            for filename, rows in (
                ("test_source1.tsv", test_source1),
                ("test_source2.tsv", test_source2),
                ("test_source3.tsv", test_source3),
            ):
                pd.DataFrame(rows, columns=source_columns).to_csv(
                    test_dir / filename, sep="\t", index=False
                )

            metrics = run_final_inference(
                ExperimentConfig(
                    train_dir=train_dir,
                    output_dir=experiment_dir,
                    max_train_entities=50,
                    negative_samples_per_entity=10,
                    chunk_size=20,
                    max_bucket_size=100,
                ),
                test_dir=test_dir,
                submission_dir=submission_dir,
            )

            candidate_path = submission_dir / "candidate_pairs.tsv"
            matching_path = submission_dir / "matching_results.tsv"
            self.assertTrue(candidate_path.is_file())
            self.assertTrue(matching_path.is_file())
            self.assertTrue((submission_dir / "metrics.json").is_file())
            with candidate_path.open(encoding="utf-8", newline="") as handle:
                candidates = list(csv.DictReader(handle, delimiter="\t"))
            with matching_path.open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle, delimiter="\t"))
            expected_source1_ids = {row[0] for row in test_source1}
            self.assertEqual(
                {row["source1_entity_id"] for row in candidates},
                expected_source1_ids,
            )
            self.assertEqual(
                {row["source1_entity_id"] for row in predictions},
                expected_source1_ids,
            )
            candidates_by_source1 = {
                row["source1_entity_id"]: set(
                    filter(None, row["candidate_entity_ids"].split(","))
                )
                for row in candidates
            }
            test_target_ids = {
                row[0] for row in test_source2 + test_source3
            }
            for row in candidates:
                candidate_ids = candidates_by_source1[row["source1_entity_id"]]
                self.assertTrue(candidate_ids.issubset(test_target_ids))
            for row in predictions:
                predicted_ids = set(
                    filter(None, row["matched_entity_ids"].split(","))
                )
                self.assertTrue(
                    predicted_ids.issubset(
                        candidates_by_source1[row["source1_entity_id"]]
                    )
                )
            self.assertEqual(metrics["source1_count"], len(test_source1))
            self.assertEqual(metrics["target_count"], len(test_target_ids))
            self.assertEqual(
                metrics["prediction_count"],
                sum(
                    len(list(filter(None, row["matched_entity_ids"].split(","))))
                    for row in predictions
                ),
            )
            validator = (
                Path(__file__).resolve().parents[1]
                / "student_resource"
                / "utils"
                / "validate_submission.py"
            )
            validation = subprocess.run(
                [
                    sys.executable,
                    str(validator),
                    "--matching",
                    str(matching_path),
                    "--candidate",
                    str(candidate_path),
                    "--test-dir",
                    str(test_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                validation.returncode,
                0,
                msg=validation.stdout + validation.stderr,
            )


if __name__ == "__main__":
    unittest.main()
