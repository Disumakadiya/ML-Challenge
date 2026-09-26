import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.M2_E4.run_experiment import ExperimentConfig, run_experiment


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


if __name__ == "__main__":
    unittest.main()
