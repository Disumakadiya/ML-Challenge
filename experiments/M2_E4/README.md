# M2_E4: multi-pass blocking + LightGBM

Run the train-only validation experiment from the `ML-Challenge/` project root:

```powershell
python -m pip install -r requirements.txt
python -m experiments.M2_E4.run_experiment
```

The run uses only `student_resource/dataset/train/`. It assigns Source 1 rows
to a deterministic, match-count-stratified 20% validation fold (`seed=42`),
indexes every S2/S3 training record in a temporary SQLite database, and fits
LightGBM on a deterministic 25,000-row cap from the remaining Source 1 rows.
Training negatives are deterministically capped at 30 candidates per sampled
entity and weighted for the sampling rate. Validation scores and candidates
cover the full training target universe. No test data is used.

The run writes validation-fold `candidate_pairs.tsv` and
`matching_results.tsv`, full metrics/configuration in `metrics.json` and
`config.json`, and appends one row to `experiments/experiment_results.csv`.
These are validation artifacts, not test submissions. Candidate pruning skips
any individual blocking key whose target bucket exceeds 100 records.

Precision, recall and F0.5 are reported both per-entity macro (the competition
scoring basis) and as micro pair aggregates where applicable. The prediction
threshold is fixed at 0.5 by default and is not selected using test data.

Defaults are configurable through CLI options; use `python -m
experiments.M2_E4.run_experiment --help` for the option list. The notebook
`notebooks/member2/M2_E4_multipass_lgbm.ipynb` runs final inference and creates
the test-set files required for submission. From the project root, the same
operation can be run without Jupyter:

```powershell
python -m experiments.M2_E4.run_experiment --final-inference
```

It trains using the training TSVs and ground truth only, then builds a
disk-backed candidate index from all test Source 2 and Source 3 records and
writes one output row for every test Source 1 ID. The official submission files
are `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
`output/metrics.json` records candidate counts and runtime; it is not one of
the two challenge TSV submissions. Every predicted ID is selected from the
candidate set by construction, and candidates come from the test target index.
The threshold is fixed at 0.5; the test labels are never read or used for model
selection.

After completion, validate the files from `student_resource/`:

```powershell
python utils/validate_submission.py `
  --matching ../output/matching_results.tsv `
  --candidate ../output/candidate_pairs.tsv `
  --test-dir dataset/test
```

The inference process uses temporary SQLite indexes under the output
directory's parent and removes them on successful completion or failure. It
does not write to `student_resource/dataset/`. Full train/test indexing may take
substantial time and temporary disk space; do not stop the process before it
finishes. An interrupted/failed run can leave incomplete temporary TSV files;
rerun to replace them.
