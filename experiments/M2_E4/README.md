# M2_E4: multi-pass blocking + LightGBM

Run from the `ML-Challenge/` project root:

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
`notebooks/member2/M2_E4_multipass_lgbm.ipynb` is an orchestration wrapper for
the same runner, not a separate implementation.
