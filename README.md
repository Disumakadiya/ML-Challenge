# Amazon ML Challenge 2026 — Business Entity Resolution

Shared team repository for the Amazon ML Challenge 2026 Business Entity
Resolution task.

## 1. Problem overview

Given business records from three independent sources, identify which Source 2
and/or Source 3 records refer to each Source 1 business. Source 1 is the
deduplicated reference source. A Source 1 record may match zero, one, or many
records across Sources 2 and 3. This is a record-linkage task, not a one-to-one
assignment problem.

Country is an open-set string field. Training includes the US and India; test
also includes France. Do not filter or hard-code country values.

## 2. Challenge data

All challenge data is tab-separated TSV. Read using an explicit tab separator,
and preserve IDs as strings.

```text
student_resource/dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

The three source files contain `entity_id`, `business_name`,
`business_address`, and `country`. IDs use the source prefixes `S1-`, `S2-`,
and `S3-`. Ground truth contains `source1_entity_id` and
`matched_entity_ids`, a comma-separated list of matching S2/S3 IDs that may
be empty or contain multiple matches.

The original Amazon-provided files are under `student_resource/`; do not modify
or overwrite them. The dataset directory is ignored by Git because the files
are large and must remain local.

## 3. Evaluation and submission

The challenge metric is macro F0.5 per Source 1 entity. Singletons are
included: an empty prediction for a true singleton scores 1.0, and predicting
any match for it scores 0.0.

The final output directory must contain:

- `matching_results.tsv`: exactly `source1_entity_id` and
  `matched_entity_ids`, with exactly one row for every test Source 1 record.
- `candidate_pairs.tsv`: exactly `source1_entity_id` and
  `candidate_entity_ids`, containing the final candidate set passed to the
  matching model. Every predicted match must appear among its candidates.

Both files are TSV. Match/candidate ID lists are comma-separated, contain only
existing S2/S3 IDs, and contain no duplicates. Empty lists are represented by
empty cells. Candidate generation must scale; do not compare all possible
source pairs.

Validate final submissions from `student_resource/` using the supplied
validator:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## 4. Repository layout

```text
ML-Challenge/
├── src/
│   ├── preprocessing.py
│   ├── blocking/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── submission/
├── experiments/
├── notebooks/
├── tests/
├── student_resource/
├── requirements.txt
└── run_pipeline.py
```

Keep each member's experiments separate and integrate through shared function
interfaces.

## 5. Shared preprocessing

All members should import the common utilities from `src.preprocessing` while
working from the `ML-Challenge/` project directory. Do not copy or independently
redefine common normalization.

```python
from src.preprocessing import (
    iter_prepared_tsv,
    load_ground_truth,
    load_source,
    prepare_source,
    write_tsv,
)

source1 = prepare_source(
    load_source("student_resource/dataset/train/train_source1.tsv", "S1")
)
truth = load_ground_truth(
    "student_resource/dataset/train/train_ground_truth.tsv"
)
```

`load_source(path, expected_source=None)` reads a TSV with `sep="\t"`,
preserves string values and blank cells, and validates source columns and IDs.
`prepare_source(frame, expected_source=None)` preserves the raw fields and adds
order-preserving `norm_name`, `norm_address`, `norm_country`, their
`name_norm`/`address_norm`/`country_norm` aliases, compact and sorted-token
views, address-number signatures, source identity, lengths, token counts, and
missingness flags.

Normalization uses Unicode NFKC, case-folding, and punctuation/symbol
replacement with spaces while preserving letters, combining marks, and numbers
from all scripts. It does not remove legal suffixes, reorder the normalized
text, translate/transliterate, infer missing data, geocode, or perform external
identity lookups. Country values are not whitelisted. Compact and sorted forms
are separate optional retrieval features; they do not replace the raw or
order-preserving normalized fields.

`load_ground_truth` parses comma-separated matches into Python lists; blank
cells become empty lists. `iter_prepared_tsv` validates and preprocesses source
TSVs in configurable chunks while checking duplicate IDs across chunks.
`write_tsv` writes UTF-8 TSV without an index.

Large-file example:

```python
from src.preprocessing import iter_prepared_tsv

for chunk in iter_prepared_tsv(
    "student_resource/dataset/train/train_source1.tsv",
    chunksize=100_000,
    expected_source="S1",
):
    process(chunk)
```

Run preprocessing checks and stream-validate a source file with:

```bash
python -m src.preprocessing --self-test
python -m src.preprocessing --validate \
  student_resource/dataset/train/train_source1.tsv \
  --expected-source S1 --chunksize 100000
```

Fit learned vectorizers/scalers within each experiment's training fold.
Chunked loading alone does not make corpus-wide blocking or TF-IDF out-of-core.

## 6. M2_E4 experiment

M2_E4's reusable multi-pass blocker is in
`src/blocking/multipass_blocking.py`; its train-only LightGBM validation runner
is in `experiments/M2_E4/run_experiment.py`. From this project root:

```bash
python -m pip install -r requirements.txt
python -m experiments.M2_E4.run_experiment
```

The runner uses a deterministic Source 1 match-count-stratified 20% validation
holdout (seed 42), indexes all S2/S3 training records in temporary SQLite, and
fits on a reproducible cap of Source 1 training entities with capped,
deterministically sampled negatives. It does not load the test split. See
`experiments/M2_E4/README.md` for options and artifact details. Validation
outputs are not test submissions.

## 7. Experiment reporting and constraints

Record each experiment's owner, configuration, fixed validation split,
blocking strategy, candidate count/reduction/recall, model features, macro
F0.5, precision/recall, singleton results, runtime, and known limitations.
Use the same validation split and metric implementation when comparing
experiments.

External business identity lookup and data augmentation are prohibited,
including business registries, entity-resolution services, geocoding APIs, and
internet-based business data. The final model must meet the challenge's stated
MIT/Apache 2.0 license and 8-billion-parameter limit. Follow the official
challenge materials if they differ from this README.

Keep datasets, credentials, caches, model checkpoints, and large generated
artifacts out of Git. Use fixed seeds where applicable and keep the final
inference process reproducible.
