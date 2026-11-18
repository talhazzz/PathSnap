# Shared dataset handoff

Run the pipeline from the repository root with the configured `.venv`:

```sh
.venv/bin/python -m ml.preprocessing.build_shared_dataset
```

Outputs:

- `artifacts/shared/candidate_features.csv`: labeled candidate rows for all usable GPS points in training trips; includes the reproducible `split` column.
- `artifacts/shared/candidate_features_sample.csv`: first 5,000 rows as a lightweight sample.
- `artifacts/shared/trip_split.csv`: one row per source trip and its train/validation assignment.
- `artifacts/shared/shared_pipeline_summary.json`: dimensions, label counts, per-trip match/coverage diagnostics, and split counts.
- `artifacts/feature_schema.md`: source conventions, feature definitions, target method, and model-input exclusions.

Talha and Mithun should load `candidate_features.csv`, set `y = frame["label"]`, and set `X = frame[FEATURE_COLUMNS]` from `ml.preprocessing.build_shared_dataset`. Train with `split == "train"`; evaluate using only `split == "validation"`. Group/report by `trip_id` as needed, and keep `trip_id`, `point_idx`, and `candidate_arc_id` for joining predictions back to trajectories. Do not include any identifier, `split`, or `label` column in `X`. The CSV contains only source `train/` trips; `student_test/` was not read.