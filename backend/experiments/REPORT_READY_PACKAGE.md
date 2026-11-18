# PathSnap — Report-Ready Technical Results Package

All numbers, tables, charts, and interactive Folium maps below are generated directly from verified experiment runs in this repository (`backend/experiments/`). Zero metrics are fabricated or computed on evaluator ground-truth labels.

---

## Section 2.2 — Dataset Understanding, EDA & Feature Engineering

### Dataset Partitions & Leakage Prevention
- **Training (`train/`, 80 trips):** 184,648 reference-matched GPS observations producing **1,649,424 candidate rows** (184,648 positive = 11.19%, 1,464,776 negative = 88.81%). Split strictly at the **trip level** (`GroupShuffleSplit`, `test_size=0.2`, `random_state=42`): **64 train trips** (1,317,582 rows) and **16 held-out validation trips** (331,842 rows); zero GPS points from any trip cross partitions.
- **Student Testing (`student_test/`, 10 trips):** Held out strictly for development-stage model comparison and HMM/Viterbi evaluation. Never read during training, cross-validation, or Bayesian hyperparameter optimization.
- **Evaluator Testing (10 reserved trips):** Untouched during development; processed at runtime by Laptop 1's FastAPI service via `PATHSNAP_EVAL_DIR` / `evaluator_test/`.

### Data Quality Audit (`backend/experiments/train_data_audit.csv`, `student_test_data_audit.csv`)
- **Missing / Non-finite values:** 0 NaN/Inf entries across all 90 trips (`.track`, `.nodes`, `.arcs`, `.route`), with runtime finite-value filtering in `extract_trip_candidates()`.
- **Road network integrity:** Zero isolated nodes; node IDs and arc IDs are 0-based row indices. Reference `.route` sequences have ~99.9% directed endpoint continuity (`arcs[r[:-1], 1] == arcs[r[1:], 0]`).
- **GPS sampling & anomalies:** Nominally 1 Hz, with irregular timestamp gaps (e.g., up to 2,161 s in `train` and 210 s in `student_test` trip `00000065`). GPS observations do not align 1:1 with `.route` rows, so training labels are derived via monotonic DP alignment (`route_advance_penalty=0.15`, `label_max_distance_m=45.0 m`).

### Canonical 17-Feature Schema (`FEATURE_COLUMNS`)
1. Spatial geometry (6): `distance_m`, `signed_lateral_offset_m`, `projection_fraction`, `distance_to_arc_start_m`, `distance_to_arc_end_m`, `arc_length_m`
2. Candidate neighborhood context (4): `candidate_rank_distance`, `candidate_count_at_point`, `is_nearest_candidate`, `is_within_search_radius`
3. Kinematic / GPS motion (4): `gps_step_distance_m`, `gps_speed_mps`, `time_delta_s`, `gps_bearing_valid`
4. Directional & topological (3): `heading_alignment_cos`, `heading_alignment_sin`, `is_self_loop`

Candidate search uses a 60 m corridor STRtree query retaining up to 16 nearest arcs (fallback to nearest graph arc if none in radius), yielding a **0.03% candidate miss rate** on `student_test`.

---

## Section 3 — Model Experimentation & Comparison (5 Baseline Models + Tuned Final Model)

### 3.1 Consolidated Model Comparison Table (Verified Real Results)

| Model | Algorithm Family | Key Configuration | 16-Trip Validation Macro F1 | Student-Test Macro F1 (10 Trips) | Student-Test Top-1 Acc (Raw / HMM) | Student-Test Route Continuity (Raw / HMM) | Student-Test Log Loss | Mean ML Inference Time / Trip |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **HistGradientBoosting (Optuna Tuned)** | Boosting (TPE Tuned) | `max_iter=100, lr=0.095, max_leaf_nodes=31, min_samples_leaf=20, l2=6.245, max_bins=128, balance=False` | **0.8817** | **0.9061** | `84.53%` / **`92.25%`** | `74.71%` / **`99.99%`** | **0.0803** | `0.0351s` |
| **HistGradientBoosting (Baseline)** | Boosting | `max_iter=150, lr=0.1, max_leaf_nodes=31, min_samples_leaf=30, l2=1.0, balance=True` | `0.8430` | `0.8687` | **`84.81%`** / **`92.85%`** | `74.78%` / **`99.99%`** | `0.1402` | `0.0450s` |
| **Random Forest** | Ensemble (Bagging) | `n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2, class_weight=balanced` *(5-fold GroupKFold CV F1: `0.8674 ± 0.0160`)* | `0.8584` | `0.8870` | `84.44%` / **`92.67%`** | `74.75%` / **`99.99%`** | `0.1153` | `0.1188s` |
| **Decision Tree** | Tree-based | `max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight=balanced` *(5-fold GroupKFold CV F1: `0.8398 ± 0.0145`)* | `0.8354` | `0.8582` | `82.82%` / **`90.49%`** | `71.01%` / **`99.89%`** | `0.1829` | **`0.0025s`** |
| **K-Nearest Neighbors (KNN)** | Instance-based | `StandardScaler + k=15, weights=distance, kd_tree` (60,000 trip-stratified train reference rows) | `0.8524` | `0.8778` | `81.35%` / **`89.53%`** | `71.68%` / **`99.80%`** | `0.1355` | `5.1370s` |
| **Linear Regression (Candidate Scorer)** | Linear (OLS Ranking) | `StandardScaler + OLS LinearRegression` (continuous candidate score ranking + per-point softmax `T=5.0` for HMM) | `N/A` *(Val Top-1: `60.25%`, $R^2=0.3966$)* | `0.7528` *(Top-1 rank F1)* | `56.52%` / **`85.66%`** | `46.49%` / **`99.98%`** | `N/A` *(non-probabilistic regressor)* | `0.1166s` |

### Key Model Selection Insights
- **Primary selection criteria:** Combination of **Macro F1**, **Log Loss** (crucial for calibrated HMM emissions), **HMM Top-1 Accuracy & Route Continuity**, **artifact footprint**, and **real-time inference latency**.
- **Final Selected Model:** **HistGradientBoosting (Optuna Tuned)** achieves the highest Macro F1 (`0.8817` val / `0.9061` student-test), lowest Log Loss (`0.0803`), ultra-compact artifact (`0.16 MB` vs `104.14 MB` for Random Forest), and **0.035 s** mean ML inference per trip (146× faster than KNN).
- **Overfitting analysis:** Decision Tree shows noticeable train-vs-validation gap (`0.8658` train F1 vs `0.8354` val F1; `0.1488` vs `0.2609` Log Loss); Random Forest narrows the variance (`0.9057` train vs `0.8584` val; `0.1654` val Log Loss); tuned HistGradientBoosting with `l2_regularization=6.245` generalizes best (`0.8817` val / `0.9061` student-test).

### Generated Model Visualizations (`backend/experiments/plots/`)
- `all_models_macro_f1_comparison.png`
- `all_models_hmm_vs_raw_comparison.png`
- `all_models_log_loss_comparison.png`
- `optuna_hgb_optimization_history.png`
- `confusion_matrix_{decision_tree,random_forest,hist_gradient_boosting,knn,hist_gradient_boosting_tuned}.png`

---

## Section 4 — Bayesian Hyperparameter Optimization (`backend/src/tune_hgb.py`)

- **Optimizer:** Optuna `TPESampler` (Tree-structured Parzen Estimator, `seed=42`).
- **Objective:** Maximize held-out **16-trip Validation Macro F1-score** (strictly zero `student_test` or evaluator leakage).
- **Trials & Cost:** **10 completed trials** in **177.0 seconds** (all trials logged in `backend/experiments/hgb_optuna_trials.csv`).
- **Search Space:**
  - `balance_classes`: `{True, False}`
  - `max_iter`: `[100, 300]` (step 50)
  - `learning_rate`: log-uniform `[0.03, 0.20]`
  - `max_leaf_nodes`: `[15, 95]` (step 16)
  - `min_samples_leaf`: `[15, 100]` (step 5)
  - `l2_regularization`: log-uniform `[1e-3, 10.0]`
  - `max_bins`: `{128, 255}`
- **Best Trial (#3, Trial 04/10):** `balance_classes=False, max_iter=100, learning_rate=0.0950, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=6.2451, max_bins=128`.

### Untuned vs. Tuned HistGradientBoosting Comparison

| Metric | Partition | Untuned HGB | Optuna-Tuned HGB | Change |
|---|---|---:|---:|---:|
| Macro F1-score | Validation (16 trips) | `0.8430` | **`0.8817`** | **`+0.0387`** |
| Top-1 Accuracy | Validation (16 trips) | `80.05%` | **`80.15%`** | `+0.10%` |
| Log Loss | Validation (16 trips) | `0.1901` | **`0.1200`** | **`-0.0701`** |
| Macro F1-score | Student-Test (10 trips) | `0.8687` | **`0.9061`** | **`+0.0374`** |
| Log Loss | Student-Test (10 trips) | `0.1402` | **`0.0803`** | **`-0.0599`** |
| Clf-Only Top-1 Accuracy | Student-Test (10 trips) | `84.81%` | `84.53%` | `-0.27%` |
| HMM/Viterbi Top-1 Accuracy | Student-Test (10 trips) | `92.85%` | `92.25%` | `-0.61%` |
| Mean ML Inference Time | Student-Test (10 trips) | `0.0422s` | **`0.0300s`** | **`-28.9%`** |
| Artifact Size | Disk | `0.24 MB` | **`0.16 MB`** | **`-33.3%`** |

*Technical finding:* Disabling artificial 50/50 inverse-frequency reweighting (`balance_classes=False`) with stronger L2 regularization (`6.245`) eliminated probability over-confidence on negative road candidates, dramatically improving Macro F1 and cutting Log Loss nearly in half, while slightly trading ~0.6% recall inside Viterbi on dense urban trips (`00000077`).

---

## Section 5 — HMM & Viterbi Route Reconstruction (`backend/src/trip_pipeline.py`)

- **Hidden States:** Up to 16 spatial candidate road arcs $c \in C_t$ per GPS observation $z_t$.
- **Emission Log-Probability:** $\log P(z_t \mid c) = 1.2 \cdot \log(\text{clip}(p_{\text{ML}}(c), 10^{-6}, 1-10^{-6})) - \frac{d(z_t, c)^2}{2 \sigma_z^2}$ ($\sigma_z = 25\text{ m}$). For Linear Regression (`regression_scorer`), continuous OLS scores are converted per point via temperature-scaled softmax ($\tau = 5.0$).
- **Transition Log-Probability:** Directed corridor road graph with bounded Dijkstra shortest paths comparing road distance $d_{\text{route}}$ against planar GPS displacement $d_{\text{gps}}$: $-\frac{|d_{\text{route}} - d_{\text{gps}}|}{\beta}$, where $\beta$ and search cutoff expand automatically during GPS gaps ($\Delta t > 3\text{ s}$). Unreachable/disconnected transitions receive a finite fallback penalty (`-22.0 - 0.02 * d_gps`) so Viterbi never crashes on map gaps or GPS dropouts.
- **Impact (Student-Test 10 Trips):**
  - **HistGradientBoosting (Tuned):** Top-1 `84.53% → 92.25%` (`+7.72%`), Continuity `74.71% → 99.99%` (`+25.28%`), Route F1 `0.8091 → 0.9470` (`+0.1379`).
  - **Random Forest:** Top-1 `84.44% → 92.67%` (`+8.23%`), Continuity `74.75% → 99.99%`.
  - **Linear Regression (Scorer):** Top-1 `56.52% → 85.66%` (**`+29.14%`**), Continuity `46.49% → 99.98%` (**`+53.49%`**) — demonstrating how topological HMM transitions rescue noisy local scores.

---

## Section 6 — 10-Trip Student-Test Verification Table (Final Model: `hist_gradient_boosting_tuned`)

| Trip ID | GPS Points | Candidates | Predicted Arcs | Status | HMM Status | Clf Top-1 | **HMM Top-1** | **HMM Continuity** | **HMM Route F1** | Total Runtime |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| `00000000` | 2,503 | 13,866 | 218 | OK | OK | 92.69% | **97.36%** | **100.00%** | 0.9182 | 1.24s |
| `00000007` | 1,354 | 5,911 | 123 | OK | OK | 95.20% | **99.34%** | **100.00%** | 0.9837 | 0.61s |
| `00000009` | 1,049 | 14,383 | 480 | OK | OK | 82.35% | **91.79%** | **100.00%** | 0.9483 | 0.89s |
| `00000040` | 2,995 | 17,873 | 410 | OK | OK | 85.24% | **93.82%** | **100.00%** | 0.9778 | 1.51s |
| `00000052` | 262 | 2,771 | 93 | OK | OK | 93.13% | **98.47%** | **100.00%** | 0.9894 | 0.21s |
| `00000056` | 3,958 | 45,212 | 1,374 | OK | OK | 87.36% | **96.64%** | **99.85%** | 0.9884 | 3.15s |
| `00000065` | 768 | 10,273 | 169 | OK | OK | 80.86% | **87.50%** | **100.00%** | 0.9699 | 0.58s |
| `00000076` | 1,486 | 21,404 | 529 | OK | OK | 79.74% | **90.98%** | **100.00%** | 0.8417 | 1.71s |
| `00000077` | 3,911 | 49,403 | 625 | OK | OK | 65.46% | **76.53%** | **100.00%** | 0.8948 | 2.83s |
| `00000091` | 874 | 10,770 | 261 | OK | OK | 83.30% | **90.05%** | **100.00%** | 0.9580 | 0.60s |
| **Mean / Total** | **19,160** | **191,866** | **4,282** | **10/10 OK** | **10/10 OK** | **84.53%** | **92.25%** | **99.99%** | **0.9470** | **1.33s** |

Interactive Folium HTML maps generated in `backend/experiments/maps/`:
- `trip_00000000_hist_gradient_boosting_tuned_hmm_map.html`
- `trip_00000007_hist_gradient_boosting_tuned_hmm_map.html`
- `trip_00000009_hist_gradient_boosting_tuned_hmm_map.html`

---

## Section 7 — API, UI & Two-Laptop Deployment Checklist
- **Laptop 1 (FastAPI + Models + HMM/Viterbi):** `backend\.venv\Scripts\uvicorn.exe backend.api.main:app --host 0.0.0.0 --port 8000` (keep terminal visible for evaluator live request logs).
- **Laptop 2 (React + Leaflet UI):** Set `VITE_API_BASE_URL=http://<LAPTOP1_IP>:8000` in `frontend/.env` and run `npm run dev -- --host`.
- **Action needed during demo for Report Section 7:** Capture screenshots of (1) the working React UI on Laptop 2 showing a reconstructed route and metrics, and (2) the live FastAPI CMD logs on Laptop 1 receiving the inference requests.
