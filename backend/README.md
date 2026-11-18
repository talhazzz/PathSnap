# PathSnap Backend — ML Models, HMM/Viterbi & FastAPI Deployment (Laptop 1)

This directory contains the PathSnap **Backend API, Decision Tree & Random Forest ML pipeline, Optuna Bayesian hyperparameter optimization, HMM/Viterbi route reconstruction engine, and deployment scripts** for the Two-Laptop Distributed Evaluation setup.

---

## Architecture (Two-Laptop Deployment)

| Laptop | Component | Responsibilities |
|---|---|---|
| **Laptop 1** | **Model & FastAPI Backend (`backend/`)** | Hosts trained classifiers (`.joblib`), 17-feature candidate extractor, directed-graph HMM/Viterbi route decoder, road-network data, Folium map generator, and server-side evaluation (never exposes ground-truth labels to model or UI). |
| **Laptop 2** | **Evaluator UI (`frontend/`)** | Connects to Laptop 1 over LAN (`http://<LAPTOP1_IP>:8000`), displays the 10 evaluation trips, triggers route reconstruction, and renders predicted routes and server-side evaluation metrics. |

---

## Quick Setup & Reproduction

Run all commands from the repository root (`PathSnap_repo/`):

### 1. Environment Setup
```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### 2. Place Raw Dataset
Extract `Student folder.zip` into:
```text
backend/data/raw/Student folder/
├── train/         (80 trips — used for training, GroupKFold CV, and Bayesian tuning)
├── student_test/  (10 trips — development-stage evaluation only)
└── evaluator_test/ (optional — drop the 10 reserved evaluator trips here at demo time)
```

### 3. Train & Evaluate All 5 Models (Trip-Level Splits, Zero Leakage)
```powershell
# Decision Tree & Random Forest (5-fold GroupKFold CV + validation + student_test)
backend\.venv\Scripts\python.exe -m backend.src.train_models

# HistGradientBoosting (ml-dev config), KNN (scaled KD-tree), Linear Regression (OLS candidate scorer)
backend\.venv\Scripts\python.exe -m backend.src.train_extra_models
```
Outputs:
- `backend/models/{decision_tree,random_forest,hist_gradient_boosting,knn,linear_regression}.joblib`
- `backend/experiments/decision_tree_random_forest_training.csv`
- `backend/experiments/model_comparison_mithun.json`
- `backend/experiments/extra_models_metrics.json`

### 4. Run Optuna TPE Bayesian Hyperparameter Optimization (HistGradientBoosting)
```powershell
backend\.venv\Scripts\python.exe -m backend.src.tune_hgb
```
Outputs:
- `backend/models/hist_gradient_boosting_tuned.joblib`
- `backend/experiments/hgb_optuna_trials.csv`
- `backend/experiments/hgb_tuning_summary.json`

### 5. Evaluate All Models Across All 10 Evaluation Trips (Classifier-Only vs HMM/Viterbi)
```powershell
backend\.venv\Scripts\python.exe -m backend.src.evaluate_all_models
```
Outputs:
- `backend/experiments/all_models_comparison.csv`
- `backend/experiments/student_test_10trips_final_model.csv`
- `backend/experiments/all_models_student_test_summary.json`

### 6. Generate Evaluation Plots & Interactive Folium Maps
```powershell
backend\.venv\Scripts\python.exe -m backend.src.generate_artifacts
```
Outputs:
- `backend/experiments/plots/*.png` (5-model + tuned comparisons, confusion matrices, Optuna TPE history, HMM vs raw model charts)
- `backend/experiments/maps/*.html` (interactive Folium maps with GPS points, road network, reference route, predicted HMM route, legend, and layer controls)

### 6. Run Automated End-to-End Tests
```powershell
backend\.venv\Scripts\pytest.exe backend/tests/test_backend_e2e.py -v
```

---

## Live Two-Laptop LAN Deployment (3:00 PM Final Evaluation)

### On Laptop 1 (API Server)
1. **Find Laptop 1's LAN IP** (both laptops must be on the same Wi-Fi / hotspot):
   ```powershell
   ipconfig
   ```
   Look for the `IPv4 Address` under your Wi-Fi adapter (e.g., `192.168.x.y`).

2. **Point to the 10 Reserved Evaluator Trips** (optional — if evaluators provide a folder path):
   ```powershell
   $env:PATHSNAP_EVAL_DIR = "C:\path\to\evaluator_trips"
   ```
   *(Or copy the 10 evaluator trip folders into `backend/data/raw/Student folder/evaluator_test/`. If absent, the API defaults to `student_test`.)*

3. **Start FastAPI on `0.0.0.0:8000`** (keep this CMD/PowerShell terminal visible for the evaluator to verify live inference logs!):
   ```powershell
   backend\.venv\Scripts\uvicorn.exe backend.api.main:app --host 0.0.0.0 --port 8000
   ```

4. **If Windows Firewall blocks incoming LAN requests on port 8000** (run PowerShell as Administrator once):
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```

### On Laptop 2 (Evaluator UI)
Point the frontend API base URL to:
```text
http://<LAPTOP1_IP>:8000
```
Quick connectivity check from Laptop 2's browser or terminal:
```text
http://<LAPTOP1_IP>:8000/health
http://<LAPTOP1_IP>:8000/trips
http://<LAPTOP1_IP>:8000/docs
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info, active evaluation partition, default model |
| `GET` | `/health` | Healthcheck and loaded model list |
| `GET` | `/models` | Lists available `.joblib` models and artifact sizes |
| `GET` | `/trips` | Lists the 10 evaluation trips with non-leaky telemetry (no ground truth exposed) |
| `GET` | `/trips/{trip_id}/geometry` | Returns raw GPS trajectory polyline for a trip (no ground truth exposed) |
| `POST` | `/predict` | Batch candidate-level scoring/prediction (`model_name`, `features` list of 17-feature dicts) |
| `POST` | `/predict/trip/{trip_id}` | Full end-to-end map matching: 17-feature candidate extraction -> ML classifier -> HMM/Viterbi decoding -> predicted route polyline + server-side accuracy (never leaks raw reference route) |
| `GET` | `/trips/{trip_id}/map` | Interactive Folium HTML map for the reconstructed trip route |
| `GET` | `/metrics/summary` | Returns recorded CV, student-test, Bayesian tuning, and HMM evaluation summaries |
| `POST` | `/ai/explain-trip` | Optional Gemini-powered AI Route Analyst explaining trip summary statistics (gracefully degrades if `GEMINI_API_KEY` is unset; never receives ground truth) |

---

## Optional Gemini AI Route Analyst Configuration

PathSnap includes an optional natural-language AI Route Analyst (`backend/src/gemini_explainer.py`, `POST /ai/explain-trip`) powered by the official `google-genai` SDK (`gemini-3.8-flash`). Core map matching is **100% independent** and continues working normally if no API key is set.

1. Copy `backend/.env.example` to `backend/.env` (`.env` is gitignored — never commit real keys).
2. Add your free-tier Gemini API key:
   ```text
   GEMINI_API_KEY=YOUR_KEY_HERE
   GEMINI_MODEL=gemini-3.8-flash
   ```
3. Restart FastAPI on Laptop 1. Check `GET /health` → `"ai_analyst_ready": true`.
