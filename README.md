# demand-forecast-ai

> **Industrial energy demand forecasting with CNN-LSTM + LLM-powered anomaly explanation**
> Steel factory dataset · FastAPI backend · Claude API integration · Real-time HTML dashboard

---

## What This Project Does

This system forecasts hourly energy consumption in a steel manufacturing plant and automatically explains anomalies using Claude AI.

```
CSV Input (ds, y, load_type)
         ↓
   Feature Engineering
   (31 features: cyclical encoding, lags, rolling stats, load type)
         ↓
   ┌──────────────┐   ┌─────────────┐   ┌─────────────┐
   │  CNN-LSTM    │   │    LSTM     │   │  Prophet v3 │
   │  MAE 20.5   │   │  MAE 22.0  │   │  MAE 5.8   │
   │  (DEFAULT)   │   │  (baseline) │   │  (optional) │
   └──────┬───────┘   └──────┬──────┘   └──────┬──────┘
          └─────────────────┬──────────────────┘
                      Anomaly Detection
                   (rolling z-score, 24h window)
                            ↓
                   Anomaly? ──YES──→ Claude API
                            ↓              ↓
                         NO               Explanation + Recommended Action
                            ↓
                     FastAPI Response → Dashboard
```

---

## Results

| Model | MAE | RMSE | Anomaly Rate | Notes |
|-------|-----|------|--------------|-------|
| **CNN-LSTM** | **20.5 kWh** (18.7%) | — | ~6% realistic | Default model |
| LSTM baseline | 22.0 kWh (20.1%) | — | ~6% realistic | Simpler, comparable |
| Prophet v3 | **5.8 kWh** (5.3%) | — | Variable | Lower MAE but requires load_type in advance |

- Dataset: Steel Industry Energy Consumption (Kaggle) — 35,039 rows at 15-min intervals → 8,760 hourly
- Training: Jan–Sep 2018 · Test: Oct–Dec 2018
- Target: `Usage_kWh` range 9.8–553.2 kWh, mean 109.6 kWh
- CNN-LSTM improves 6.7% over pure LSTM baseline

---

## Project Structure

```
demand-forecast-ai/
├── api/
│   ├── main.py               # FastAPI app with CORS
│   ├── forecast.py           # POST /forecast/predict endpoint
│   ├── explain.py            # Claude API anomaly explanation
│   └── schemas.py            # Pydantic models
├── dashboard/
│   └── index.html            # Single-file HTML dashboard (Chart.js)
├── data/
│   ├── raw/                  # Steel_industry_data.csv
│   └── processed/            # steel_clean.csv, steel_features.csv
├── models/
│   ├── prophet/
│   │   ├── train.py          # Prophet v3 (logistic growth + regressors)
│   │   └── evaluate.py
│   └── lstm/
│       ├── model.py          # DemandLSTM + CNNLSTMDemand architectures
│       ├── train.py          # LSTM training loop with early stopping
│       ├── train_cnn_lstm.py # CNN-LSTM training + side-by-side benchmark
│       └── evaluate.py
├── pipeline/
│   ├── clean.py              # 15min → hourly resampling
│   └── features.py           # 31-feature engineering
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Rothvichea/demand-forecast-ai.git
cd demand-forecast-ai

conda create -n industrial-ai python=3.10
conda activate industrial-ai
pip install -r requirements.txt
```

### 2. Environment

```bash
cp .env.example .env
# Add your Anthropic API key to .env
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Data Pipeline

```bash
# Download dataset from Kaggle: csafrit2/steel-industry-energy-consumption
# Place Steel_industry_data.csv in data/raw/

python pipeline/clean.py      # 35039 rows → 8760 hourly
python pipeline/features.py   # builds 31 features
```

### 4. Train Models

```bash
# Train CNN-LSTM (default, recommended)
python -m models.lstm.train_cnn_lstm
# → saves: models/lstm/best_cnn_lstm.pt, best_lstm.pt, scaler_X.pkl, scaler_y.pkl
# → prints benchmark: CNN-LSTM vs LSTM comparison table

# Train Prophet (optional)
python -m models.prophet.train
# → saves: models/prophet/prophet_model.pkl
```

### 5. Run API

```bash
uvicorn api.main:app --reload
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 6. Open Dashboard

```bash
xdg-open dashboard/index.html
# Upload a CSV and select CNN-LSTM (default), LSTM, or Prophet
```

---

## Model Architecture

### CNN-LSTM (Default)

```
Input: (batch, seq_len=48, n_features=31)
  ↓
permute → (batch, 31, 48)
  ↓
Conv1d(31 → 64, kernel=3) + BatchNorm + ReLU + Dropout(0.15)
Conv1d(64 → 128, kernel=3) + BatchNorm + ReLU
MaxPool1d(2) → (batch, 128, 24)   # halves sequence length
  ↓
permute → (batch, 24, 128)
  ↓
LSTM(hidden=128, layers=2, dropout=0.3)
  ↓
Linear(128 → 64) + ReLU + Dropout(0.3)
Linear(64 → 1)
  ↓
Output: predicted kWh (next hour)
```

Why CNN first: Conv layers detect sharp idle→production transitions (ramp events) faster than LSTM gates. MaxPool compresses redundancy. BatchNorm handles the wide scale range (idle ~10 kWh vs production ~400 kWh).

### LSTM Baseline

```
Input: (batch, seq_len=48, n_features=31)
  ↓
LSTM(hidden=128, layers=2, dropout=0.3)
  ↓
Linear(128 → 32) + ReLU
Linear(32 → 1)
  ↓
Output: predicted kWh (next hour)
```

### 31 Features

- Cyclical encoding: `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`
- Binary flags: `is_weekend`, `is_night`, `is_peak`, `is_morning`, `is_afternoon`
- Lag features: `lag_1/2/3/6/12/24/48/168`
- Rolling stats: `roll_mean_4/24/168`, `roll_std_4/24`, `roll_max_24`, `roll_min_24`
- Deltas: `delta_1`, `delta_24`
- Load type: `load_light`, `load_medium`, `load_maximum`

### Anomaly Detection

```python
errors    = |actual - predicted|
roll_mean = rolling_mean(errors, window=24)
roll_std  = rolling_std(errors, window=24)
threshold = max(roll_mean + 2 * roll_std, 10.0)   # 10 kWh floor
anomaly   = error > threshold
```

---

## Claude API Integration

Claude is called **only when an anomaly is detected** — zero API cost for normal operation.

```python
result = explain_anomaly(
    anomaly_info = {
        "is_anomaly": True,
        "actual": 281.66,
        "forecast": 211.25,
        "deviation_pct": 25.0
    },
    context = {
        "timestamp": "2018-10-29 09:00",
        "hour": 9,
        "day_of_week": "Monday"
    }
)
```

**Example Claude response:**
```json
{
  "severity": "medium",
  "explanation": "Energy consumption at 9 AM on Monday was 25% higher than forecasted, using 281.66 kWh instead of the expected 211.25 kWh.",
  "recommended_action": "Check production schedules and equipment startup logs for Monday morning to identify which systems caused the increased demand."
}
```

---

## API Reference

### POST `/forecast/predict`

```bash
curl -X POST "http://localhost:8000/forecast/predict?model=cnn_lstm" \
  -F "file=@test_october.csv"
```

**Query params:** `model=cnn_lstm` (default) | `model=lstm` | `model=prophet`

**CSV format:**
```
ds,y,load_type
2018-10-01 00:00:00,85.3,Light_Load
2018-10-01 01:00:00,79.1,Light_Load
```

**Response:**
```json
{
  "model": "cnn_lstm",
  "n_points": 63,
  "mae": 20.5,
  "forecast": [
    {
      "ds": "2018-10-29T08:00:00",
      "y_actual": 281.66,
      "yhat": 211.25,
      "anomaly": true
    }
  ],
  "explanation": {
    "anomaly_detected": true,
    "severity": "medium",
    "explanation": "...",
    "recommended_action": "..."
  }
}
```

---

## Docker

```bash
docker-compose up --build
# API at http://localhost:8000
```

---

## MLflow Tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Experiments tracked:
- `demand-forecast-lstm` — loss curves, MAE, hyperparameters per epoch
- `demand-forecast-prophet-v3` — MAE, regressor list

---

## Requirements

```
python=3.10
torch, prophet, scikit-learn, pandas, numpy
fastapi, uvicorn, pydantic, python-dotenv
anthropic, mlflow
```

---

## Dataset

**Steel Industry Energy Consumption**
Source: [Kaggle — csafrit2/steel-industry-energy-consumption](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption)
License: CC BY 4.0

---

## Author

**Rothvichea CHEA** — Mechatronics & AI Engineer
[rothvicheachea.netlify.app](https://rothvicheachea.netlify.app)
[GitHub](https://github.com/Rothvichea)
