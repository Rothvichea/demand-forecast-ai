# ⚡ demand-forecast-ai

> **Industrial energy demand forecasting with LSTM neural network + LLM-powered anomaly explanation**  
> Steel factory dataset · FastAPI backend · Claude API integration · Real-time HTML dashboard

---

## 🎯 What This Project Does

This system forecasts hourly energy consumption in a steel manufacturing plant and automatically explains anomalies using Claude AI.

```
CSV Input (ds, y, load_type)
         ↓
   Feature Engineering
   (31 features: cyclical encoding, lags, rolling stats, load type)
         ↓
   ┌─────────────┐     ┌─────────────┐
   │  LSTM v2    │     │  Prophet v3 │
   │ MAE 19 kWh  │     │ MAE 5.8 kWh │
   └──────┬──────┘     └──────┬──────┘
          └──────────┬─────────┘
               Anomaly Detection
                     ↓
            Anomaly? ──YES──→ Claude API
                     ↓              ↓
                  NO               Explanation + Recommended Action
                     ↓
              FastAPI Response → Dashboard
```

**Key insight from this project:** Prophet achieved lower MAE (5.8 kWh) but failed in production due to zero predictions during low-demand periods. LSTM showed higher MAE (19 kWh) but correctly modeled 24/7 factory behavior — proving that metrics alone don't tell the full story.

---

## 📊 Results

| Model | MAE | Anomaly Rate (Oct 2018) | Production Ready |
|-------|-----|------------------------|-----------------|
| Prophet v3 | **5.8 kWh** (5.3% of mean) | ❌ 99% false alarms at night | ❌ No |
| LSTM v2 | **19.5 kWh** (17.8% of mean) | ✅ 6.3% realistic detection | ✅ Yes |

- Dataset: Steel Industry Energy Consumption (Kaggle) — 35,039 rows at 15-min intervals → 8,760 hourly
- Training period: Jan–Sep 2018 · Test period: Oct–Dec 2018
- Target: `Usage_kWh` range 9.8–553.2 kWh, mean 109.6 kWh

---

## 🏗️ Project Structure

```
demand-forecast-ai/
├── api/
│   ├── main.py          # FastAPI app with CORS
│   ├── forecast.py      # POST /forecast/predict endpoint
│   ├── explain.py       # Claude API anomaly explanation
│   └── schemas.py       # Pydantic models
├── dashboard/
│   └── index.html       # Single-file HTML dashboard (Chart.js)
├── data/
│   ├── raw/             # Steel_industry_data.csv
│   └── processed/       # steel_clean.csv, steel_features.csv
├── models/
│   ├── prophet/
│   │   ├── train.py     # Prophet v3 training (logistic growth + 24 regressors)
│   │   └── evaluate.py
│   └── lstm/
│       ├── model.py     # DemandLSTM architecture
│       ├── train.py     # Training loop with early stopping
│       └── evaluate.py
├── pipeline/
│   ├── clean.py         # 15min → hourly resampling
│   └── features.py      # 31-feature engineering
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

---

## 🚀 Quick Start

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
# Train LSTM (recommended — production ready)
python -m models.lstm.train
# → saves: models/lstm/best_lstm.pt, scaler_X.pkl, scaler_y.pkl

# Train Prophet (optional — better MAE but night prediction issues)
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
# Upload test_october.csv and select LSTM model
```

---

## 🧠 Model Architecture

### LSTM v2

```
Input: (batch, seq_len=48, n_features=31)
  ↓
LSTM(hidden=128, layers=2, dropout=0.3)
  ↓
Linear(128 → 32) + ReLU
  ↓
Linear(32 → 1)
  ↓
Output: predicted kWh (next hour)
```

**31 Features:**
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
threshold = max(roll_mean + 3σ, actual * 0.20)
anomaly   = error > threshold
```

---

## 🤖 Claude API Integration

Claude is called **only when an anomaly is detected** — zero API cost for normal operation.

```python
# Called for the worst anomaly per request (~5-15% of inputs)
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
  "explanation": "Energy consumption at 9 AM on Monday was 25% higher than forecasted,
                  using 281.66 kWh instead of the expected 211.25 kWh.",
  "recommended_action": "Check production schedules and equipment startup logs for
                          Monday morning to identify which systems caused the increased demand."
}
```

---

## 🌐 API Reference

### POST `/forecast/predict`

```bash
curl -X POST "http://localhost:8000/forecast/predict?model=lstm" \
  -F "file=@test_october.csv"
```

**Query params:** `model=lstm` or `model=prophet`

**CSV format:**
```
ds,y,load_type
2018-10-01 00:00:00,85.3,Light_Load
2018-10-01 01:00:00,79.1,Light_Load
```

**Response:**
```json
{
  "model": "lstm",
  "n_points": 63,
  "mae": 19.46,
  "forecast": [
    {
      "ds": "2018-10-29T08:00:00",
      "yhat": 185.42,
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

## 🐳 Docker

```bash
docker-compose up --build
# API at http://localhost:8000
```

---

## 📈 MLflow Tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

Experiments tracked:
- `demand-forecast-lstm` — loss curves, MAE, hyperparameters
- `demand-forecast-prophet-v3` — MAE, regressor list

---

## 🔭 Roadmap

- [ ] **CNN-LSTM hybrid** — CNN extracts local spike patterns, LSTM handles temporal dependencies (target: MAE < 12 kWh)
- [ ] Fix Prophet night-hour CI collapse (logistic floor tuning)
- [ ] Qt6 C++ real-time dashboard
- [ ] Docker multi-stage build with TensorRT optimization

---

## 📦 Requirements

```
python=3.10
torch, prophet, scikit-learn, pandas, numpy
fastapi, uvicorn, pydantic, python-dotenv
anthropic, mlflow
```

---

## 📄 Dataset

**Steel Industry Energy Consumption**  
Source: [Kaggle — csafrit2/steel-industry-energy-consumption](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption)  
License: CC BY 4.0

---

## 👤 Author

**Rothvichea CHEA** — Mechatronics & AI Engineer  
🌐 [rothvicheachea.netlify.app](https://rothvicheachea.netlify.app)  
💼 [GitHub](https://github.com/Rothvichea)
