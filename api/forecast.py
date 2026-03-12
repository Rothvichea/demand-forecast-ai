import io
import pickle
import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from api.schemas import ForecastResponse, ForecastPoint, AnomalyExplanation
from api.explain import explain_anomaly as _explain
from models.lstm.model import DemandLSTM
from models.lstm.train import SEQ_LEN, HIDDEN, LAYERS, DROPOUT, make_sequences

router = APIRouter()

PROPHET_MODEL = "models/prophet/prophet_model.pkl"
LSTM_MODEL    = "models/lstm/best_lstm.pt"
SCALER_X      = "models/lstm/scaler_X.pkl"
SCALER_Y      = "models/lstm/scaler_y.pkl"

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_night", "is_peak", "is_morning", "is_afternoon",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "lag_24", "lag_48", "lag_168",
    "roll_mean_4", "roll_mean_24", "roll_mean_168",
    "roll_std_4", "roll_std_24",
    "roll_max_24", "roll_min_24",
    "delta_1", "delta_24",
    "load_light", "load_medium", "load_maximum"
]


# ── LOADERS ───────────────────────────────────────────────
def _load_prophet():
    with open(PROPHET_MODEL, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj["model"], obj["regressors"]
    return obj, []


def _load_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DemandLSTM(input_size=len(FEATURE_COLS),
                        hidden_size=HIDDEN, num_layers=LAYERS,
                        dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(LSTM_MODEL, map_location=device))
    model.eval()
    with open(SCALER_X, "rb") as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y, "rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y, device


# ── FEATURE BUILDER ───────────────────────────────────────
def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["load_light"]   = (df["load_type"] == "Light_Load").astype(float)
    df["load_medium"]  = (df["load_type"] == "Medium_Load").astype(float)
    df["load_maximum"] = (df["load_type"] == "Maximum_Load").astype(float)
    df["hour"]         = df["ds"].dt.hour
    df["dayofweek"]    = df["ds"].dt.dayofweek
    df["month"]        = df["ds"].dt.month
    df["is_weekend"]   = (df["dayofweek"] >= 5).astype(int)
    df["is_night"]     = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_peak"]      = ((df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)
    df["is_morning"]   = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]      = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]      = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["lag_1"]        = df["y"].shift(1)
    df["lag_2"]        = df["y"].shift(2)
    df["lag_3"]        = df["y"].shift(3)
    df["lag_6"]        = df["y"].shift(6)
    df["lag_12"]       = df["y"].shift(12)
    df["lag_24"]       = df["y"].shift(24)
    df["lag_48"]       = df["y"].shift(48)
    df["lag_168"]      = df["y"].shift(168)
    df["roll_mean_4"]  = df["y"].rolling(4).mean()
    df["roll_mean_24"] = df["y"].rolling(24).mean()
    df["roll_mean_168"]= df["y"].rolling(168).mean()
    df["roll_std_4"]   = df["y"].rolling(4).std()
    df["roll_std_24"]  = df["y"].rolling(24).std()
    df["roll_max_24"]  = df["y"].rolling(24).max()
    df["roll_min_24"]  = df["y"].rolling(24).min()
    df["delta_1"]      = df["y"].diff(1)
    df["delta_24"]     = df["y"].diff(24)
    df = df.dropna().reset_index(drop=True)
    return df


# ── MAIN ENDPOINT ─────────────────────────────────────────
@router.post("/predict", response_model=ForecastResponse)
async def predict(
    file:  UploadFile = File(...),
    model: str        = Query("lstm", enum=["lstm", "prophet"])
):
    contents = await file.read()
    try:
        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")), parse_dates=["ds"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    if not {"ds", "y", "load_type"}.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain columns: ds, y, load_type"
        )

    df = df.sort_values("ds").reset_index(drop=True)
    df = _build_features(df)

    points, mae = _predict_prophet(df) if model == "prophet" else _predict_lstm(df)

    # ── Claude explanation on worst anomaly ───────────────
    anomaly_points = [p for p in points if p.anomaly]
    explanation    = None

    if anomaly_points:
        worst      = max(anomaly_points, key=lambda p: p.yhat)
        ts         = pd.Timestamp(worst.ds)
        yhat_lower = worst.yhat_lower or worst.yhat * 0.7
        yhat_upper = worst.yhat_upper or worst.yhat * 1.3

        anomaly_info = {
            "is_anomaly":    True,
            "actual":        worst.yhat,
            "forecast":      worst.yhat * 0.75,
            "lower_bound":   yhat_lower,
            "upper_bound":   yhat_upper,
            "deviation_kwh": worst.yhat - worst.yhat * 0.75,
            "deviation_pct": 25.0
        }
        context = {
            "timestamp":   str(worst.ds),
            "load_type":   "unknown",
            "hour":        ts.hour,
            "day_of_week": ts.day_name()
        }
        exp_raw = _explain(anomaly_info, context)
        if exp_raw:
            explanation = AnomalyExplanation(
                anomaly_detected   = True,
                severity           = exp_raw.get("severity", "medium"),
                explanation        = exp_raw.get("explanation", ""),
                recommended_action = exp_raw.get("recommended_action", "")
            )

    return ForecastResponse(
        model=model, n_points=len(points),
        mae=mae, forecast=points, explanation=explanation
    )


# ── PROPHET PREDICT ───────────────────────────────────────
def _predict_prophet(df: pd.DataFrame):
    model, regressors = _load_prophet()

    split  = int(len(df) * 0.8)
    test   = df[split:].copy()

    future = test[["ds"] + regressors].copy()
    future["floor"] = 5.0
    future["cap"]   = 600.0
    for reg in regressors:
        future[reg] = future[reg].fillna(0.0)

    forecast   = model.predict(future)
    yhat       = np.clip(forecast["yhat"].values,       5.0, 600.0)
    yhat_lower = np.clip(forecast["yhat_lower"].values, 5.0, 600.0)
    yhat_upper = forecast["yhat_upper"].values
    actual     = test["y"].values
    ts         = test["ds"].values

    anomaly = (actual < yhat_lower) | (actual > yhat_upper)
    mae     = float(np.mean(np.abs(actual - yhat)))

    points = [
        ForecastPoint(
            ds         = pd.Timestamp(ts[i]).isoformat(),
            yhat       = round(float(yhat[i]),       2),
            yhat_lower = round(float(yhat_lower[i]), 2),
            yhat_upper = round(float(yhat_upper[i]), 2),
            anomaly    = bool(anomaly[i])
        )
        for i in range(len(ts))
    ]
    return points, round(mae, 2)


# ── LSTM PREDICT ──────────────────────────────────────────
def _predict_lstm(df: pd.DataFrame):
    lstm, scaler_X, scaler_y, device = _load_lstm()

    split   = int(len(df) * 0.8)
    test_df = df[split:].reset_index(drop=True)

    if len(test_df) <= SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {SEQ_LEN+1} rows, got {len(test_df)}"
        )

    test_X = scaler_X.transform(test_df[FEATURE_COLS].values)
    test_y = scaler_y.transform(test_df[["y"]].values).ravel()

    X_test, y_test = make_sequences(test_X, test_y, SEQ_LEN)
    X_tensor       = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_scaled = lstm(X_tensor).cpu().numpy()  # (N, 1)

    pred_kWh   = np.clip(scaler_y.inverse_transform(pred_scaled).ravel(), 0, None)
    actual_kWh = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    ts         = test_df["ds"].values[SEQ_LEN:]

    n          = min(len(pred_kWh), len(actual_kWh), len(ts))
    pred_kWh   = pred_kWh[:n]
    actual_kWh = actual_kWh[:n]
    ts         = ts[:n]

    errors    = np.abs(actual_kWh - pred_kWh)
    roll_mean = pd.Series(errors).rolling(24, min_periods=1).mean().fillna(0).values
    roll_std  = pd.Series(errors).rolling(24, min_periods=1).std().fillna(1).values
    threshold = np.maximum(roll_mean + 3 * roll_std, actual_kWh * 0.20)
    anomaly   = errors > threshold

    mae    = float(np.mean(errors))
    points = [
        ForecastPoint(
            ds      = pd.Timestamp(ts[i]).isoformat(),
            yhat    = round(float(pred_kWh[i]), 2),
            anomaly = bool(anomaly[i])
        )
        for i in range(n)
    ]
    return points, round(mae, 2)