import pandas as pd
import numpy as np
import mlflow
import pickle
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA      = "data/processed/steel_features.csv"  # use rich features
OUT_MODEL = "models/prophet/prophet_model.pkl"

REGRESSORS = [
    "is_weekend", "is_night", "is_peak", "is_morning", "is_afternoon",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_6", "lag_24", "lag_48", "lag_168",
    "roll_mean_4", "roll_mean_24", "roll_std_24",
    "delta_1", "delta_24",
    "load_light", "load_medium", "load_maximum"
]

def train_prophet():
    df = pd.read_csv(DATA, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # floor and cap for logistic growth
    df["floor"] = 5.0
    df["cap"]   = 600.0

    # ── Train / Test split ────────────────────────────────
    split = int(len(df) * 0.8)
    train = df[:split].copy()
    test  = df[split:].copy()
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")

    mlflow.set_experiment("demand-forecast-prophet-v3")

    with mlflow.start_run():
        # ── Build Prophet ─────────────────────────────────
        model = Prophet(
            yearly_seasonality=15,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=15.0,
            interval_width=0.95,
            growth="logistic",
            seasonality_mode="multiplicative"
        )

        # add all regressors
        for reg in REGRESSORS:
            model.add_regressor(reg)

        model.fit(train)

        # ── Predict on test ───────────────────────────────
        future = test[["ds", "floor", "cap"] + REGRESSORS].copy()
        forecast = model.predict(future)

        pred   = np.clip(forecast["yhat"].values, 5.0, 600.0)
        actual = test["y"].values

        # ── Metrics ───────────────────────────────────────
        mae     = mean_absolute_error(actual, pred)
        rmse    = np.sqrt(mean_squared_error(actual, pred))
        mae_pct = mae / actual.mean() * 100
        mask    = actual > 10.0
        mape    = np.mean(np.abs((actual[mask]-pred[mask])/actual[mask]))*100

        print(f"\n📊 Prophet v3 Results:")
        print(f"   MAE   : {mae:.2f} kWh  ({mae_pct:.1f}% of mean demand)")
        print(f"   RMSE  : {rmse:.2f} kWh")
        print(f"   MAPE  : {mape:.2f}%")

        if mae_pct < 10:
            print("   Grade : ✅ EXCELLENT")
        elif mae_pct < 20:
            print("   Grade : ✅ GOOD")
        elif mae_pct < 50:
            print("   Grade : ⚠️  ACCEPTABLE")
        else:
            print("   Grade : ❌ BAD")

        mlflow.log_params({
            "version": "v3",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 15.0,
            "n_regressors": len(REGRESSORS),
            "growth": "logistic"
        })
        mlflow.log_metrics({
            "mae": mae, "rmse": rmse,
            "mape": mape, "mae_pct": mae_pct
        })

        # ── Save ──────────────────────────────────────────
        os.makedirs("models/prophet", exist_ok=True)
        with open(OUT_MODEL, "wb") as f:
            pickle.dump({
                "model": model,
                "regressors": REGRESSORS
            }, f)
        print(f"\n✅ Model saved → {OUT_MODEL}")

    return model

if __name__ == "__main__":
    train_prophet()