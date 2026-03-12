import pandas as pd
import numpy as np
import os

IN  = "data/processed/steel_clean.csv"
OUT = "data/processed/steel_features.csv"

def build_features():
    df = pd.read_csv(IN, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # ── Calendar features ──────────────────────────────────
    df["hour"]        = df["ds"].dt.hour
    df["minute"]      = df["ds"].dt.minute
    df["day"]         = df["ds"].dt.day
    df["month"]       = df["ds"].dt.month
    df["quarter"]     = df["ds"].dt.quarter
    df["dayofweek"]   = df["ds"].dt.dayofweek
    df["is_weekend"]  = (df["dayofweek"] >= 5).astype(int)
    df["weekofyear"]  = df["ds"].dt.isocalendar().week.astype(int)

    # ── Time-of-day flags ──────────────────────────────────
    df["is_night"]    = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_peak"]     = ((df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)
    df["is_morning"]  = ((df["hour"] >= 6) & (df["hour"] < 12)).astype(int)
    df["is_afternoon"]= ((df["hour"] >= 12) & (df["hour"] < 18)).astype(int)

    # ── Cyclical encoding (hour as sin/cos) ────────────────
    # This helps LSTM understand that hour 23 is close to hour 0
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    # ── Lag features ───────────────────────────────────────
    df["lag_1"]       = df["y"].shift(1)
    df["lag_2"]       = df["y"].shift(2)
    df["lag_3"]       = df["y"].shift(3)
    df["lag_6"]       = df["y"].shift(6)
    df["lag_12"]      = df["y"].shift(12)
    df["lag_24"]      = df["y"].shift(24)
    df["lag_48"]      = df["y"].shift(48)
    df["lag_168"]     = df["y"].shift(168)  # 1 week ago

    # ── Rolling statistics ─────────────────────────────────
    df["roll_mean_4"]   = df["y"].rolling(4).mean()
    df["roll_mean_24"]  = df["y"].rolling(24).mean()
    df["roll_mean_168"] = df["y"].rolling(168).mean()
    df["roll_std_4"]    = df["y"].rolling(4).std()
    df["roll_std_24"]   = df["y"].rolling(24).std()
    df["roll_max_24"]   = df["y"].rolling(24).max()
    df["roll_min_24"]   = df["y"].rolling(24).min()

    # ── Demand delta (rate of change) ──────────────────────
    df["delta_1"]     = df["y"].diff(1)
    df["delta_24"]    = df["y"].diff(24)

    # ── Load type encoding ─────────────────────────────────
    df["load_light"]   = (df["load_type"] == "Light_Load").astype(int)
    df["load_medium"]  = (df["load_type"] == "Medium_Load").astype(int)
    df["load_maximum"] = (df["load_type"] == "Maximum_Load").astype(int)

    # ── Drop NaN from lags/rolling ─────────────────────────
    df = df.dropna().reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"✅ Features built: {len(df)} rows, {len(df.columns)} columns")
    print(f"   New feature count: {len(df.columns)}")
    return df

if __name__ == "__main__":
    build_features()