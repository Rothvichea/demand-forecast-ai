import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from models.lstm.model import DemandLSTM
from models.lstm.train import (
    DATA, OUT_DIR, SEQ_LEN, FEATURE_COLS, TARGET_COL,
    HIDDEN, LAYERS, DROPOUT, make_sequences
)

PLOT_OUT = "models/lstm/forecast_plot.png"


def evaluate():
    df = pd.read_csv(DATA, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    split = int(len(df) * 0.8)
    test_df = df[split:]

    with open(f"{OUT_DIR}/scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(f"{OUT_DIR}/scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    test_X = scaler_X.transform(test_df[FEATURE_COLS].values)
    test_y = scaler_y.transform(test_df[[TARGET_COL]].values).ravel()

    X_test, y_test = make_sequences(test_X, test_y, SEQ_LEN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DemandLSTM(input_size=len(FEATURE_COLS),
                        hidden_size=HIDDEN, num_layers=LAYERS,
                        dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_lstm.pt", map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()

    pred_kWh   = np.clip(scaler_y.inverse_transform(pred_scaled).ravel(), 0, None)
    actual_kWh = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    ts = test_df["ds"].values[SEQ_LEN:]

    mae   = np.mean(np.abs(actual_kWh - pred_kWh))
    smape = np.mean(2 * np.abs(actual_kWh - pred_kWh) /
                    (np.abs(actual_kWh) + np.abs(pred_kWh))) * 100

    errors   = np.abs(actual_kWh - pred_kWh)
    roll_std = pd.Series(errors).rolling(24, min_periods=1).std().values
    anomaly  = errors > 2 * roll_std

    n_anom = anomaly.sum()
    print(f"📊 LSTM Evaluation on {len(actual_kWh)} test hours")
    print(f"   MAE   : {mae:.2f} kWh")
    print(f"   SMAPE : {smape:.2f}%")
    print(f"   Anomalies detected: {n_anom} ({n_anom/len(actual_kWh)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts, actual_kWh, label="Actual",   color="#39D353", lw=1)
    ax.plot(ts, pred_kWh,   label="Forecast", color="#388BFD", lw=1)
    ax.scatter(ts[anomaly], actual_kWh[anomaly],
               color="#F78166", s=20, zorder=5, label="Anomaly")
    ax.set_title("LSTM Forecast — Steel Energy Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    ax.legend()
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(PLOT_OUT, dpi=120)
    print(f"\n✅ Plot saved to {PLOT_OUT}")


if __name__ == "__main__":
    evaluate()
