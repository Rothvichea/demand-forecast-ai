"""
CNN-LSTM training script — trains and benchmarks against pure LSTM.
Run:  python -m models.lstm.train_cnn_lstm
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import matplotlib.pyplot as plt

from models.lstm.model import CNNLSTMDemand, DemandLSTM
from models.lstm.train import (
    DATA, OUT_DIR, SEQ_LEN, BATCH_SIZE, EPOCHS, LR,
    FEATURE_COLS, TARGET_COL, DemandDataset, make_sequences
)
from sklearn.preprocessing import StandardScaler

CNN_MODEL_PATH = f"{OUT_DIR}/best_cnn_lstm.pt"
COMPARE_PLOT   = f"{OUT_DIR}/model_comparison.png"

HIDDEN      = 128
CNN_CHAN     = 64
LAYERS      = 2
DROPOUT     = 0.3
PATIENCE    = 15


def _train_one(model, train_loader, test_loader, device, model_name):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val  = float("inf")
    patience_counter = 0
    save_path = CNN_MODEL_PATH if "CNN" in model_name else f"{OUT_DIR}/best_lstm.pt"

    print(f"\n── Training {model_name} ──────────────────────────")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch} | best val={best_val:.4f}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def _evaluate(model, X_test, y_test, scaler_y, device):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_scaled = model(X_t).cpu().numpy()
    pred   = np.clip(scaler_y.inverse_transform(pred_scaled).ravel(), 0, None)
    actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mae    = np.mean(np.abs(actual - pred))
    rmse   = np.sqrt(np.mean((actual - pred) ** 2))
    smape  = np.mean(2 * np.abs(actual - pred) /
                     (np.abs(actual) + np.abs(pred))) * 100
    return pred, actual, mae, rmse, smape


def train_and_compare():
    # ── Data ─────────────────────────────────────────────
    df = pd.read_csv(DATA, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    split    = int(len(df) * 0.8)
    train_df = df[:split]
    test_df  = df[split:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    train_X  = scaler_X.fit_transform(train_df[FEATURE_COLS].values)
    test_X   = scaler_X.transform(test_df[FEATURE_COLS].values)
    train_y  = scaler_y.fit_transform(train_df[[TARGET_COL]].values).ravel()
    test_y   = scaler_y.transform(test_df[[TARGET_COL]].values).ravel()

    X_train, y_train = make_sequences(train_X, train_y, SEQ_LEN)
    X_test,  y_test  = make_sequences(test_X,  test_y,  SEQ_LEN)

    train_loader = DataLoader(DemandDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(DemandDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feat = len(FEATURE_COLS)
    ts     = test_df["ds"].values[SEQ_LEN:]

    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    mlflow.set_experiment("model-comparison")

    with mlflow.start_run(run_name="CNN-LSTM-vs-LSTM"):

        # ── Train CNN-LSTM ────────────────────────────────
        cnn_model = CNNLSTMDemand(input_size=n_feat, cnn_channels=CNN_CHAN,
                                   hidden_size=HIDDEN, num_layers=LAYERS,
                                   dropout=DROPOUT).to(device)
        cnn_params = sum(p.numel() for p in cnn_model.parameters())
        _train_one(cnn_model, train_loader, test_loader, device, "CNN-LSTM")
        cnn_pred, actual, cnn_mae, cnn_rmse, cnn_smape = \
            _evaluate(cnn_model, X_test, y_test, scaler_y, device)

        # ── Train pure LSTM for fair comparison ───────────
        lstm_model = DemandLSTM(input_size=n_feat, hidden_size=HIDDEN,
                                num_layers=LAYERS, dropout=DROPOUT).to(device)
        lstm_params = sum(p.numel() for p in lstm_model.parameters())
        _train_one(lstm_model, train_loader, test_loader, device, "LSTM")
        lstm_pred, _, lstm_mae, lstm_rmse, lstm_smape = \
            _evaluate(lstm_model, X_test, y_test, scaler_y, device)

        # ── Print comparison ──────────────────────────────
        print("\n" + "="*55)
        print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'SMAPE':>8}  {'Params':>8}")
        print("-"*55)
        print(f"{'LSTM':<15} {lstm_mae:>7.2f}  {lstm_rmse:>7.2f}  "
              f"{lstm_smape:>6.2f}%  {lstm_params:>7,}")
        print(f"{'CNN-LSTM':<15} {cnn_mae:>7.2f}  {cnn_rmse:>7.2f}  "
              f"{cnn_smape:>6.2f}%  {cnn_params:>7,}")
        delta = lstm_mae - cnn_mae
        pct   = delta / lstm_mae * 100
        icon  = "✅" if delta > 0 else "⚠️"
        print(f"\n{icon}  CNN-LSTM {'improved' if delta>0 else 'regressed'} "
              f"MAE by {abs(delta):.2f} kWh ({abs(pct):.1f}%)")
        print("="*55)

        # ── MLflow ────────────────────────────────────────
        mlflow.log_metrics({
            "lstm_mae": lstm_mae, "lstm_rmse": lstm_rmse,
            "cnn_lstm_mae": cnn_mae, "cnn_lstm_rmse": cnn_rmse,
            "mae_improvement_pct": pct
        })

        # ── Comparison plot ───────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        zoom = slice(0, min(500, len(actual)))   # show first 500 hours

        for ax, pred, title, color in [
            (axes[0], lstm_pred, f"LSTM  (MAE {lstm_mae:.2f} kWh)", "#F78166"),
            (axes[1], cnn_pred,  f"CNN-LSTM (MAE {cnn_mae:.2f} kWh)", "#39D353"),
        ]:
            ax.plot(ts[zoom], actual[zoom], label="Actual",
                    color="#C9D1D9", lw=1, alpha=0.8)
            ax.plot(ts[zoom], pred[zoom],   label=title,
                    color=color, lw=1.2)
            residuals = np.abs(actual[zoom] - pred[zoom])
            ax.fill_between(ts[zoom],
                            actual[zoom] - residuals,
                            actual[zoom] + residuals,
                            alpha=0.12, color=color)
            ax.set_ylabel("kWh")
            ax.legend(loc="upper right", fontsize=9)
            ax.set_facecolor("#0D1117")
            ax.tick_params(colors="#8B949E")
            for spine in ax.spines.values():
                spine.set_edgecolor("#21262D")
            ax.grid(color="#21262D", lw=0.5)

        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Model Comparison — Steel Energy Demand", color="#F0F6FC",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(COMPARE_PLOT, dpi=120, facecolor="#0D1117")
        print(f"\n✅ Comparison plot saved → {COMPARE_PLOT}")
        mlflow.log_artifact(COMPARE_PLOT)

        # save scalers (overwrite with latest)
        with open(f"{OUT_DIR}/scaler_X.pkl", "wb") as f: pickle.dump(scaler_X, f)
        with open(f"{OUT_DIR}/scaler_y.pkl", "wb") as f: pickle.dump(scaler_y, f)

    return cnn_mae, lstm_mae


if __name__ == "__main__":
    train_and_compare()
