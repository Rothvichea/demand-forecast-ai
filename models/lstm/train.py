import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import mlflow
import pickle

from models.lstm.model import DemandLSTM

DATA    = "data/processed/steel_features.csv"
OUT_DIR = "models/lstm"

SEQ_LEN    = 48      # use last 24 hours to predict next hour
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 1e-3
HIDDEN     = 128
LAYERS     = 2
DROPOUT    = 0.3

FEATURE_COLS = [
    # cyclical time encoding
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    # time flags
    "is_weekend", "is_night", "is_peak", "is_morning", "is_afternoon",
    # lags
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "lag_24", "lag_48", "lag_168",
    # rolling stats
    "roll_mean_4", "roll_mean_24", "roll_mean_168",
    "roll_std_4",  "roll_std_24",
    "roll_max_24", "roll_min_24",
    # rate of change
    "delta_1", "delta_24",
    # load type
    "load_light", "load_medium", "load_maximum"
]
TARGET_COL = "y"


class DemandDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def train_lstm():
    df = pd.read_csv(DATA, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # ── Train / Test split ─────────────────────────────────
    split = int(len(df) * 0.8)
    train_df = df[:split]
    test_df  = df[split:]

    # ── Scale features ────────────────────────────────────
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train_X = scaler_X.fit_transform(train_df[FEATURE_COLS].values)
    test_X  = scaler_X.transform(test_df[FEATURE_COLS].values)

    train_y = scaler_y.fit_transform(train_df[[TARGET_COL]].values).ravel()
    test_y  = scaler_y.transform(test_df[[TARGET_COL]].values).ravel()

    # ── Build sequences ───────────────────────────────────
    X_train, y_train = make_sequences(train_X, train_y, SEQ_LEN)
    X_test,  y_test  = make_sequences(test_X,  test_y,  SEQ_LEN)

    print(f"Train sequences: {X_train.shape} | Test sequences: {X_test.shape}")

    train_loader = DataLoader(DemandDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(DemandDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DemandLSTM(input_size=len(FEATURE_COLS),
                        hidden_size=HIDDEN,
                        num_layers=LAYERS,
                        dropout=DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # ── MLflow ────────────────────────────────────────────
    mlflow.set_experiment("demand-forecast-lstm")
    with mlflow.start_run():
        mlflow.log_params({
            "seq_len": SEQ_LEN, "hidden": HIDDEN, "layers": LAYERS,
            "dropout": DROPOUT, "lr": LR, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "features": FEATURE_COLS
        })

        best_val_loss = float("inf")
        patience_counter = 0
        PATIENCE = 70          # stop if no improvement for 10 epochs
        for epoch in range(1, EPOCHS + 1):
            # train
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

            # validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item() * len(xb)
            val_loss /= len(test_loader.dataset)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{OUT_DIR}/best_lstm.pt")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n⏹️  Early stopping at epoch {epoch} — best val_loss: {best_val_loss:.4f}")
                    break
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                      f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss},
                               step=epoch)

        # ── Final evaluation in original scale ────────────
        model.load_state_dict(torch.load(f"{OUT_DIR}/best_lstm.pt",
                                         map_location=device))
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
                actuals.append(yb.numpy())

        pred_scaled   = np.vstack(preds)
        actual_scaled = np.vstack(actuals)

        pred_kWh   = scaler_y.inverse_transform(pred_scaled).ravel()
        actual_kWh = scaler_y.inverse_transform(actual_scaled).ravel()

        pred_kWh = np.clip(pred_kWh, 0, None)

        mae   = np.mean(np.abs(actual_kWh - pred_kWh))
        rmse  = np.sqrt(np.mean((actual_kWh - pred_kWh) ** 2))
        smape = np.mean(2 * np.abs(actual_kWh - pred_kWh) /
                        (np.abs(actual_kWh) + np.abs(pred_kWh))) * 100

        print(f"\n📊 LSTM Results:")
        print(f"   MAE   : {mae:.2f} kWh")
        print(f"   RMSE  : {rmse:.2f} kWh")
        print(f"   SMAPE : {smape:.2f}%")

        mlflow.log_metrics({"test_mae": mae, "test_rmse": rmse, "test_smape": smape})

        # save scaler for inference
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(f"{OUT_DIR}/scaler_X.pkl", "wb") as f:
            pickle.dump(scaler_X, f)
        with open(f"{OUT_DIR}/scaler_y.pkl", "wb") as f:
            pickle.dump(scaler_y, f)

        mlflow.log_artifact(f"{OUT_DIR}/best_lstm.pt")
        print(f"\n✅ Model saved to {OUT_DIR}/best_lstm.pt")

    return model, pred_kWh, actual_kWh


if __name__ == "__main__":
    train_lstm()
