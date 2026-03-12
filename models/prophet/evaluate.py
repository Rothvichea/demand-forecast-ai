import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

DATA  = "data/processed/steel_clean.csv"
MODEL = "models/prophet/prophet_model.pkl"
REGRESSORS = ["load_light", "load_medium", "load_maximum"]

def evaluate():
    df = pd.read_csv(DATA, parse_dates=["ds"])
    df["load_light"]   = (df["load_type"] == "Light_Load").astype(float)
    df["load_medium"]  = (df["load_type"] == "Medium_Load").astype(float)
    df["load_maximum"] = (df["load_type"] == "Maximum_Load").astype(float)

    split = int(len(df) * 0.8)
    test  = df[split:][["ds", "y"] + REGRESSORS]

    with open(MODEL, "rb") as f:
        model = pickle.load(f)

    future = model.make_future_dataframe(periods=len(test), freq="h")
    future = future.merge(df[["ds"] + REGRESSORS], on="ds", how="left")
    for reg in REGRESSORS:
        future[reg] = future[reg].fillna(0.0)
    future.loc[future[REGRESSORS].sum(axis=1) == 0, "load_light"] = 1.0

    forecast = model.predict(future)
    pred = forecast.tail(len(test))

    result = test.copy().reset_index(drop=True)
    result["yhat"]       = np.clip(pred["yhat"].values, 0, None)
    result["yhat_lower"] = np.clip(pred["yhat_lower"].values, 0, None)
    result["yhat_upper"] = pred["yhat_upper"].values

    result["anomaly"] = (
        (result["y"] < result["yhat_lower"]) |
        (result["y"] > result["yhat_upper"])
    )

    mae     = np.mean(np.abs(result["y"] - result["yhat"]))
    smape   = np.mean(2 * np.abs(result["y"] - result["yhat"]) /
                      (np.abs(result["y"]) + np.abs(result["yhat"]))) * 100
    n_anom  = result["anomaly"].sum()

    print(f"📊 Evaluation on {len(result)} test hours")
    print(f"   MAE   : {mae:.2f} kWh")
    print(f"   SMAPE : {smape:.2f}%")
    print(f"   Anomalies detected: {n_anom} ({n_anom/len(result)*100:.1f}%)")
    print(f"\n🔍 Sample anomalies:")
    print(result[result["anomaly"]].head(5)[["ds","y","yhat","yhat_lower","yhat_upper"]])

    # ── Plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result["ds"], result["y"],    label="Actual",   color="#39D353", lw=1)
    ax.plot(result["ds"], result["yhat"], label="Forecast", color="#388BFD", lw=1)
    ax.fill_between(result["ds"], result["yhat_lower"], result["yhat_upper"],
                    alpha=0.2, color="#388BFD", label="95% CI")
    ax.scatter(result[result["anomaly"]]["ds"],
               result[result["anomaly"]]["y"],
               color="#F78166", s=20, zorder=5, label="Anomaly")
    ax.set_title("Prophet Forecast — Steel Energy Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("kWh")
    ax.legend()
    plt.tight_layout()

    os.makedirs("models/prophet", exist_ok=True)
    plt.savefig("models/prophet/forecast_plot.png", dpi=120)
    print(f"\n✅ Plot saved to models/prophet/forecast_plot.png")

if __name__ == "__main__":
    evaluate()
