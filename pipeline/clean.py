import pandas as pd
import os

RAW = "data/raw/Steel_industry_data.csv"
OUT = "data/processed/steel_clean.csv"

def load_and_clean():
    df = pd.read_csv(RAW, parse_dates=["date"], dayfirst=True)

    # rename for consistency
    df = df.rename(columns={
        "date":        "ds",
        "Usage_kWh":   "y",
        "Load_Type":   "load_type",
        "WeekStatus":  "week_status",
        "Day_of_week": "day_of_week"
    })

    # sort by time
    df = df.sort_values("ds").reset_index(drop=True)

    # remove duplicates
    df = df.drop_duplicates(subset="ds")

    # drop nulls
    df = df.dropna(subset=["ds", "y"])

    # remove zero/negative demand (sensor errors)
    df = df[df["y"] > 0]

    # ── Resample to hourly — Prophet works much better ────
    df = df.set_index("ds")
    df = df.resample("1h").agg({
        "y":           "sum",   # total kWh per hour
        "load_type":   "last",  # dominant load type
        "week_status": "last",
        "day_of_week": "last"
    }).reset_index()
    df = df.dropna(subset=["y"])
    df = df[df["y"] > 0]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"✅ Cleaned: {len(df)} rows saved to {OUT}")
    print(f"   Date range: {df['ds'].min()} → {df['ds'].max()}")
    print(f"   Demand range: {df['y'].min():.2f} → {df['y'].max():.2f} kWh")
    print(f"   Load types: {df['load_type'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    load_and_clean()