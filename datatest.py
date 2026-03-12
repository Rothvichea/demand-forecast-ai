import argparse
from pathlib import Path
from typing import Optional
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

TIME_COL = "__time__"
VALUE_COL = "__value__"


def _parse_datetime_best(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        best = pd.to_datetime(series, errors="coerce")
    best_ratio = best.notna().mean()
    for dayfirst in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best = parsed
            best_ratio = ratio
    return best


def _find_datetime_column(df: pd.DataFrame) -> tuple[Optional[str], Optional[pd.Series]]:
    cols = list(df.columns)
    keyword_cols = [c for c in cols if any(k in c.lower() for k in ["date", "time", "timestamp", "ds"])]
    if keyword_cols:
        scored = keyword_cols
    else:
        scored = [
            c for c in cols
            if pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
        ]

    best_col = None
    best_parsed = None
    best_ratio = 0.0

    for col in scored:
        parsed = _parse_datetime_best(df[col])
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best_col = col
            best_parsed = parsed
            best_ratio = ratio

    if best_col is None or best_ratio < 0.6:
        return None, None
    return best_col, best_parsed


def _find_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
            continue
        as_num = pd.to_numeric(df[col], errors="coerce")
        if as_num.notna().mean() >= 0.8:
            numeric_cols.append(col)
    return numeric_cols


def _find_value_column(df: pd.DataFrame, numeric_cols: list[str], time_col: Optional[str]) -> str:
    candidates = [c for c in numeric_cols if c != time_col]
    if not candidates:
        raise ValueError("No numeric column found. Use --value-col to select one.")

    keywords = ["usage", "demand", "load", "kwh", "target", "value", "y"]
    preferred = [c for c in candidates if any(k in c.lower() for k in keywords)]
    if preferred:
        return preferred[0]
    return candidates[0]


def load_data(
    csv_path: Path,
    datetime_col: Optional[str],
    value_col: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    if datetime_col and datetime_col not in df.columns:
        raise ValueError(f"--datetime-col '{datetime_col}' not found in CSV columns.")
    if value_col and value_col not in df.columns:
        raise ValueError(f"--value-col '{value_col}' not found in CSV columns.")

    detected_time_col = datetime_col
    detected_time = None
    if detected_time_col is None:
        detected_time_col, detected_time = _find_datetime_column(df)
    elif detected_time_col is not None:
        detected_time = _parse_datetime_best(df[detected_time_col])

    numeric_cols = _find_numeric_columns(df)
    detected_value_col = value_col or _find_value_column(df, numeric_cols, detected_time_col)

    work = df.copy()
    work[VALUE_COL] = pd.to_numeric(work[detected_value_col], errors="coerce")

    if detected_time_col is not None and detected_time is not None:
        work[TIME_COL] = detected_time
        work = work.sort_values(TIME_COL).reset_index(drop=True)

    meta = {
        "time_col": detected_time_col,
        "value_col": detected_value_col,
    }
    return work, meta


def run_checks(df: pd.DataFrame, meta: dict, head_rows: int) -> None:
    print("==== Data Checks ====")
    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nAuto-selected value column: {meta['value_col']}")
    print(f"Auto-selected time column: {meta['time_col']}")
    print(f"\nHead ({head_rows} rows):")
    print(df.head(head_rows).to_string(index=False))

    print("\nMissing values (top 10 columns):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    if meta["time_col"] is not None and TIME_COL in df.columns:
        print(f"\nDate range: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")
        print(f"Duplicate timestamps: {df.duplicated(subset=[TIME_COL]).sum()}")

    print(f"\nStats for '{meta['value_col']}':")
    print(df[VALUE_COL].describe().round(2))


def quick_insights(df: pd.DataFrame, meta: dict) -> None:
    clean = df.dropna(subset=[VALUE_COL]).copy()
    print("\n==== Quick Insights ====")

    if meta["time_col"] is not None and TIME_COL in clean.columns:
        clean = clean.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
        if clean.empty:
            return

        print("\nTop 5 peak points:")
        print(clean.nlargest(5, VALUE_COL)[[TIME_COL, VALUE_COL]].to_string(index=False))

        jumps = clean[VALUE_COL].diff().abs()
        jump_idx = jumps.nlargest(5).index
        print("\nLargest 5 step changes:")
        for idx in jump_idx:
            if idx <= 0 or pd.isna(jumps.iloc[idx]):
                continue
            prev_ts = clean.loc[idx - 1, TIME_COL]
            prev_val = clean.loc[idx - 1, VALUE_COL]
            now_ts = clean.loc[idx, TIME_COL]
            now_val = clean.loc[idx, VALUE_COL]
            delta = jumps.iloc[idx]
            print(f"{prev_ts} -> {now_ts}: {prev_val:.2f} -> {now_val:.2f} (delta={delta:.2f})")


def make_plots(
    df: pd.DataFrame, meta: dict, output_path: Optional[Path] = None, show: bool = True
) -> None:
    clean = df.dropna(subset=[VALUE_COL]).copy()
    has_time = meta["time_col"] is not None and TIME_COL in clean.columns

    if has_time:
        clean = clean.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)

    # Plot: time series only
    if has_time:
        ax.plot(clean[TIME_COL], clean[VALUE_COL], color="tab:blue", linewidth=0.8)
        ax.set_title(
            f"{meta['value_col']} Over {meta['time_col']} (move mouse to inspect points)"
        )
        ax.set_xlabel(meta["time_col"])

        # Hover tooltip: shows nearest timestamp + value when moving mouse.
        x_times = pd.to_datetime(clean[TIME_COL])
        y_vals = clean[VALUE_COL].to_numpy()
        x_nums = mdates.date2num(x_times.to_numpy(dtype="datetime64[ns]"))

        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.5", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "0.4"},
        )
        annot.set_visible(False)

        def _nearest_idx(x: float) -> int:
            idx = np.searchsorted(x_nums, x)
            idx = min(max(idx, 0), len(x_nums) - 1)
            if idx > 0 and abs(x - x_nums[idx - 1]) < abs(x_nums[idx] - x):
                idx -= 1
            return idx

        def _on_move(event) -> None:
            if event.inaxes != ax or event.xdata is None:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                return

            idx = _nearest_idx(event.xdata)
            x_point = x_nums[idx]
            y_point = y_vals[idx]
            ts_text = pd.Timestamp(x_times.iloc[idx]).strftime("%Y-%m-%d %H:%M:%S")
            annot.xy = (x_point, y_point)
            annot.set_text(f"{meta['time_col']}: {ts_text}\n{meta['value_col']}: {y_point:.2f}")
            annot.set_visible(True)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_move)
    else:
        ax.plot(clean.index, clean[VALUE_COL], color="tab:blue", linewidth=0.8)
        ax.set_title(f"{meta['value_col']} Over Data Sequence")
        ax.set_xlabel("Row index")
    ax.set_ylabel(meta["value_col"])

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-inspect and plot any CSV (auto-detects time/value/category columns)."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to input CSV file")
    parser.add_argument("--datetime-col", type=str, default=None, help="Optional datetime column")
    parser.add_argument("--value-col", type=str, default=None, help="Optional numeric column to plot")
    parser.add_argument("--head", type=int, default=5, help="How many head rows to print")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save output plot image",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plot display",
    )
    args = parser.parse_args()

    df, meta = load_data(
        args.csv,
        datetime_col=args.datetime_col,
        value_col=args.value_col,
    )
    run_checks(df, meta, head_rows=args.head)
    quick_insights(df, meta)
    make_plots(df, meta, output_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
