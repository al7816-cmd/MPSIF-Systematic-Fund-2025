#!/usr/bin/env python3
"""
Backtest Engine
---------------
- Loads Compustat parquet price file (must contain daily returns `ret_daily`)
- Loads CSV weights (permno columns, first column `date`)
- Computes daily portfolio returns
- Saves:
    * summary_stats.csv
    * daily_portfolio_returns.csv
    * ff5_regression.txt (alpha vs Fama–French 5-factor model)
    * performance_chart.png (strategy vs market, rolling Sharpe, drawdown)
- Uses Fama–French daily 5-factor data in: data/FFM_Daily_Data.csv
"""

import os
# Limit BLAS threads to avoid WRDS limits
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
parser = argparse.ArgumentParser(description="Run backtest engine.")
parser.add_argument(
    "--prices",
    required=True,
    help="Parquet file containing daily Compustat data with `ret_daily`."
)
parser.add_argument(
    "--weights",
    required=True,
    help="CSV file containing weight time series (first column 'date')."
)
parser.add_argument(
    "--outdir",
    default="backtests",
    help="Base output folder for backtest results."
)
args = parser.parse_args()

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
BASE_DIR = Path(".").resolve()
PRICES_PATH = BASE_DIR / args.prices
WEIGHTS_PATH = BASE_DIR / args.weights
FF_PATH = BASE_DIR / "data" / "FFM_Daily_Data.csv"

if not PRICES_PATH.exists():
    raise FileNotFoundError(f"Price file not found: {PRICES_PATH}")

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")

if not FF_PATH.exists():
    raise FileNotFoundError(f"Fama–French file not found: {FF_PATH}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_FOLDER = (BASE_DIR / args.outdir / f"backtest_{timestamp}").resolve()
OUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"📁 Output folder: {OUT_FOLDER}")

# ----------------------------------------------------------
# Load & reshape prices
# ----------------------------------------------------------
print("Loading Compustat prices...")
prices_raw = pd.read_parquet(PRICES_PATH)

required_cols = {"permno", "date", "ret_daily"}
missing = required_cols - set(prices_raw.columns)
if missing:
    raise ValueError(f"Price file missing required columns: {missing}")

# Ensure proper dtypes
prices_raw = prices_raw.copy()
prices_raw["date"] = pd.to_datetime(prices_raw["date"])
prices_raw["permno_int"] = prices_raw["permno"].astype("Int64")

# Pivot to [date x permno]
prices_pivot = prices_raw.pivot(
    index="date",
    columns="permno_int",
    values="ret_daily"
).sort_index()

# Convert to plain float with NaN for missing
prices_pivot = prices_pivot.astype(float)

# ----------------------------------------------------------
# Load & clean weights
# ----------------------------------------------------------
print("Loading weights...")
weights = pd.read_csv(WEIGHTS_PATH)

if "date" not in weights.columns:
    raise ValueError("Weights CSV must have a 'date' column as the first column.")

weights["date"] = pd.to_datetime(weights["date"])

# Identify permno columns (everything except 'date')
perm_cols_raw = [c for c in weights.columns if c != "date"]

col_map = {}
for c in perm_cols_raw:
    try:
        p = int(float(c))
        col_map[c] = p
    except Exception:
        # Ignore non-numeric columns
        pass

if not col_map:
    raise ValueError("No valid permno columns found in weights file.")

# Rename columns to integer permnos
weights = weights.rename(columns=col_map)

# Set index to date
weights = weights.set_index("date").sort_index()

# ----------------------------------------------------------
# Align dates & permnos
# ----------------------------------------------------------
common_dates = prices_pivot.index.intersection(weights.index)
if len(common_dates) == 0:
    raise ValueError("No overlapping dates between prices and weights.")

print(f"Common dates: {common_dates.min().date()} → {common_dates.max().date()}")
print(f"{len(common_dates)} days")

prices_aligned = prices_pivot.loc[common_dates]

# Only keep permnos present in both prices and weights
common_permnos = sorted(set(prices_aligned.columns).intersection(set(weights.columns)))
if not common_permnos:
    raise ValueError("No overlapping permnos between prices and weights.")

prices_aligned = prices_aligned[common_permnos]
weights_aligned = weights[common_permnos].loc[common_dates]

# Replace NaNs: returns NaN → 0 contribution; weights NaN → 0
returns_np = prices_aligned.to_numpy(dtype="float64")
returns_np = np.where(np.isnan(returns_np), 0.0, returns_np)

weights_np = weights_aligned.fillna(0.0).to_numpy(dtype="float64")

# ----------------------------------------------------------
# Compute portfolio returns
# ----------------------------------------------------------
print("Computing portfolio returns...")

if returns_np.shape != weights_np.shape:
    raise ValueError(
        f"Shape mismatch: returns {returns_np.shape}, weights {weights_np.shape}"
    )

portfolio_ret = np.sum(weights_np * returns_np, axis=1)  # assumes weights are in weights file

# Store as Series for convenience
portfolio_ret = pd.Series(portfolio_ret, index=common_dates, name="portfolio_ret")

# Drop any non-finite values just in case
valid_mask = np.isfinite(portfolio_ret.values)
portfolio_ret = portfolio_ret[valid_mask]

if portfolio_ret.empty:
    raise ValueError("No valid portfolio returns (all NaN/inf).")

# ----------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------
ann = 252
ret_vals = portfolio_ret.values

cum_return = np.prod(1.0 + ret_vals) - 1.0
ann_return = (1.0 + cum_return) ** (ann / len(ret_vals)) - 1.0
ann_vol = np.std(ret_vals) * np.sqrt(ann)
sharpe = np.mean(ret_vals) / np.std(ret_vals) if ann_vol > 0 else np.nan

summary = pd.DataFrame(
    {
        "cumulative_return": [cum_return],
        "annualized_return": [ann_return],
        "annualized_vol": [ann_vol],
        "sharpe_ratio": [sharpe],
    }
)
summary.to_csv(OUT_FOLDER / "summary_stats.csv", index=False)

# Save daily returns
daily_df = portfolio_ret.reset_index()
daily_df.columns = ["date", "portfolio_ret"]
daily_df["daily_return"] = daily_df["portfolio_ret"]
daily_df.to_csv(OUT_FOLDER / "daily_portfolio_returns.csv", index=False)

# ----------------------------------------------------------
# Load Fama–French 5-factor data (daily)
# ----------------------------------------------------------
print("Loading Fama–French factors from FFM_Daily_Data.csv...")

ff_raw = pd.read_csv(FF_PATH, skiprows=4)
# First column holds the YYYYMMDD dates plus some junk lines at bottom
date_col_name = ff_raw.columns[0]
date_num = pd.to_numeric(ff_raw[date_col_name], errors="coerce")

# Keep only rows with numeric dates
ff5 = ff_raw[date_num.notna()].copy()
ff5.rename(columns={date_col_name: "date"}, inplace=True)

# Convert date and factors
ff5["date"] = pd.to_datetime(ff5["date"].astype(int).astype(str), format="%Y%m%d")

factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
for col in factor_cols:
    if col not in ff5.columns:
        raise ValueError(f"Expected factor column '{col}' not found in FFM_Daily_Data.csv")

ff5[factor_cols] = ff5[factor_cols].astype(float) / 100.0

# Rename "Mkt-RF" so we can use it easily in Python
ff5 = ff5.rename(columns={"Mkt-RF": "Mkt_RF"})

# ----------------------------------------------------------
# Merge with portfolio returns & run FF5 regression
# ----------------------------------------------------------
print("Running Fama–French 5-factor regression...")

merged = daily_df.merge(ff5, on="date", how="inner")
if merged.empty:
    print("⚠️ No overlap between portfolio returns and FF5 data; skipping regression.")
else:
    # Excess return
    merged["excess_ret"] = merged["portfolio_ret"] - merged["RF"]

    # Drop rows with NaN in any regression variable
    reg_cols = ["excess_ret", "Mkt_RF", "SMB", "HML", "RMW", "CMA"]
    merged_reg = merged.dropna(subset=reg_cols).copy()

    if merged_reg.empty:
        print("⚠️ No valid rows for regression after dropping NaNs; skipping regression.")
    else:
        X = merged_reg[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]].astype(float)
        X = sm.add_constant(X)
        y = merged_reg["excess_ret"].astype(float)

        model = sm.OLS(y, X).fit()
        with open(OUT_FOLDER / "ff5_regression.txt", "w") as f:
            f.write(model.summary().as_text())

# ----------------------------------------------------------
# Enhanced visualization (strategy vs market, Sharpe, drawdown)
# ----------------------------------------------------------
print("Plotting performance chart...")

if merged.empty:
    # Fallback: only plot cumulative portfolio returns
    cumret = (1.0 + portfolio_ret).cumprod()
    plt.figure(figsize=(10, 5))
    plt.plot(cumret.index, cumret.values, label="Portfolio")
    plt.title("Cumulative Return")
    plt.ylabel("Cumulative Return (×)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FOLDER / "performance_chart.png", dpi=200)
    plt.close()
else:
    # Use merged (only dates with factor data) for plots
    merged_plot = merged.dropna(subset=["portfolio_ret", "Mkt_RF", "RF"]).copy()
    merged_plot = merged_plot.sort_values("date")

    strategy_rets = merged_plot.set_index("date")["portfolio_ret"].astype(float)

    # Market total return = Mkt_RF (excess) + RF
    mkt_rets = (merged_plot["Mkt_RF"] + merged_plot["RF"]).astype(float)
    benchmark_rets = pd.Series(mkt_rets.values, index=merged_plot["date"])

    strategy_cum = (1.0 + strategy_rets).cumprod()
    benchmark_cum = (1.0 + benchmark_rets).cumprod()

    # Align for rolling metrics
    common_idx = strategy_cum.index.intersection(benchmark_cum.index)
    strategy_cum = strategy_cum.loc[common_idx]
    benchmark_cum = benchmark_cum.loc[common_idx]
    strategy_rets = strategy_rets.loc[common_idx]
    benchmark_rets = benchmark_rets.loc[common_idx]

    # Rolling stats
    rolling_window = 252
    rolling_mean = strategy_rets.rolling(rolling_window).mean() * 252.0
    rolling_std = strategy_rets.rolling(rolling_window).std() * np.sqrt(252.0)
    rolling_sharpe = rolling_mean / rolling_std

    cum = (1.0 + strategy_rets).cumprod()
    cum_max = cum.cummax()
    rolling_drawdown = cum / cum_max - 1.0

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0, 1.0]}
    )

    # 1️⃣ Cumulative performance
    axes[0].plot(strategy_cum.index, strategy_cum.values,
                 label="Strategy", linewidth=2.0)
    axes[0].plot(benchmark_cum.index, benchmark_cum.values,
                 label="Market (Mkt_RF + RF)", linewidth=2.0)
    axes[0].set_title(
        f"Strategy vs Market Performance "
        f"({common_idx.min().date()} – {common_idx.max().date()})",
        fontsize=14, fontweight="bold"
    )
    axes[0].set_ylabel("Cumulative Return (×)", fontsize=11)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")

    # Annotate ending performance
    try:
        axes[0].text(
            strategy_cum.index[-1],
            strategy_cum.iloc[-1] * 0.9,
            f"Strategy: {strategy_cum.iloc[-1] - 1.0:.1%}",
            fontsize=9
        )
        axes[0].text(
            benchmark_cum.index[-1],
            benchmark_cum.iloc[-1] * 0.8,
            f"Market: {benchmark_cum.iloc[-1] - 1.0:.1%}",
            fontsize=9
        )
    except Exception:
        pass

    # 2️⃣ Rolling Sharpe
    axes[1].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
    axes[1].axhline(0.0, color="gray", lw=1.0, linestyle="--")
    axes[1].set_ylabel("Rolling Sharpe (252d)", fontsize=11)
    axes[1].set_title("Rolling Sharpe Ratio", fontsize=12)

    # 3️⃣ Drawdown
    axes[2].fill_between(
        rolling_drawdown.index,
        rolling_drawdown.values,
        0.0,
        alpha=0.4
    )
    axes[2].set_ylabel("Drawdown", fontsize=11)
    axes[2].set_xlabel("Date", fontsize=11)
    axes[2].set_title("Rolling Drawdown", fontsize=12)

    plt.tight_layout()
    plt.savefig(OUT_FOLDER / "performance_chart.png", dpi=300)
    plt.close()

print("✅ Backtest complete.")
print(f"📁 Results saved to: {OUT_FOLDER}")
