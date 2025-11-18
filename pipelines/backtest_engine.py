#!/usr/bin/env python3
"""
Backtest Engine (robust version)

- Loads Compustat parquet price file (with ret_daily)
- Loads CSV weights (permno columns as strings)
- Aligns dates and permnos, handles missing data robustly
- Computes daily portfolio returns
- Saves summary stats + daily returns CSV
- Loads Fama-French 5-factor data from: data/FFM_Daily_Data.csv
- Runs FF5 regression (excess return vs Mkt-RF, SMB, HML, RMW, CMA)
- Plots cumulative return curve
- Writes everything into a timestamped results folder
"""

import os
# Limit BLAS threads due to WRDS process limits
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
import statsmodels.api as sm

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--prices", required=True, help="Parquet file containing daily Compustat data with ret_daily")
parser.add_argument("--weights", required=True, help="CSV file containing weight time series (permno columns)")
parser.add_argument("--outdir", default="backtests", help="Output directory for results")
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
    raise FileNotFoundError(f"FF5 file not found: {FF_PATH}")

# ----------------------------------------------------------
# Output folder
# ----------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_FOLDER = BASE_DIR / args.outdir / f"backtest_{timestamp}"
OUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"📁 Output folder: {OUT_FOLDER}")

# ----------------------------------------------------------
# Load Compustat daily returns
# ----------------------------------------------------------
print("Loading Compustat prices...")
prices_raw = pd.read_parquet(PRICES_PATH)

# Expect columns: permno, date, ret_daily (plus some others)
# Keep only what's needed and drop rows with missing ret_daily
if "ret_daily" not in prices_raw.columns:
    raise ValueError("Price file must contain a 'ret_daily' column.")

price_cols_needed = ["permno", "date", "ret_daily"]
missing_cols = [c for c in price_cols_needed if c not in prices_raw.columns]
if missing_cols:
    raise ValueError(f"Price file missing required columns: {missing_cols}")

prices = prices_raw[price_cols_needed].copy()
prices["date"] = pd.to_datetime(prices["date"])
# Make permno integer for clean matching
prices["permno"] = prices["permno"].astype(int)

# Drop rows with NA ret_daily (no return info)
prices = prices[prices["ret_daily"].notna()].copy()

# Pivot: rows = date, columns = permno, values = ret_daily
prices_pivot = prices.pivot(index="date", columns="permno", values="ret_daily").sort_index()

# ----------------------------------------------------------
# Load weights
# ----------------------------------------------------------
print("Loading weights...")
weights = pd.read_csv(WEIGHTS_PATH)

if "date" not in weights.columns:
    raise ValueError("Weights file must have a 'date' column.")

weights["date"] = pd.to_datetime(weights["date"])
weights = weights.set_index("date").sort_index()

# Convert weight permno columns (currently strings) to int where possible
weight_cols = list(weights.columns)
new_cols = []
for c in weight_cols:
    # Try to interpret as permno (float or int in string form)
    try:
        new_cols.append(int(float(c)))
    except Exception:
        # non-permno column (shouldn't really happen beyond 'date', which is already index)
        new_cols.append(c)

weights.columns = new_cols

# ----------------------------------------------------------
# Align dates
# ----------------------------------------------------------
common_dates = prices_pivot.index.intersection(weights.index)
if common_dates.empty:
    raise ValueError("No overlapping dates between prices and weights.")

print(f"Common dates: {common_dates.min().date()} → {common_dates.max().date()}")
print(f"{len(common_dates)} days")

prices_pivot = prices_pivot.loc[common_dates]
weights = weights.loc[common_dates]

# ----------------------------------------------------------
# Align permno columns and compute portfolio returns
# ----------------------------------------------------------
print("Computing portfolio returns...")

# Common permnos between prices and weights
common_permnos = sorted(set(prices_pivot.columns) & set(weights.columns))
if not common_permnos:
    raise ValueError("No overlapping PERMNOs between prices and weights.")

# Subset both to the common permnos, same order
prices_pivot = prices_pivot[common_permnos]
weights = weights[common_permnos]

# Convert to numeric, coerce any junk to NaN, then fill NaN with 0 in weights
weights = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
# For returns: coerce to numeric, keep NaN where missing, then treat NaN returns as 0 (no move)
prices_pivot = prices_pivot.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Convert to numpy arrays
returns_np = prices_pivot.to_numpy(dtype=float)
weights_np = weights.to_numpy(dtype=float)

# Pointwise product and sum across assets → portfolio daily return
portfolio_ret = np.sum(weights_np * returns_np, axis=1)

# Put in a DataFrame
daily_df = pd.DataFrame(
    {
        "date": common_dates,
        "portfolio_ret": portfolio_ret.astype(float),
    }
)

# ----------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------
ann = 252  # trading days
ret_series = daily_df["portfolio_ret"].astype(float)

cum_return = (1.0 + ret_series).prod() - 1.0
ann_return = (1.0 + ret_series).prod() ** (ann / len(ret_series)) - 1.0
ann_vol = ret_series.std(ddof=1) * np.sqrt(ann)
sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

summary = pd.DataFrame(
    {
        "cumulative_return": [cum_return],
        "annualized_return": [ann_return],
        "annualized_vol": [ann_vol],
        "sharpe_ratio": [sharpe],
    }
)
summary.to_csv(OUT_FOLDER / "summary_stats.csv", index=False)

# ----------------------------------------------------------
# Load Fama-French 5-factor data from FFM_Daily_Data.csv
# ----------------------------------------------------------
print("Loading Fama-French factors...")

# From inspection: skip the 4 header rows, then we have:
# columns: 'Unnamed: 0', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'
ff_raw = pd.read_csv(FF_PATH, skiprows=4)

# Rename date column
if "Unnamed: 0" not in ff_raw.columns:
    raise ValueError("Unexpected FF file format: first column not 'Unnamed: 0'.")

ff_raw = ff_raw.rename(columns={"Unnamed: 0": "date"})

# Strip spaces from column names
ff_raw.columns = [c.strip() for c in ff_raw.columns]

# Convert date from YYYYMMDD to datetime, drop rows that can't be parsed
ff_raw["date"] = pd.to_datetime(ff_raw["date"].astype(str), format="%Y%m%d", errors="coerce")
ff_raw = ff_raw.dropna(subset=["date"])

factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
for col in factor_cols:
    if col not in ff_raw.columns:
        raise ValueError(f"FF data missing required column: {col}")
    ff_raw[col] = pd.to_numeric(ff_raw[col], errors="coerce") / 100.0  # percent → decimal

ff5 = ff_raw[["date"] + factor_cols].copy()

# ----------------------------------------------------------
# Merge with portfolio returns and run regression
# ----------------------------------------------------------
print("Running Fama–French 5-factor regression...")

merged = daily_df.merge(ff5, on="date", how="inner")

# Drop rows with any NaNs in required columns
needed_for_reg = ["portfolio_ret"] + factor_cols
reg_df = merged.dropna(subset=needed_for_reg).copy()

# Excess return over risk-free
reg_df["excess_ret"] = reg_df["portfolio_ret"] - reg_df["RF"]

X = reg_df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].astype(float)
X = sm.add_constant(X)
y = reg_df["excess_ret"].astype(float)

model = sm.OLS(y, X).fit()

with open(OUT_FOLDER / "ff5_regression.txt", "w") as f:
    f.write(model.summary().as_text())

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
print("Plotting cumulative return curve...")

cumret_series = (1.0 + ret_series).cumprod()
plt.figure(figsize=(10, 5))
plt.plot(common_dates, cumret_series, label="Portfolio")
plt.title("Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth of $1")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_FOLDER / "cumulative_return.png", dpi=200)
plt.close()

# Save daily returns series
daily_df.to_csv(OUT_FOLDER / "daily_portfolio_returns.csv", index=False)

print("✅ Backtest complete.")
print(f"📁 Results saved to: {OUT_FOLDER}")
