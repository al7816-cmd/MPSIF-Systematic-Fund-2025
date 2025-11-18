#!/usr/bin/env python3
"""
Final Backtest Engine (Column Alignment Fixed)
----------------------------------------------
This version ensures weights and price return matrices have identical permno
columns, fixing the broadcast error.

Outputs:
- summary_stats.csv
- daily_portfolio_returns.csv
- ff5_regression.txt
- cumulative_return.png
"""

# ------------------------------------------------------------------
# WRDS Cloud: force single-thread BLAS
# ------------------------------------------------------------------
import os
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

# ==================================================================
# CLI
# ==================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--prices", required=True, help="Compustat parquet path")
parser.add_argument("--weights", required=True, help="Weights CSV path")
parser.add_argument("--outdir", default="backtests")
args = parser.parse_args()

BASE_DIR = Path(".").resolve()
PRICES_PATH = BASE_DIR / args.prices
WEIGHTS_PATH = BASE_DIR / args.weights
FF_PATH = BASE_DIR / "data" / "FFM_Daily_Data.csv"

# ==================================================================
# Output folder
# ==================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_FOLDER = BASE_DIR / args.outdir / f"backtest_{timestamp}"
OUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"\n📁 Output folder: {OUT_FOLDER}")

# ==================================================================
# Load Compustat daily returns
# ==================================================================
print("Loading Compustat prices...")

prices = pd.read_parquet(PRICES_PATH)
prices = prices[prices["ret_daily"].notna()].copy()
prices = prices.pivot(index="date", columns="permno", values="ret_daily")

prices.columns = prices.columns.astype(int)

# ==================================================================
# Load weights
# ==================================================================
print("Loading weights...")

weights = pd.read_csv(WEIGHTS_PATH)
weights["date"] = pd.to_datetime(weights["date"])
weights.set_index("date", inplace=True)

weights.columns = weights.columns.astype(int)

# ==================================================================
# Align dates
# ==================================================================
common_dates = prices.index.intersection(weights.index)
print(f"Common dates: {common_dates.min().date()} → {common_dates.max().date()}")
print(f"{len(common_dates)} days")

prices = prices.loc[common_dates]
weights = weights.loc[common_dates]

# ==================================================================
# FIX: Align permno columns
# ==================================================================
print("Aligning permno columns...")

all_permnos = sorted(set(prices.columns) | set(weights.columns))

prices = prices.reindex(columns=all_permnos, fill_value=0.0)
weights = weights.reindex(columns=all_permnos, fill_value=0.0)

# ==================================================================
# Safe conversion to NumPy
# ==================================================================
returns_np = prices.to_numpy(dtype=float)
weights_np = weights.to_numpy(dtype=float)

# ==================================================================
# Portfolio returns
# ==================================================================
print("Computing portfolio returns...")

portfolio_ret = np.sum(weights_np * returns_np, axis=1)

daily_df = pd.DataFrame({
    "date": common_dates,
    "portfolio_ret": portfolio_ret
})

# ==================================================================
# Summary statistics
# ==================================================================
ann = 252

summary = pd.DataFrame({
    "cumulative_return": [(1 + portfolio_ret).prod() - 1],
    "annualized_return": [(1 + portfolio_ret).prod() ** (ann / len(portfolio_ret)) - 1],
    "annualized_vol": [np.std(portfolio_ret) * np.sqrt(ann)],
    "sharpe_ratio": [np.mean(portfolio_ret) / np.std(portfolio_ret)]
})

summary.to_csv(OUT_FOLDER / "summary_stats.csv", index=False)

# ==================================================================
# Load FF5 factors
# ==================================================================
print("\nLoading Fama-French factors...")

raw = pd.read_csv(FF_PATH, header=None)

header_row = raw[raw.apply(lambda r: r.astype(str).str.contains("Mkt-RF").any(), axis=1)].index[0]
ff5 = pd.read_csv(FF_PATH, skiprows=header_row)

ff5 = ff5.rename(columns={ff5.columns[0]: "date"})
ff5["date"] = pd.to_datetime(ff5["date"].astype(str), format="%Y%m%d", errors="coerce")
ff5 = ff5.dropna(subset=["date"])

ff5.columns = [c.replace("-", "_") for c in ff5.columns]

factor_cols = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
for c in factor_cols:
    ff5[c] = ff5[c].astype(float) / 100.0

# ==================================================================
# Merge returns + factors
# ==================================================================
print("Running FF5 regression...")

merged = daily_df.merge(ff5, on="date", how="inner")
merged["excess_ret"] = merged["portfolio_ret"] - merged["RF"]

X = merged[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]]
X = sm.add_constant(X)
y = merged["excess_ret"]

model = sm.OLS(y, X).fit()

with open(OUT_FOLDER / "ff5_regression.txt", "w") as f:
    f.write(model.summary().as_text())

# ==================================================================
# Plot cumulative return curve
# ==================================================================
print("Plotting cumulative return...")

cumret = (1 + portfolio_ret).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(common_dates, cumret)
plt.title("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_FOLDER / "cumulative_return.png", dpi=200)
plt.close()

daily_df.to_csv(OUT_FOLDER / "daily_portfolio_returns.csv", index=False)

print("\n✅ Backtest complete.")
print(f"📁 Results saved to: {OUT_FOLDER}\n")
