#!/usr/bin/env python3
"""
Compare Strategy vs Benchmark Performance
-----------------------------------------

Usage:
    python visualization/compare_strategy_vs_benchmark.py --file strategy_returns_20251029_1832.csv

Inputs:
- A strategy returns CSV (daily returns from backtest)
- CRSP benchmark data from data/crsp_sp500_10yr.csv

Outputs:
- Cumulative performance, rolling Sharpe, and drawdown plots
- Automatically saved PNG under data/strategy_visualizations/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize strategy vs benchmark performance.")
parser.add_argument("--file", type=str, required=True, help="Filename of strategy returns CSV (in /data folder)")
args = parser.parse_args()

# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025")
DATA_DIR = BASE_DIR / "data"
VIS_DIR = DATA_DIR / "strategy_visualizations"
VIS_DIR.mkdir(exist_ok=True)

CRSP_FILE = DATA_DIR / "crsp_sp500_10yr.csv"
STRATEGY_FILE = DATA_DIR / args.file

if not STRATEGY_FILE.exists():
    raise FileNotFoundError(f"Strategy file not found: {STRATEGY_FILE}")

print(f"Loading strategy returns from: {STRATEGY_FILE}")

# ---------------------------------------------------------------------
# Load CRSP data (benchmark)
# ---------------------------------------------------------------------
df = pd.read_csv(CRSP_FILE)
df.columns = [c.strip().lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df["prc"] = df["prc"].abs()
rets = df.pivot(index="date", columns="lpermno", values="ret_total").fillna(0)

# Equal-weighted benchmark
benchmark = rets.mean(axis=1)
benchmark_cum = (1 + benchmark).cumprod()

# ---------------------------------------------------------------------
# Load strategy returns
# ---------------------------------------------------------------------
strategy_rets = pd.read_csv(STRATEGY_FILE, index_col="date", parse_dates=True).squeeze("columns")
strategy_cum = (1 + strategy_rets).cumprod()

# Align both
common_idx = strategy_cum.index.intersection(benchmark_cum.index)
strategy_cum = strategy_cum.loc[common_idx]
benchmark_cum = benchmark_cum.loc[common_idx]
strategy_rets = strategy_rets.loc[common_idx]
benchmark = benchmark.loc[common_idx]

# ---------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------
rolling_window = 252
rolling_sharpe = (
    (strategy_rets.rolling(rolling_window).mean() * 252)
    / (strategy_rets.rolling(rolling_window).std() * np.sqrt(252))
)
cummax = (1 + strategy_rets).cumprod().cummax()
rolling_drawdown = ((1 + strategy_rets).cumprod() / cummax - 1)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                         gridspec_kw={'height_ratios': [2.5, 1, 1]})

# 1️⃣ Cumulative performance
axes[0].plot(strategy_cum, label="Strategy", color="#2E86AB", linewidth=2.2)
axes[0].plot(benchmark_cum, label="Equal-Weighted S&P 500", color="#E74C3C", linewidth=2.2)
axes[0].set_title(f"Strategy vs Benchmark Performance ({common_idx.min().date()} – {common_idx.max().date()})",
                  fontsize=15, fontweight="bold")
axes[0].set_ylabel("Cumulative Return (×)", fontsize=11)
axes[0].legend(frameon=False, fontsize=10, loc="upper left")

axes[0].text(strategy_cum.index[-250], strategy_cum.iloc[-1]*0.9,
             f"Strategy: {strategy_cum.iloc[-1]-1:.1%}", color="#2E86AB", fontsize=10, fontweight="bold")
axes[0].text(benchmark_cum.index[-250], benchmark_cum.iloc[-1]*0.9,
             f"Benchmark: {benchmark_cum.iloc[-1]-1:.1%}", color="#E74C3C", fontsize=10, fontweight="bold")

# 2️⃣ Rolling Sharpe ratio
axes[1].plot(rolling_sharpe, color="#2E86AB", linewidth=1.8)
axes[1].axhline(0, color="gray", lw=1, linestyle="--")
axes[1].set_ylabel("Rolling Sharpe (252d)", fontsize=11)
axes[1].set_title("Rolling Sharpe Ratio", fontsize=12)

# 3️⃣ Drawdown
axes[2].fill_between(rolling_drawdown.index, rolling_drawdown, 0, color="#E74C3C", alpha=0.4)
axes[2].set_ylabel("Drawdown", fontsize=11)
axes[2].set_xlabel("Date", fontsize=11)
axes[2].set_title("Rolling Drawdown", fontsize=12)

plt.tight_layout()

# ---------------------------------------------------------------------
# Save visualization
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_file = VIS_DIR / f"visualization_{args.file.replace('.csv','')}_{timestamp}.png"
plt.savefig(output_file, dpi=300)
print(f"Saved visualization to: {output_file}")

plt.show()
