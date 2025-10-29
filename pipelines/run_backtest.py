#!/usr/bin/env python3
"""
Run Backtest for Systematic Fund Strategies
-------------------------------------------

This script:
- Loads CRSP data
- Runs a selected strategy (from /strategies)
- Computes performance metrics
- Saves daily strategy returns to /data/strategy_returns_<timestamp>.csv

Usage:
    python pipelines/run_backtest.py --strategy strategies/example_momentum.py
"""

import argparse
import importlib.util
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run backtest for a given strategy.")
parser.add_argument("--strategy", type=str, required=True, help="Path to strategy file.")
args = parser.parse_args()

STRATEGY_PATH = args.strategy
DATA_PATH = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data/crsp_sp500_10yr.csv")
OUTPUT_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Load CRSP data
# ---------------------------------------------------------------------
print("Loading CRSP data...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df["prc"] = df["prc"].abs()
df = df.sort_values(["lpermno", "date"])

print(f"Loaded {len(df):,} rows from {df['date'].min().date()} to {df['date'].max().date()}.")

# Pivot to wide format
prices = df.pivot(index="date", columns="lpermno", values="prc").fillna(method="ffill")
rets = df.pivot(index="date", columns="lpermno", values="ret_total").fillna(0)

# ---------------------------------------------------------------------
# Load strategy module
# ---------------------------------------------------------------------
print(f"Loaded strategy: {STRATEGY_PATH}")
spec = importlib.util.spec_from_file_location("strategy", STRATEGY_PATH)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)

# ---------------------------------------------------------------------
# Run strategy
# ---------------------------------------------------------------------
print("Running strategy...")
weights = strategy.generate_weights(prices, rets)

# Ensure weights and returns aligned
weights, rets = weights.align(rets, join="inner", axis=0)
daily_portfolio_rets = (weights.shift(1) * rets).sum(axis=1)
strategy_rets = daily_portfolio_rets.fillna(0)

# ---------------------------------------------------------------------
# Compute performance metrics
# ---------------------------------------------------------------------
total_return = (1 + strategy_rets).prod() - 1
ann_return = (1 + total_return) ** (252 / len(strategy_rets)) - 1
ann_vol = strategy_rets.std() * np.sqrt(252)
sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
cummax = (1 + strategy_rets).cumprod().cummax()
drawdown = ((1 + strategy_rets).cumprod() / cummax - 1).min()

print("\n===== Performance Summary =====")
print(f"Total Return:      {total_return:8.2%}")
print(f"Annualized Return: {ann_return:8.2%}")
print(f"Annualized Vol:    {ann_vol:8.2%}")
print(f"Sharpe Ratio:      {sharpe:8.2f}")
print(f"Max Drawdown:      {drawdown:8.2%}")

# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
file_name = f"strategy_returns_{timestamp}.csv"
output_path = OUTPUT_DIR / file_name
strategy_rets.to_csv(output_path, index_label="date")

print(f"\nSaved daily strategy returns to: {output_path}")

# Optionally, save summary log for team tracking
summary_log = OUTPUT_DIR / "backtest_results.csv"
summary_entry = pd.DataFrame([{
    "timestamp": timestamp,
    "strategy": Path(STRATEGY_PATH).stem,
    "total_return": total_return,
    "annualized_return": ann_return,
    "annualized_vol": ann_vol,
    "sharpe_ratio": sharpe,
    "max_drawdown": drawdown
}])

if summary_log.exists():
    existing = pd.read_csv(summary_log)
    combined = pd.concat([existing, summary_entry], ignore_index=True)
    combined.to_csv(summary_log, index=False)
else:
    summary_entry.to_csv(summary_log, index=False)

print(f"Appended summary results to: {summary_log}")
