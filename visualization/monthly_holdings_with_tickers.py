#!/usr/bin/env python3
"""
Generate Monthly Holdings with Ticker Names
-------------------------------------------

Usage:
    python visualization/monthly_holdings_with_tickers.py \
        --weights data/weights_20251029_1939.csv \
        --tickers data/crsp_stocknames.csv

Output:
    data/strategy_visualizations/monthly_holdings_tickers_YYYYMMDD_HHMM.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Convert daily weights to monthly top-10 tickers with tickernames.")
parser.add_argument("--weights", type=str, required=True, help="Path to daily weights CSV file.")
parser.add_argument("--tickers", type=str, required=True, help="Path to CRSP stocknames file.")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "strategy_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Load weights
# ---------------------------------------------------------------------
weights = pd.read_csv(args.weights, index_col=0, parse_dates=True)
weights = weights.loc[:, (weights != 0).any(axis=0)].fillna(0)

# ---------------------------------------------------------------------
# Load ticker lookup (CRSP stocknames)
# ---------------------------------------------------------------------
lookup = pd.read_csv(args.tickers)
lookup.columns = [c.strip().lower() for c in lookup.columns]

if "permno" not in lookup.columns or "ticker" not in lookup.columns:
    raise KeyError("Ticker file must contain 'permno' and 'ticker' columns.")

permno_to_ticker = dict(zip(lookup["permno"], lookup["ticker"]))

def rename_column(c):
    try:
        return permno_to_ticker.get(int(c), c)
    except ValueError:
        return c

weights.columns = [rename_column(c) for c in weights.columns]

# ---------------------------------------------------------------------
# Aggregate to monthly average weights
# ---------------------------------------------------------------------
monthly = weights.resample("ME").mean()

# ---------------------------------------------------------------------
# Select top 10 tickers per month
# ---------------------------------------------------------------------
monthly_top10 = []
for date, row in monthly.iterrows():
    top = row.nlargest(10)
    top.name = date
    monthly_top10.append(top)

monthly_top10_df = pd.DataFrame(monthly_top10).fillna(0)

# ---------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
out_path = OUTPUT_DIR / f"monthly_holdings_tickers_{Path(args.weights).stem}_{timestamp}.csv"
monthly_top10_df.to_csv(out_path, float_format="%.6f")

print(f"✅ Saved monthly top-10 holdings (with tickers) to: {out_path}")
print(f"Shape: {monthly_top10_df.shape}")
print(monthly_top10_df.tail(5))
