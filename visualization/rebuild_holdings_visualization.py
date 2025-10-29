#!/usr/bin/env python3
"""
Visualize Portfolio Holdings from Weights File
----------------------------------------------

Usage:
    python visualization/rebuild_holdings_visualization.py --weights weights_20251029_1926.csv

This script:
- Loads a weights CSV file (output from backtest)
- Maps PERMNOs to tickers (if available)
- Produces a stacked area chart of top holdings
- Saves output to data/strategy_visualizations/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize portfolio holdings from weights file.")
parser.add_argument("--weights", type=str, required=True, help="Weights CSV file in /data folder")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025")
DATA_DIR = BASE_DIR / "data"
VIS_DIR = DATA_DIR / "strategy_visualizations"
VIS_DIR.mkdir(exist_ok=True)

WEIGHTS_FILE = DATA_DIR / args.weights
TICKER_FILE = DATA_DIR / "crsp_stocknames.csv"

# ---------------------------------------------------------------------
# Load weights file
# ---------------------------------------------------------------------
print(f"Loading weights from: {WEIGHTS_FILE}")
weights = pd.read_csv(WEIGHTS_FILE, index_col=0, parse_dates=True)
weights = weights.sort_index()

# Normalize if necessary
weights_sum = weights.sum(axis=1)
if (weights_sum.mean() < 0.95) or (weights_sum.mean() > 1.05):
    print("⚠️ Warning: weights do not sum to 1 on average. Normalizing now.")
    weights = weights.div(weights_sum, axis=0).fillna(0)

# ---------------------------------------------------------------------
# Map lpermno → ticker names (if lookup file exists)
# ---------------------------------------------------------------------
if TICKER_FILE.exists():
    lookup = pd.read_csv(TICKER_FILE)
    lookup = lookup.dropna(subset=["ticker"])
    permno_to_ticker = dict(zip(lookup["permno"], lookup["ticker"]))

    def rename_column(c):
        try:
            return permno_to_ticker.get(int(c), c)
        except ValueError:
            return c

    weights.columns = [rename_column(c) for c in weights.columns]
else:
    print("⚠️ Ticker lookup file not found. Using numeric IDs instead.")

# ---------------------------------------------------------------------
# Keep top holdings for clarity
# ---------------------------------------------------------------------
top_n = 10
top_assets = weights.mean().nlargest(top_n).index
weights_top = weights[top_assets].copy()
weights_top["Others"] = 1 - weights_top.sum(axis=1)
weights_top["Others"] = weights_top["Others"].clip(lower=0)

# ---------------------------------------------------------------------
# Plot holdings
# ---------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
weights_top.plot.area(ax=ax, cmap="tab20", linewidth=0)

ax.set_title(f"Portfolio Holdings Over Time — {Path(args.weights).stem}",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Portfolio Weight", fontsize=11)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)

plt.tight_layout()

# ---------------------------------------------------------------------
# Save visualization
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_path = VIS_DIR / f"holdings_{Path(args.weights).stem}_{timestamp}.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved holdings visualization to: {output_path}")
