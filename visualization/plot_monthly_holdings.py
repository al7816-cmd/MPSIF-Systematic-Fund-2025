#!/usr/bin/env python3
"""
Visualize Monthly Portfolio Holdings (Stacked Area)
--------------------------------------------------

Usage:
    python visualization/plot_monthly_holdings.py \
        --file data/strategy_visualizations/monthly_holdings_tickers_weights_20251029_1957_20251029_1957.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize monthly holdings from a weights CSV.")
parser.add_argument("--file", type=str, required=True, help="Path to the monthly holdings CSV file.")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025")
DATA_DIR = BASE_DIR / "data"
VIS_DIR = DATA_DIR / "strategy_visualizations"
VIS_DIR.mkdir(exist_ok=True)

file_path = BASE_DIR / args.file
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

# ---------------------------------------------------------------------
# Load monthly holdings
# ---------------------------------------------------------------------
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
df = df.fillna(0)
df.index.name = "Date"

# Keep top N tickers overall (for cleaner visualization)
top_n = 10
top_tickers = df.mean().nlargest(top_n).index
df_top = df[top_tickers].copy()
df_top["Others"] = 1 - df_top.sum(axis=1)
df_top["Others"] = df_top["Others"].clip(lower=0)

# ---------------------------------------------------------------------
# Plot stacked area
# ---------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
df_top.plot.area(ax=ax, cmap="tab20", linewidth=0)

ax.set_title("Monthly Portfolio Holdings Over Time", fontsize=15, fontweight="bold")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Portfolio Weight", fontsize=11)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)

plt.tight_layout()

# ---------------------------------------------------------------------
# Save visualization
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
out_path = VIS_DIR / f"holdings_visualization_{file_path.stem}_{timestamp}.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"✅ Saved holdings visualization to: {out_path}")
