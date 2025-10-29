#!/usr/bin/env python3
"""
Rebuild and Visualize Strategy Holdings With Tickers
---------------------------------------------------

Usage:
    python visualization/rebuild_holdings_visualization.py --strategy strategies/example_momentum.py

This script:
- Reloads CRSP price data
- Recomputes holdings using the given strategy logic
- Maps PERMNOs to tickers via CRSP stocknames
- Produces a stacked area chart of top holdings
- Saves output to data/strategy_visualizations/
"""

import argparse
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Rebuild and visualize holdings for a given strategy.")
parser.add_argument("--strategy", type=str, required=True, help="Path to strategy file.")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025")
DATA_DIR = BASE_DIR / "data"
VIS_DIR = DATA_DIR / "strategy_visualizations"
VIS_DIR.mkdir(exist_ok=True)

CRSP_FILE = DATA_DIR / "crsp_sp500_10yr.csv"
TICKER_FILE = DATA_DIR / "crsp_stocknames.csv"

# ---------------------------------------------------------------------
# Load CRSP data
# ---------------------------------------------------------------------
print("Loading CRSP data...")
df = pd.read_csv(CRSP_FILE)
df.columns = [c.strip().lower() for c in df.columns]
df["date"] = pd.to_datetime(df["date"])
df["prc"] = df["prc"].abs()
df = df.sort_values(["lpermno", "date"])

prices = df.pivot(index="date", columns="lpermno", values="prc").fillna(method="ffill")
rets = df.pivot(index="date", columns="lpermno", values="ret_total").fillna(0)

print(f"Loaded {len(df):,} rows from {df['date'].min().date()} to {df['date'].max().date()}.")

# ---------------------------------------------------------------------
# Load strategy dynamically
# ---------------------------------------------------------------------
print(f"Loaded strategy: {args.strategy}")
spec = importlib.util.spec_from_file_location("strategy", args.strategy)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)

# ---------------------------------------------------------------------
# Generate weights using strategy logic
# ---------------------------------------------------------------------
weights = strategy.generate_weights(prices, rets)

# ---------------------------------------------------------------------
# Map lpermno → ticker names (if lookup file exists)
# ---------------------------------------------------------------------
if TICKER_FILE.exists():
    lookup = pd.read_csv(TICKER_FILE)
    lookup = lookup.dropna(subset=["ticker"])
    permno_to_ticker = dict(zip(lookup["permno"], lookup["ticker"]))

    # Rename columns if match found
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
weights_top = weights[top_assets]
weights_top["Others"] = 1 - weights_top.sum(axis=1)
weights_top["Others"] = weights_top["Others"].clip(lower=0)

# ---------------------------------------------------------------------
# Plot holdings
# ---------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))
weights_top.plot.area(ax=ax, cmap="tab20", linewidth=0)

ax.set_title(f"Portfolio Holdings Over Time — {Path(args.strategy).stem}",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Portfolio Weight", fontsize=11)
ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)

plt.tight_layout()

# ---------------------------------------------------------------------
# Save visualization
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_path = VIS_DIR / f"holdings_{Path(args.strategy).stem}_{timestamp}.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved holdings visualization to: {output_path}")
