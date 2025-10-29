#!/usr/bin/env python3
"""
Visualize Portfolio Holdings with True Weights
----------------------------------------------

Usage:
    python visualization/holdings_true_weights.py --weights data/weights_20251029_1926.csv

Displays the actual percentage weight held in each top ticker (no scaling).
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize true holdings from weights CSV.")
parser.add_argument("--weights", type=str, required=True, help="Path to weights CSV file.")
args = parser.parse_args()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025")
DATA_DIR = BASE_DIR / "data"
VIS_DIR = DATA_DIR / "strategy_visualizations"
VIS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Load weights
# ---------------------------------------------------------------------
print(f"Loading weights file: {args.weights}")
weights = pd.read_csv(args.weights, index_col=0, parse_dates=True)

# ---------------------------------------------------------------------
# Check total portfolio weight
# ---------------------------------------------------------------------
weights["total_weight"] = weights.sum(axis=1)
print(f"Median total portfolio weight: {weights['total_weight'].median():.3f}")

# ---------------------------------------------------------------------
# Select top assets by average weight
# ---------------------------------------------------------------------
top_n = 10
top_assets = weights.drop(columns=["total_weight"], errors="ignore").mean().nlargest(top_n).index
weights_top = weights[top_assets].copy()

# Compute "Others" = all remaining weights
weights_top["Others"] = (weights.drop(columns=["total_weight"], errors="ignore").sum(axis=1)
                         - weights_top.sum(axis=1)).clip(lower=0)

# ---------------------------------------------------------------------
# Plot
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
output_file = VIS_DIR / f"holdings_true_{Path(args.weights).stem}_{timestamp}.png"
plt.savefig(output_file, dpi=300)
plt.show()

print(f"✅ Saved true-weight holdings visualization to: {output_file}")
