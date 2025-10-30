#!/usr/bin/env python3
"""
Monthly Top Holdings Bar Plots + Animation
------------------------------------------
Usage:
    python visualization/plot_top_holdings_bar_with_gif.py \
        --file data/strategy_visualizations/monthly_holdings_tickers_weights_20251029_1957_20251029_1957.csv \
        --top 10
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize top holdings as monthly bar charts + animation.")
parser.add_argument("--file", type=str, required=True, help="Path to monthly holdings CSV with tickers as columns.")
parser.add_argument("--top", type=int, default=10, help="Number of top tickers per month.")
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
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv(file_path, index_col=0, parse_dates=True).fillna(0)
df.index.name = "Date"

# Create persistent color map
unique_tickers = df.columns
color_map = {t: plt.cm.tab20(i % 20) for i, t in enumerate(unique_tickers)}

# ---------------------------------------------------------------------
# Generate monthly bar plots
# ---------------------------------------------------------------------
frames = []
for date, row in df.iterrows():
    top = row.nlargest(args.top).sort_values()
    colors = [color_map[t] for t in top.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot.barh(ax=ax, color=colors)

    ax.set_title(f"Top {args.top} Holdings — {date.strftime('%b %Y')}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Portfolio Weight", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylabel("Ticker", fontsize=11)
    for i, v in enumerate(top.values):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
    plt.tight_layout()

    # Save each frame
    out_path = VIS_DIR / f"top{args.top}_holdings_{date.strftime('%Y%m')}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    frames.append(Image.open(out_path))

# ---------------------------------------------------------------------
# Create single legend
# ---------------------------------------------------------------------
fig_leg, ax_leg = plt.subplots(figsize=(10, 1))
handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[t]) for t in list(color_map)[:20]]
ax_leg.legend(handles, list(color_map)[:20], loc='center', ncol=10, frameon=False)
ax_leg.axis('off')
legend_path = VIS_DIR / "holdings_color_legend.png"
plt.savefig(legend_path, dpi=300)
plt.close(fig_leg)

# ---------------------------------------------------------------------
# Combine frames into an animated GIF
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
gif_path = VIS_DIR / f"holdings_animation_{file_path.stem}_{timestamp}.gif"
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=800,   # milliseconds per frame
    loop=0
)

print(f"✅ Saved monthly bar charts and animation to: {VIS_DIR}")
print(f"🎞 Animated GIF: {gif_path}")
print(f"🎨 Color legend: {legend_path}")
