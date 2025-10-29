import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from datetime import datetime

def load_strategy_results(file_path):
    print(f"Loading strategy returns from: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values("date")
    df["daily_return"] = df["daily_return"].clip(-1, 1)  # avoid overflow
    df["cum_return"] = (1 + df["daily_return"]).cumprod()
    return df

def load_benchmark(start_date, end_date):
    # Simple synthetic benchmark — flat 8% annualized
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    daily_ret = (1 + 0.08) ** (1/252) - 1
    bench = pd.DataFrame({"date": dates, "benchmark_return": daily_ret})
    bench["cum_return"] = (1 + bench["benchmark_return"]).cumprod()
    return bench

def visualize(strategy_df, benchmark_df, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(strategy_df["date"], strategy_df["cum_return"], label="Strategy", color="#2E86AB", linewidth=2.2)
    plt.plot(benchmark_df["date"], benchmark_df["cum_return"], label="Benchmark", color="#B03A2E", linestyle="--", linewidth=1.8)

    plt.title("Strategy vs Benchmark Performance", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Growth of $1)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Handle scalar extraction safely
    strat_final = float(strategy_df["cum_return"].iloc[-1])
    bench_final = float(benchmark_df["cum_return"].iloc[-1])

    plt.text(strategy_df["date"].iloc[-1], strat_final, f"Strategy: {strat_final-1:.1%}",
             color="#2E86AB", fontsize=10, fontweight="bold", va="bottom", ha="right")
    plt.text(benchmark_df["date"].iloc[-1], bench_final, f"Benchmark: {bench_final-1:.1%}",
             color="#B03A2E", fontsize=10, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize strategy vs benchmark performance.")
    parser.add_argument("--file", required=True, help="Path to strategy returns CSV file")
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        file_path = f"/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data/{args.file}"

    df = load_strategy_results(file_path)
    benchmark = load_benchmark(df["date"].min(), df["date"].max())

    os.makedirs("/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data/strategy_visualizations", exist_ok=True)
    out_path = f"/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data/strategy_visualizations/visualization_{os.path.basename(file_path).replace('.csv','.png')}"
    visualize(df, benchmark, out_path)

    print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
