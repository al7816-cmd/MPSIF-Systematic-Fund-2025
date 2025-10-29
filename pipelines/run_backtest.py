import pandas as pd
import numpy as np
import argparse
import importlib.util
import os
from datetime import datetime

def load_strategy(strategy_path):
    """Dynamically import a strategy file."""
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    return strategy

def backtest(prices, rets, strategy, full_investment=True):
    """
    Run backtest using dynamic capital tracking.
    - Keeps portfolio 100% invested
    - Rebalances on strategy’s own schedule
    """
    print("Running strategy...")

    weights = strategy.generate_weights(prices, rets)
    weights = weights.reindex(prices.index).fillna(method="ffill").fillna(0)

    # Ensure daily full investment exposure if specified
    if full_investment:
        weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Compute portfolio daily returns
    port_rets = (weights.shift(1) * rets).sum(axis=1)
    port_rets = port_rets.fillna(0)

    # Track portfolio value over time
    port_val = (1 + port_rets).cumprod()

    # Compute performance metrics
    total_ret = port_val.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / len(port_val)) - 1
    ann_vol = port_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    running_max = port_val.cummax()
    drawdown = (port_val - running_max) / running_max
    max_dd = drawdown.min()

    print("\n===== Performance Summary =====")
    print(f"Total Return:      {total_ret * 100:8.2f}%")
    print(f"Annualized Return: {ann_ret * 100:8.2f}%")
    print(f"Annualized Vol:    {ann_vol * 100:8.2f}%")
    print(f"Sharpe Ratio:      {sharpe:8.2f}")
    print(f"Max Drawdown:      {max_dd * 100:8.2f}%")

    return port_val, port_rets, weights

def main():
    parser = argparse.ArgumentParser(description="Run systematic backtest")
    parser.add_argument("--strategy", required=True, help="Path to strategy .py file")
    args = parser.parse_args()

    print("Loading CRSP data...")
    data_path = "/home/nyu/willwu24/MPSIF-Systematic-Fund-2025/data/crsp_sp500_10yr.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])

    df = df.rename(columns={"lpermno": "permno"})
    df = df.sort_values(["permno", "date"])
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

    prices = df.pivot(index="date", columns="permno", values="prc")
    rets = df.pivot(index="date", columns="permno", values="ret")

    print(f"Loaded {len(df):,} rows from {df.date.min().date()} to {df.date.max().date()}.")

    strategy = load_strategy(args.strategy)
    port_val, port_rets, weights = backtest(prices, rets, strategy)

    # Save results
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_path = f"data/strategy_returns_{timestamp}.csv"
    weights_path = f"data/weights_{timestamp}.csv"

    pd.DataFrame({
        "date": port_val.index,
        "portfolio_value": port_val.values,
        "daily_return": port_rets.values
    }).to_csv(results_path, index=False)

    weights.to_csv(weights_path)

    print(f"\nSaved backtest results to {results_path}")
    print(f"Saved weight history to {weights_path}")

if __name__ == "__main__":
    main()
