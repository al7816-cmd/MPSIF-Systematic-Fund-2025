import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    12-month momentum strategy with monthly rebalancing
    ---------------------------------------------------
    - Skip the most recent month
    - Compute trailing 12-month returns for all stocks
    - Pick top 25 by momentum each month
    - Weight by inverse volatility over past year
    - Normalize to full exposure every month
    """

    prices.index = pd.to_datetime(prices.index)

    # --- 1. Monthly data ---
    monthly_prices = prices.resample("M").last()
    momentum = (monthly_prices / monthly_prices.shift(12)) - 1
    momentum = momentum.shift(1)  # skip most recent month
    vol = rets.rolling(252).std().resample("M").last().replace(0, np.nan)

    # --- 2. Monthly rebalance dates ---
    rebalance_dates = momentum.index
    weight_list = []

    for date in rebalance_dates:
        mom = momentum.loc[date].dropna()
        sigma = vol.loc[date].dropna()
        common = mom.index.intersection(sigma.index)
        if len(common) < 25:
            continue

        mom = mom.loc[common]
        sigma = sigma.loc[common]

        # top 25 by momentum
        top = mom.nlargest(25)
        inv_vol = 1 / sigma.loc[top.index]
        inv_vol = inv_vol / inv_vol.sum()
        weight_list.append(pd.Series(inv_vol, name=date))

    weights = pd.concat(weight_list, axis=1).T

    # --- 3. Expand to daily frequency (ffill within each month) ---
    weights = weights.reindex(prices.index).ffill().fillna(0)

    # --- 4. Normalize exposure daily among live stocks ---
    for date in prices.index:
        live = prices.loc[date].dropna().index.intersection(weights.columns)
        if len(live) == 0:
            continue
        w = weights.loc[date, live]
        total = w.sum()
        if total > 0:
            weights.loc[date, live] = w / total
        else:
            weights.loc[date, :] = 0

    return weights
