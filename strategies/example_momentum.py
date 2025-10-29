import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    Robust 12-month momentum strategy (top 25 stocks, quarterly rebalance)
    - Excludes most recent month
    - Selects top 25 by momentum
    - Inverse-vol weighting
    - Quarterly rebalance
    - Dynamic renormalization each day across live stocks
    """

    prices.index = pd.to_datetime(prices.index)

    # --- 1. Compute monthly momentum and volatility ---
    monthly_prices = prices.resample("M").last()
    momentum = (monthly_prices / monthly_prices.shift(12)) - 1
    momentum = momentum.shift(1)  # exclude most recent month
    vol = rets.rolling(252).std().resample("M").last().replace(0, np.nan)

    # --- 2. Quarterly rebalance ---
    rebalance_dates = momentum.resample("Q").last().index
    weight_list = []

    for date in rebalance_dates:
        if date not in momentum.index:
            continue
        mom = momentum.loc[date].dropna()
        sigma = vol.loc[date].dropna()
        common = mom.index.intersection(sigma.index)
        if len(common) < 25:
            continue

        mom = mom.loc[common]
        sigma = sigma.loc[common]

        top = mom.nlargest(25)
        inv_vol = 1 / sigma.loc[top.index]
        inv_vol = inv_vol / inv_vol.sum()
        weight_list.append(pd.Series(inv_vol, name=date))

    weights = pd.concat(weight_list, axis=1).T

    # --- 3. Forward-fill to daily frequency ---
    weights = weights.reindex(prices.index).ffill().fillna(0)

    # --- 4. Daily renormalization only over live + existing columns ---
    for date in prices.index:
        live = prices.loc[date].dropna().index
        live = live.intersection(weights.columns)   # ✅ key fix
        if len(live) == 0:
            continue
        w = weights.loc[date, live]
        total = w.sum()
        if total > 0:
            weights.loc[date, live] = w / total
        else:
            weights.loc[date, :] = 0

    return weights
