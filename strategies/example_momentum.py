import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    Enhanced 12-month momentum strategy (top 25 stocks, quarterly rebalance)
    - Excludes the most recent month from lookback
    - Selects top 25 by momentum each quarter
    - Scales positions by inverse volatility
    - Reallocates delisted stocks to surviving names
    """

    prices.index = pd.to_datetime(prices.index)

    # --- STEP 1: Compute 12-month momentum (skip most recent month) ---
    monthly_prices = prices.resample("M").last()
    momentum = (monthly_prices / monthly_prices.shift(12)) - 1
    momentum = momentum.shift(1)  # exclude most recent month

    # --- STEP 2: Rolling volatility (1-year daily std) ---
    vol = rets.rolling(252).std().resample("M").last()
    vol = vol.replace(0, np.nan)

    # --- STEP 3: Quarterly rebalance ---
    rebalance_dates = momentum.resample("Q").last().index
    weights_list = []

    for date in rebalance_dates:
        mom_scores = momentum.loc[date].dropna()
        vol_scores = vol.loc[date].dropna()
        valid = mom_scores.index.intersection(vol_scores.index)
        if len(valid) < 25:
            continue

        mom_scores = mom_scores.loc[valid]
        vol_scores = vol_scores.loc[valid]

        # Select top 25 momentum stocks
        top = mom_scores.nlargest(25)

        # Inverse-volatility weighting
        inv_vol = 1 / vol_scores.loc[top.index]
        inv_vol = inv_vol / inv_vol.sum()

        weights = pd.Series(inv_vol, name=date)
        weights_list.append(weights)

    # --- STEP 4: Expand to daily, forward-fill, and renormalize ---
    weights_df = pd.concat(weights_list, axis=1).T
    weights_df = weights_df.reindex(prices.index).ffill().fillna(0)

    # 🔧 Keep total exposure = 100% after delistings
    weights_df = weights_df.div(weights_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    return weights_df
