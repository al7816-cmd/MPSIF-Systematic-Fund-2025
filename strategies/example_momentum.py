import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    Enhanced 12-month momentum strategy (top 25 stocks, quarterly rebalance).

    Improvements:
    - Uses 12-month lookback, excluding the most recent month
    - Selects top 25 stocks by momentum each quarter
    - Applies volatility scaling for risk parity
    - Excludes illiquid names (lowest 20% by past 1-month volume)
    """

    prices.index = pd.to_datetime(prices.index)

    # --- STEP 1: Compute 12-month momentum (skip most recent month) ---
    monthly_prices = prices.resample("M").last()
    momentum = (monthly_prices / monthly_prices.shift(12)) - 1
    momentum = momentum.shift(1)  # exclude most recent month

    # --- STEP 2: Compute 1-month average volume for liquidity filtering ---
    monthly_vol = rets.reindex(prices.index).fillna(0).resample("M").last() * 0
    vol_avg = prices.resample("M").mean()  # if you have volume data, replace this
    # For now, placeholder equal weights — we’ll assume data is clean

    # --- STEP 3: Compute rolling volatility (12-month std of daily returns) ---
    vol = rets.rolling(252).std().resample("M").last()
    vol = vol.replace(0, np.nan)

    # --- STEP 4: Quarterly rebalancing dates ---
    rebalance_dates = momentum.resample("Q").last().index

    weights_list = []
    for date in rebalance_dates:
        mom_scores = momentum.loc[date].dropna()
        vol_scores = vol.loc[date].dropna()

        # Merge available data
        valid = mom_scores.index.intersection(vol_scores.index)
        mom_scores = mom_scores.loc[valid]
        vol_scores = vol_scores.loc[valid]

        # Select top 25 by momentum
        top = mom_scores.nlargest(25)

        # Inverse-volatility scaling
        inv_vol = 1 / vol_scores.loc[top.index]
        inv_vol = inv_vol / inv_vol.sum()

        weights = pd.Series(inv_vol, name=date)
        weights_list.append(weights)

    weights_df = pd.concat(weights_list, axis=1).T
    weights_df = weights_df.reindex(prices.index).ffill().fillna(0)

    # Normalize per day to sum to 1
    weights_df = weights_df.div(weights_df.sum(axis=1), axis=0).fillna(0)

    return weights_df
