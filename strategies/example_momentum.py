import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    Momentum strategy:
    - Ranks stocks by 12-month return (excluding most recent month)
    - Selects top 10
    - Allocates weights proportional to momentum scores (so total = 1)
    """

    lookback = 252       # ~12 months
    skip = 21            # skip most recent month
    top_n = 10           # top 10 by momentum

    # Compute rolling 12-month returns excluding last month
    past_ret = prices.pct_change(lookback + skip).shift(skip)

    # Monthly rebalancing
    monthly_dates = prices.resample("M").last().index
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for date in monthly_dates:
        if date not in past_ret.index:
            continue

        # Momentum scores (remove NaN and negative values)
        mom_scores = past_ret.loc[date].dropna()
        if mom_scores.empty:
            continue

        # Select top N
        top = mom_scores.nlargest(top_n)

        # Convert raw momentum scores to proportional weights
        positive_scores = top.clip(lower=0)  # drop negatives
        if positive_scores.sum() == 0:
            w = pd.Series(0, index=prices.columns)
        else:
            w = positive_scores / positive_scores.sum()

        # Assign weights for these tickers only
        weights.loc[date, w.index] = w

    # Forward-fill until next rebalance
    weights = weights.ffill().fillna(0)

    return weights
