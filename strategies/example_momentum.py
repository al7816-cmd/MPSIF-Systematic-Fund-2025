import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    Momentum strategy:
    - Ranks stocks by 12-month return (excluding most recent month)
    - Selects top 10
    - Allocates proportional to momentum scores
    - Renormalizes to ensure total = 1.0 every month
    """

    lookback = 252       # ~12 months
    skip = 21            # skip most recent month
    top_n = 10           # top 10 by momentum

    past_ret = prices.pct_change(lookback + skip).shift(skip)
    monthly_dates = prices.resample("M").last().index
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for date in monthly_dates:
        if date not in past_ret.index:
            continue

        # Compute momentum
        mom_scores = past_ret.loc[date].dropna()
        if mom_scores.empty:
            continue

        top = mom_scores.nlargest(top_n).clip(lower=0)

        if top.sum() == 0:
            w = pd.Series(0, index=prices.columns)
        else:
            # Normalize so total = 1
            w = top / top.sum()

        weights.loc[date, :] = 0
        weights.loc[date, w.index] = w

    # Forward fill between rebalances
    weights = weights.ffill().fillna(0)
    return weights
