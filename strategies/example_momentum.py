import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    Momentum strategy:
    - Rank stocks by 12-month return (excluding the most recent month)
    - Hold top 20% monthly, equally weighted
    """

    lookback = 252       # ~12 months
    skip = 21            # skip most recent month
    top_quantile = 0.2   # top 20%

    past_ret = prices.pct_change(lookback + skip).shift(skip)

    # Use month-end dates for rebalancing
    monthly_dates = prices.resample("M").last().index
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for date in monthly_dates:
        # Find the closest date available in the data
        if date not in past_ret.index:
            nearest_date = past_ret.index[past_ret.index.get_indexer([date], method='nearest')[0]]
        else:
            nearest_date = date

        mom_scores = past_ret.loc[nearest_date].dropna()
        if mom_scores.empty:
            continue

        cutoff = mom_scores.quantile(1 - top_quantile)
        winners = mom_scores[mom_scores >= cutoff].index

        w = pd.Series(0.0, index=prices.columns)
        if len(winners) > 0:
            w[winners] = 1.0 / len(winners)
        weights.loc[nearest_date] = w

    # Forward fill between rebalances
    weights = weights.ffill().fillna(0)
    return weights
