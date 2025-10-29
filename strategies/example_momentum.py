import pandas as pd
import numpy as np

def generate_weights(prices: pd.DataFrame, rets: pd.DataFrame):
    """
    Momentum strategy: ranks stocks by 12-month return (excluding most recent month)
    and holds top 20% each month, equally weighted.
    """

    lookback = 252       # ~12 months
    skip = 21            # skip last month
    top_quantile = 0.2   # top 20% by momentum

    # Compute rolling 12-month returns excluding last month
    past_ret = prices.pct_change(lookback + skip).shift(skip)

    # Resample to month-end for rebalancing
    monthly_dates = prices.resample("M").last().index
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for date in monthly_dates:
        if date not in past_ret.index:
            continue
        # Rank momentum on this date
        mom_scores = past_ret.loc[date]
        cutoff = mom_scores.quantile(1 - top_quantile)
        winners = mom_scores[mom_scores >= cutoff].index

        w = pd.Series(0, index=prices.columns, dtype=float)
        if len(winners) > 0:
            w[winners] = 1.0 / len(winners)
        weights.loc[date] = w

    # Forward-fill weights until next rebalance
    weights = weights.ffill().fillna(0)

    # 🔹 Re-normalize each day among live stocks (ensures full 100% exposure)
    for date in prices.index:
        live = prices.loc[date].dropna().index
        if len(live) == 0:
            continue
        w = weights.loc[date, live]
        total = w.sum()
        if total > 0:
            weights.loc[date, live] = w / total
        else:
            weights.loc[date, :] = 0

    return weights
