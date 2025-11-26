# src/baselines/equal_weight.py
import numpy as np
import pandas as pd

def equity_curve_equal_weight(prices_df: pd.DataFrame) -> pd.Series:
    prices_df = prices_df.dropna()
    n = prices_df.shape[1]
    if n == 0 or len(prices_df) == 0:
        raise ValueError("Empty price DataFrame.")
    weights = np.full(n, 1 / n)
    rel = prices_df / prices_df.iloc[0]
    equity = (rel * weights).sum(axis=1)
    equity.name = "equity_equal_weight"
    return equity
