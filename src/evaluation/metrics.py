# src/evaluation/metrics.py

import pandas as pd


def simple_metrics(equity: pd.Series) -> dict:
    """
    Compute basic financial metrics from a portfolio equity curve:
    - total return
    - daily volatility
    - Sharpe ratio (rf = 0)
    """
    returns = equity.pct_change().dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    volatility = float(returns.std())  # daily std
    sharpe = float(0 if volatility == 0 else returns.mean() / volatility)

    return {
        "total_return": total_return,
        "daily_volatility": volatility,
        "daily_sharpe": sharpe,
    }
