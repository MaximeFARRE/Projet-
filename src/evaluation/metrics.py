# src/evaluation/metrics.py
import pandas as pd

def simple_metrics(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    volatility = float(r.std())
    sharpe = float(0 if volatility == 0 else r.mean() / volatility)
    return {
        "Total Return": total_return,
        "Volatility (daily std)": volatility,
        "Sharpe (rf=0, daily)": sharpe,
    }
