# src/features/technical_indicators.py
import pandas as pd

def add_moving_averages(prices: pd.DataFrame, windows=(20, 60)) -> pd.DataFrame:
    df = prices.copy()
    for w in windows:
        df[f"MA{w}"] = prices.rolling(window=w).mean()
    return df

def add_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol = returns.rolling(window=window).std()
    vol.columns = [f"{c}_VOL{window}" for c in vol.columns]
    return vol
