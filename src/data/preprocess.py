# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.config import TEST_RATIO, SEED

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.dropna()
    return prices

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def split_train_test(prices: pd.DataFrame):
    np.random.seed(SEED)
    split_idx = int(len(prices) * (1 - TEST_RATIO))
    prices_train = prices.iloc[:split_idx]
    prices_test = prices.iloc[split_idx:]
    return prices_train, prices_test
