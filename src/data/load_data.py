# src/data/load_data.py
import pandas as pd
import kagglehub
from . import preprocess
from src.config import RAW_DIR, TICKERS

def download_sp500_csv() -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("camnugent/sandp500")
    csv_path = f"{path}/all_stocks_5yr.csv"
    return csv_path

def load_prices() -> pd.DataFrame:
    csv_path = download_sp500_csv()
    df_raw = pd.read_csv(csv_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df = df_raw.loc[df_raw["Name"].isin(TICKERS), ["date", "Name", "close"]].copy()
    df.rename(columns={"Name": "ticker", "close": "price"}, inplace=True)
    prices = df.pivot(index="date", columns="ticker", values="price").sort_index()
    prices = preprocess.clean_prices(prices)
    return prices
