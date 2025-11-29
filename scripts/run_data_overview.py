# scripts/run_data_overview.py

import sys
from pathlib import Path

# Make sure Python can import from the project root (src package)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, TICKERS
from src.data.load_data import load_prices
from src.data.preprocess import compute_returns


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load clean prices
    prices = load_prices()
    print("Prices loaded")
    print("Shape:", prices.shape)
    print("Date range:", prices.index.min().date(), "->", prices.index.max().date())
    print("Tickers:", list(prices.columns))

    # 2) Compute returns
    returns = compute_returns(prices)
    print("\nReturns computed")
    print("Shape:", returns.shape)

    # 3) Save to disk
    prices.to_csv(PROCESSED_DIR / "prices_clean.csv")
    returns.to_csv(PROCESSED_DIR / "returns_clean.csv")

    print("\nSaved:")
    print(" -", PROCESSED_DIR / "prices_clean.csv")
    print(" -", PROCESSED_DIR / "returns_clean.csv")


if __name__ == "__main__":
    main()
