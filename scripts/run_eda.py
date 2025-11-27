# scripts/run_eda.py

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_DIR, REPORT_FIG_DIR
from src.data.load_data import load_prices
from src.data.preprocess import compute_returns


def main():
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    prices = load_prices()
    returns = compute_returns(prices)

    # ===== FIGURE 1 — Normalized prices =====
    plt.figure(figsize=(10, 6))
    (prices / prices.iloc[0]).plot()
    plt.title("Normalized Prices")
    plt.ylabel("Normalized value (base = 1)")
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "prices_normalized.png")

    # ===== FIGURE 2 — Returns distribution =====
    plt.figure(figsize=(10, 6))
    returns.plot(kind="hist", bins=50, alpha=0.7)
    plt.title("Distribution des rendements")
    plt.xlabel("Daily return")
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "returns_distribution.png")

    # ===== FIGURE 3 — Correlation heatmap =====
    plt.figure(figsize=(8, 6))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation des rendements")
    plt.savefig(REPORT_FIG_DIR / "correlation_heatmap.png")

    # ===== FIGURE 4 — Rolling volatility =====
    rolling_vol = returns.rolling(20).std()

    plt.figure(figsize=(10, 6))
    rolling_vol.plot()
    plt.title("Rolling Volatility (20 days)")
    plt.ylabel("Volatility")
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "rolling_volatility.png")

    print("EDA completed. Figures saved in:", REPORT_FIG_DIR)


if __name__ == "__main__":
    main()
