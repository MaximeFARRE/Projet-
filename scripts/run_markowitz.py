# scripts/run_markowitz.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test, compute_returns
from src.baselines.markowitz import (
    compute_min_variance_weights,
    equity_curve_markowitz,
)
from src.evaluation.metrics import simple_metrics


def main():

    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    prices = load_prices()
    prices_train, prices_test = split_train_test(prices)

    # 2) Compute returns (train only)
    returns_train = compute_returns(prices_train)

    # 3) Compute MVP weights
    weights_mvp = compute_min_variance_weights(returns_train)
    print("Optimal Markowitz weights:", weights_mvp)

    # 4) Compute equity curves (train & test)
    equity_train = equity_curve_markowitz(prices_train, weights_mvp)
    equity_test = equity_curve_markowitz(prices_test, weights_mvp)

    # 5) Metrics
    metrics_train = simple_metrics(equity_train)
    metrics_test = simple_metrics(equity_test)

    # 6) Save metrics
    pd.DataFrame([metrics_train]).to_csv(
        REPORT_TABLE_DIR / "metrics_train_markowitz.csv", index=False
    )
    pd.DataFrame([metrics_test]).to_csv(
        REPORT_TABLE_DIR / "metrics_test_markowitz.csv", index=False
    )

    # 7) Plot
    plt.figure(figsize=(10, 5))
    equity_train.plot(label="Train")
    equity_test.plot(label="Test")
    plt.title("Markowitz Minimum Variance Portfolio")
    plt.legend()
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "markowitz_min_variance.png")

    print("Train metrics:", metrics_train)
    print("Test metrics:", metrics_test)
    print("Figure saved.")


if __name__ == "__main__":
    main()
