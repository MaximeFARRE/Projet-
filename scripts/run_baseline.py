# scripts/run_baseline.py

import sys
from pathlib import Path

# Ensure project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.baselines.equal_weight import equity_curve_equal_weight
from src.evaluation.metrics import simple_metrics


def main():
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    prices = load_prices()

    # 2) Train-test split
    prices_train, prices_test = split_train_test(prices)

    # 3) Compute baseline equity curves
    equity_train = equity_curve_equal_weight(prices_train)
    equity_test = equity_curve_equal_weight(prices_test)

    # 4) Metrics
    metrics_train = simple_metrics(equity_train)
    metrics_test = simple_metrics(equity_test)

    # 5) Save metrics
    pd.DataFrame([metrics_train]).to_csv(
        REPORT_TABLE_DIR / "metrics_train_baseline.csv", index=False
    )

    pd.DataFrame([metrics_test]).to_csv(
        REPORT_TABLE_DIR / "metrics_test_baseline.csv", index=False
    )

    # 6) Plot equity curves
    plt.figure(figsize=(10, 5))
    equity_train.plot(label="Train")
    equity_test.plot(label="Test")
    plt.title("Equal-Weight Buy & Hold Baseline")
    plt.legend()
    plt.grid()
    plt.savefig(REPORT_FIG_DIR / "baseline_equal_weight.png")

    print("Baseline generated!")
    print("Train metrics:", metrics_train)
    print("Test metrics:", metrics_test)


if __name__ == "__main__":
    main()
