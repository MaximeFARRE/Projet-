# scripts/run_baseline.py
from pathlib import Path
import sys
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.baselines.equal_weight import equity_curve_equal_weight
from src.evaluation.metrics import simple_metrics
import pandas as pd

def main():
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    prices_train, prices_test = split_train_test(prices)

    equity_train = equity_curve_equal_weight(prices_train)
    equity_test = equity_curve_equal_weight(prices_test)

    # courbe
    plt.figure(figsize=(10, 5))
    equity_train.plot(label="Train")
    equity_test.plot(label="Test")
    plt.legend()
    plt.title("Equal-Weight Buy & Hold")
    plt.savefig(REPORT_FIG_DIR / "baseline_equal_weight.png")

    # métriques
    metrics_train = simple_metrics(equity_train)
    metrics_test = simple_metrics(equity_test)

    pd.DataFrame([metrics_train]).to_csv(
        REPORT_TABLE_DIR / "metrics_train_baseline.csv", index=False
    )
    pd.DataFrame([metrics_test]).to_csv(
        REPORT_TABLE_DIR / "metrics_test_baseline.csv", index=False
    )

if __name__ == "__main__":
    main()
    