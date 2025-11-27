# scripts/run_baseline.py

import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORT_FIG_DIR, REPORT_TABLE_DIR
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.baselines.equal_weight import equity_curve_equal_weight
from src.evaluation.metrics import simple_metrics

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Création des dossiers de sortie
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Charger les prix
    prices = load_prices()

    # 2) Split Train / Test
    prices_train, prices_test = split_train_test(prices)

    # 3) Calculer la stratégie Buy & Hold Equal Weight
    equity_train = equity_curve_equal_weight(prices_train)
    equity_test = equity_curve_equal_weight(prices_test)

    # 4) Calculer les métriques
    metrics_train = simple_metrics(equity_train)
    metrics_test = simple_metrics(equity_test)

    # 5) Sauvegarder les métriques
    pd.DataFrame([metrics_train]).to_csv(
        REPORT_TABLE_DIR / "metrics_train_baseline.csv",
        index=False
    )

    pd.DataFrame([metrics_test]).to_csv(
        REPORT_TABLE_DIR / "metrics_test_baseline.csv",
        index=False
    )

    # 6) Graphique equity curve
    plt.figure(figsize=(10, 5))
    equity_train.plot(label="Train")
    equity_test.plot(label="Test")
    plt.legend()
    plt.title("Equal-Weight Buy & Hold")
    plt.savefig(REPORT_FIG_DIR / "baseline_equal_weight.png")

    # Optionnel : afficher aussi le graph en direct
    # plt.show()

    print("\n=== BASELINE GENERATED ===")
    print("Train metrics:", metrics_train)
    print("Test metrics:", metrics_test)
    print("Figures saved in:", REPORT_FIG_DIR)
    print("Tables saved in:", REPORT_TABLE_DIR)


if __name__ == "__main__":
    main()
