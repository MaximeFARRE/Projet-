# src/agents/evaluate_ppo.py
from stable_baselines3 import PPO
from src.env.portfolio_env import SimplePortfolioEnv
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.baselines.equal_weight import equity_curve_equal_weight
from src.evaluation.metrics import simple_metrics

def evaluate_ppo():
    prices = load_prices()
    _, prices_test = split_train_test(prices)
    env = SimplePortfolioEnv(prices_test)

    model = PPO.load("models/ppo_portfolio")
    # boucle d'évaluation ici (env.reset(), model.predict, etc.)

if __name__ == "__main__":
    evaluate_ppo()
