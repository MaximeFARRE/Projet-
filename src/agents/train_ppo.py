# src/agents/train_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.env.portfolio_env import SimplePortfolioEnv
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test

def train_ppo():
    prices = load_prices()
    prices_train, _ = split_train_test(prices)
    env = SimplePortfolioEnv(prices_train)

    check_env(env, warn=True)  # quand il sera au format Gym

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("models/ppo_portfolio")

if __name__ == "__main__":
    train_ppo()
