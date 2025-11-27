# src/agents/train_ppo.py

import os
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import MODELS_DIR, PPO_MODEL_PATH
from src.data.load_data import load_prices
from src.data.preprocess import split_train_test
from src.features.technical_indicators import compute_technical_features
from src.env.portfolio_env import PortfolioEnv


def make_train_env(prices_train, features_train):
    """
    Create a callable that builds a PortfolioEnv for DummyVecEnv.
    """

    def _init():
        # Align prices with features index
        prices_aligned = prices_train.loc[features_train.index]
        env = PortfolioEnv(
            prices=prices_aligned,
            features=features_train,
            initial_capital=1.0,
            transaction_cost=0.001,  # 0.1% costs per unit turnover
        )
        return env

    return _init


def train_ppo(
    total_timesteps: int = 100_000,
    seed: int = 42,
) -> None:
    """
    Train a PPO agent on the PortfolioEnv using the training data.
    """

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load prices and split
    prices = load_prices()
    prices_train, _ = split_train_test(prices)

    # 2) Compute technical features on train set
    features_train = compute_technical_features(prices_train)

    # 3) Build vectorized environment
    train_env_fn = make_train_env(prices_train, features_train)
    env = DummyVecEnv([train_env_fn])

    # 4) Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
    )

    # 5) Train
    model.learn(total_timesteps=total_timesteps)

    # 6) Save model
    model.save(str(PPO_MODEL_PATH))
    print(f"PPO model saved to: {PPO_MODEL_PATH}.zip")


def main():
    # default training run
    train_ppo(total_timesteps=100_000, seed=42)


if __name__ == "__main__":
    main()
