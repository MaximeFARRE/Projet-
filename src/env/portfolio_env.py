# src/env/portfolio_env.py
import numpy as np
import pandas as pd

class SimplePortfolioEnv:
    """
    Minimal environment for RL portfolio allocation.
    """

    def __init__(self, prices: pd.DataFrame, window: int = 20):
        self.prices = prices.copy()
        self.window = window
        self.n_assets = prices.shape[1]
        self.t = window
        self.reset()

    def reset(self):
        self.t = self.window
        obs_window = self.prices.iloc[self.t - self.window:self.t]
        obs = (obs_window / obs_window.iloc[0]).values
        return obs

    def step(self, weights):
        weights = np.asarray(weights)
        weights = weights / weights.sum()

        day_ret = self.prices.pct_change().iloc[self.t]
        port_ret = float(np.dot(weights, day_ret.values))

        self.t += 1
        done = self.t >= len(self.prices)

        if not done:
            obs_window = self.prices.iloc[self.t - self.window:self.t]
            obs = (obs_window / obs_window.iloc[0]).values
        else:
            obs = None

        reward = port_ret
        info = {"portfolio_return": port_ret}
        return obs, reward, done, info
