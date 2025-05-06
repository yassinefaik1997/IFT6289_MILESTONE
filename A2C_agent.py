#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple, Dict, Optional


# Configurations 

INITIAL_BALANCE = 1_000_000
COMMISSION_RATE = 0.001
A2C_LR = 7e-4
GAMMA = 0.99
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
RMSPROP_ALPHA = 0.99
RMSPROP_EPS = 1e-5
NUM_EPISODES = 650

# Data Load


def load_stock_data():
    folder_path = os.path.join("data_6289", "*.csv")
    csv_files = glob.glob(folder_path)
    data_frames = []
    for file in csv_files:
        symbol = os.path.splitext(os.path.basename(file))[0]
        print(f"Processing {symbol} from {file}...")
        df = pd.read_csv(file, parse_dates=['date'])
        mask = (df['date'] >= '2010-01-01') & (df['date'] <= '2023-01-01')
        df = df.loc[mask]
        df = df[['date', 'adj close']].set_index('date')
        df.rename(columns={'adj close': symbol}, inplace=True)
        data_frames.append(df)
    merged_df = pd.concat(data_frames, axis=1)
    merged_df.sort_index(inplace=True)
    return merged_df

merged_df = load_stock_data()
train_data = merged_df.loc['2010-01-01':'2016-01-01']

train_log_returns = np.log(train_data / train_data.shift(1)).dropna()
window_size = 120
correlation_matrices = []
for start in range(0, len(train_log_returns) - window_size + 1, window_size):
    window_data = train_log_returns.iloc[start:start+window_size]
    corr_matrix = window_data.corr().values
    correlation_matrices.append(corr_matrix)

n_assets = train_log_returns.shape[1]
features = [mat[np.triu_indices(n_assets, k=1)] for mat in correlation_matrices]
features = np.array(features)

Z = linkage(features, method='ward')
n_clusters = 4
clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

rep_corr_matrices = []
for cluster_id in range(1, n_clusters + 1):
    indices = np.where(clusters == cluster_id)[0]
    avg_corr = np.mean([correlation_matrices[i] for i in indices], axis=0)
    rep_corr_matrices.append(avg_corr)

annual_factor = 252
annual_drift = train_log_returns.mean() * annual_factor
annual_vol = train_log_returns.std() * np.sqrt(annual_factor)
mu = annual_drift.values
sigmas = annual_vol.values
S0s = train_data.iloc[-1].values

T = 1
dt = 1 / 252

def simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix):
    n_assets = len(S0s)
    n_steps = int(T / dt)
    prices = np.zeros((n_steps + 1, n_assets))
    prices[0] = S0s
    L = np.linalg.cholesky(corr_matrix)
    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, size=n_assets)
        correlated_Z = L @ Z
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * correlated_Z)
    return prices

def simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr_matrix, num_paths=100):
    sims = []
    for _ in range(num_paths):
        sim = simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix)
        sims.append(sim)
    return np.array(sims)

all_simulations = {}
num_paths = 100
for idx, corr in enumerate(rep_corr_matrices):
    sims = simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr, num_paths=num_paths)
    all_simulations[idx] = sims

combined_simulations = np.concatenate(list(all_simulations.values()), axis=0)
dummy_dates = pd.date_range(start="2018-01-01", periods=combined_simulations.shape[1], freq="B")
asset_names = [f"Stock{i+1}" for i in range(combined_simulations.shape[2])]

def get_simulated_episode():
    episode_idx = np.random.choice(combined_simulations.shape[0])
    return pd.DataFrame(combined_simulations[episode_idx],
                        index=dummy_dates, columns=asset_names)


# Portfolio Environment 

class PortfolioEnv:
    def __init__(
        self,
        price_data: pd.DataFrame,
        window_obs: int = 1,
        initial_balance: float = INITIAL_BALANCE,
        commission_rate: float = COMMISSION_RATE
    ):
        if not isinstance(price_data, pd.DataFrame):
            raise ValueError("price_data must be a pandas DataFrame")

        price_data = price_data.apply(pd.to_numeric, errors='coerce')
        price_data = price_data.fillna(method='ffill').fillna(method='bfill')
        price_data = price_data.fillna(0)

        if price_data.empty or price_data.shape[0] < 2:
            raise ValueError("Price data is empty or too short after cleaning.")

        self.price_data = price_data.reset_index(drop=True)
        self.n_assets = price_data.shape[1]
        self.n_steps = len(self.price_data)
        self.asset_names = price_data.columns.tolist()

        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.window_obs = window_obs

        self.state_dim = self.n_assets * 2 + 1
        self.action_dim = self.n_assets

        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.balance = float(self.initial_balance)
        self.portfolio_value = float(self.initial_balance)
        self.history = [self.portfolio_value]
        self.done = False

        if self.n_steps <= 1:
            self.done = True
            return np.zeros(self.state_dim, dtype=np.float32)

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self._get_state(), 0.0, True, {"error": "Episode finished"}

        current_prices = self._get_current_prices()
        if current_prices is None:
            self.done = True
            return self._get_state(), 0.0, True, {"error": "Stepped beyond data end"}

        start_value = self._calculate_portfolio_value(current_prices)
        action = np.array(action, dtype=np.float32)

        sell_indices = np.where(action < 0)[0]
        for idx in sell_indices:
            shares_to_sell = min(abs(action[idx]), self.shares_held[idx])
            if shares_to_sell > 1e-6:
                sell_value = shares_to_sell * current_prices[idx]
                commission = sell_value * self.commission_rate
                self.balance += sell_value - commission
                self.shares_held[idx] -= shares_to_sell

        buy_indices = np.where(action > 0)[0]
        buy_costs = {}
        total_buy_cost_needed = 0.0
        for idx in buy_indices:
            shares_to_buy = abs(action[idx])
            if shares_to_buy > 1e-6:
                buy_value = shares_to_buy * current_prices[idx]
                commission = buy_value * self.commission_rate
                cost = buy_value + commission
                if cost > 1e-9:
                    buy_costs[idx] = {'shares': shares_to_buy, 'cost': cost}
                    total_buy_cost_needed += cost

        available_balance = self.balance
        scale_factor = 1.0
        if total_buy_cost_needed > available_balance:
            scale_factor = available_balance / total_buy_cost_needed if total_buy_cost_needed > 1e-9 else 0.0
        if available_balance <= 1e-9:
            scale_factor = 0.0

        for idx in buy_indices:
            if idx in buy_costs:
                scaled_shares = buy_costs[idx]['shares'] * scale_factor
                scaled_cost = buy_costs[idx]['cost'] * scale_factor
                if scaled_shares > 1e-6 and scaled_cost <= self.balance + 1e-6:
                    self.balance -= scaled_cost
                    self.shares_held[idx] += scaled_shares

        self.balance = max(self.balance, 0.0)

        self.current_step += 1

        if self.current_step >= self.n_steps - 1:
            self.done = True
            final_prices = self._get_current_prices()
            if final_prices is None:
                final_prices = self.price_data.iloc[-1].values
            end_value = self._calculate_portfolio_value(final_prices)
            reward = end_value - start_value
            self.history.append(end_value)
            return self._get_state(), reward, True, {}

        next_prices = self._get_current_prices()
        if next_prices is None:
            self.done = True
            reward = 0.0
            self.history.append(start_value)
            return self._get_state(), reward, True, {"error": "No next prices unexpectedly"}

        end_value = self._calculate_portfolio_value(next_prices)
        reward = end_value - start_value
        self.portfolio_value = end_value
        self.history.append(end_value)

        return self._get_state(), reward, False, {}

    def _get_state(self) -> np.ndarray:
        step = min(self.current_step, self.n_steps - 1)
        current_prices = self.price_data.iloc[step].values.astype(np.float32)

        norm_prices = current_prices / (self.price_data.iloc[0].values + 1e-6)
        shares_norm_factor = (self.initial_balance / self.n_assets) / (self.price_data.iloc[0].values + 1e-6) + 1.0
        norm_shares = self.shares_held / (shares_norm_factor + 1e-6)
        norm_balance = np.array([self.balance / self.initial_balance], dtype=np.float32)

        state = np.concatenate([norm_prices, norm_shares, norm_balance], axis=0)
        return state.astype(np.float32)

    def _get_current_prices(self) -> Optional[np.ndarray]:
        if self.current_step < self.n_steps:
            return self.price_data.iloc[self.current_step].values.astype(np.float32)
        return None

    def _calculate_portfolio_value(self, prices: np.ndarray) -> float:
        prices = np.array(prices, dtype=np.float32)
        return float(np.sum(self.shares_held * prices) + self.balance)

    def render(self, title: str = "Portfolio Value Over Time") -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.history)
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.show()


#  A2C Network and Agent

class A2CNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        value = self.value_head(x).squeeze(-1)
        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std)
        return mean, std, value

class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = A2C_LR,
        gamma: float = GAMMA,
        entropy_coef: float = ENTROPY_COEF,
        value_coef: float = VALUE_COEF,
        device: str = 'cpu',
        optimizer_alpha: float = RMSPROP_ALPHA,
        optimizer_eps: float = RMSPROP_EPS
    ):
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.network = A2CNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.RMSprop(
            self.network.parameters(), lr=lr, alpha=optimizer_alpha, eps=optimizer_eps
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[np.ndarray, float, float]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.network(state_tensor)
        dist = Normal(mean, std)
        action = mean if evaluate else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy().flatten(), log_prob.item(), value.item()

    def update(self, trajectory: List[Dict]):
        if not trajectory:
            return
        states = torch.tensor(np.stack([t['state'] for t in trajectory]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack([t['action'] for t in trajectory]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t['reward'] for t in trajectory], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t['done'] for t in trajectory], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t['next_state'] for t in trajectory]), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, _, last_value = self.network(next_states[-1].unsqueeze(0))

        returns = torch.zeros_like(rewards)
        R = last_value if not dones[-1] else torch.tensor(0.0, device=self.device)
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1.0 - dones[t])
            returns[t] = R

        means, stds, values = self.network(states)
        dist = Normal(means, stds)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.smooth_l1_loss(values, returns)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"A2C Agent saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"A2C Agent loaded from {filename}")


#  A2C Training 



def collect_trajectory_a2c(agent: A2CAgent, env: PortfolioEnv) -> Tuple[List[Dict], float]:
    trajectory = []
    state = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0
    max_steps = len(env.price_data) - 2
    while not done and steps < max_steps:
        action, logp, val = agent.select_action(state, evaluate=False)
        next_state, reward, done, _ = env.step(action)
        trajectory.append({
            'state': state,
            'action': action,
            'log_prob': logp,
            'value': val,
            'reward': reward,
            'next_state': next_state,
            'done': float(done)
        })
        state = next_state
        ep_reward += reward
        steps += 1
    return trajectory, ep_reward

def train_a2c(
    agent: A2CAgent,
    num_episodes: int = NUM_EPISODES,
    env_fn = get_simulated_episode,
    device: str = 'cpu'
) -> List[float]:
    all_rewards = []
    print(f"Starting A2C training for {num_episodes} episodes...")
    for ep in range(num_episodes):
        sim_data = env_fn()
        if sim_data.empty:
            print(f"Warning: Skipping ep {ep+1}, empty sim data.")
            continue
        env = PortfolioEnv(sim_data)
        if env.done:
            print(f"Warning: Skipping ep {ep+1}, env started done.")
            continue

        trajectory, ep_reward = collect_trajectory_a2c(agent, env)
        if trajectory:
            agent.update(trajectory)

        all_rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {ep+1}/{num_episodes} | Reward={ep_reward:.2f} | Avg(100)={avg_reward:.2f}")
    print("Training finished.")
    return all_rewards

def evaluate_agent_a2c(
    agent: A2CAgent,
    historical_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    device: str = 'cpu'
) -> Tuple[float, List[float]]:
    print(f"Evaluating A2C from {start_date} to {end_date}...")
    test_df = historical_data.loc[start_date:end_date].copy()
    if test_df.empty:
        print("Warning: No data in evaluation period.")
        return INITIAL_BALANCE, [INITIAL_BALANCE]

    env = PortfolioEnv(test_df)
    state = env.reset()
    if env.done:
        print("Warning: Eval env started done.")
        return env.initial_balance, [env.initial_balance]

    while not env.done:
        action, _, _ = agent.select_action(state, evaluate=True)
        state, _, done, _ = env.step(action)
    print(f"Evaluation complete. Final Value: {env.portfolio_value:.2f}")
    return env.portfolio_value, env.history


# Main

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dummy_env = PortfolioEnv(get_simulated_episode())
        state_dim = dummy_env.state_dim
        action_dim = dummy_env.action_dim
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
    except Exception as e:
        print(f"Error getting dims from dummy env: {e}. Using fallback.")
        state_dim = 2 * n_assets + 1
        action_dim = n_assets
        print(f"Fallback dims: State={state_dim}, Action={action_dim}")

    agent = A2CAgent(state_dim, action_dim, device=device)

    a2c_rewards = train_a2c(agent, env_fn=get_simulated_episode, device=device)

