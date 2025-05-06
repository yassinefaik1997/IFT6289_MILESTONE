#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster 

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

# Compute log returns for correlation analysis
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

T = 1       # 1 year simulation
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
num_paths = 1000
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


# Configuration 

INITIAL_BALANCE = 1_000_000
COMMISSION_RATE = 0.001
MAX_SHARES_TRADE = 100
PPO_LR = 3e-4         
GAMMA = 0.99           
EPS_CLIP = 0.2         
K_EPOCHS = 10          
ENTROPY_COEF = 0.01    
VALUE_COEF = 0.5       

# Training Loop Hyperparameters
NUM_UPDATES = 250
TRAJECTORIES_PER_UPDATE = 10


class PortfolioEnv:
    """
    Environment for portfolio management.
    Expects a pandas DataFrame of prices (rows: time, cols: assets).
    """
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

        # State: [normalized prices, normalized holdings, normalized balance]
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

        if self.n_steps <= self.current_step + 1:
            self.done = True
            return np.zeros(self.state_dim, dtype=np.float32)

        return self._get_state()

    def step(self, action: np.ndarray):
        if self.done:
            return self._get_state(), 0.0, True, {"error": "Episode finished"}

        current_prices = self._get_current_prices()
        if current_prices is None:
            self.done = True
            return self._get_state(), 0.0, True, {"error": "End of price data"}

        start_value = self._calculate_portfolio_value(current_prices)
        action = np.array(action, dtype=np.float32)

        # Sells
        sell_indices = np.where(action < 0)[0]
        for idx in sell_indices:
            shares_to_sell = min(abs(action[idx]), self.shares_held[idx])
            if shares_to_sell > 1e-6:
                sell_value = shares_to_sell * current_prices[idx]
                commission = sell_value * self.commission_rate
                self.balance += sell_value - commission
                self.shares_held[idx] -= shares_to_sell

        # Buys
        buy_indices = np.where(action > 0)[0]
        available_balance = self.balance
        buy_costs = {}
        total_cost = 0.0

        for idx in buy_indices:
            shares_to_buy = abs(action[idx])
            if shares_to_buy > 1e-6:
                buy_value = shares_to_buy * current_prices[idx]
                commission = buy_value * self.commission_rate
                cost = buy_value + commission
                buy_costs[idx] = {'shares': shares_to_buy, 'cost': cost}
                total_cost += cost

        if total_cost > 0 and available_balance < total_cost:
            scale = available_balance / total_cost
        elif available_balance <= 0:
            scale = 0
        else:
            scale = 1.0

        for idx, info in buy_costs.items():
            scaled_shares = info['shares'] * scale
            scaled_cost = info['cost'] * scale
            if scaled_shares > 1e-6 and scaled_cost <= self.balance + 1e-6:
                self.balance -= scaled_cost
                self.shares_held[idx] += scaled_shares

        self.balance = max(self.balance, 0.0)
        self.current_step += 1

        # Terminal
        if self.current_step >= self.n_steps - 1:
            self.done = True
            final_prices = self._get_current_prices()
            if final_prices is None:
                self.price_data.iloc[-1].values
            end_value = self._calculate_portfolio_value(final_prices)
            reward = end_value - start_value
            self.history.append(end_value)
            return self._get_state(), reward, True, {}

        # Non-terminal
        next_prices = self._get_current_prices()
        if next_prices is None:
            self.done = True
            end_value = start_value
            reward = 0.0
            self.history.append(end_value)
            return self._get_state(), reward, True, {"error": "No next prices"}

        end_value = self._calculate_portfolio_value(next_prices)
        reward = end_value - start_value
        self.portfolio_value = end_value
        self.history.append(end_value)

        return self._get_state(), reward, False, {}

    def _get_state(self) -> np.ndarray:
        step = min(self.current_step, self.n_steps - 1)
        prices = self.price_data.iloc[step].values.astype(np.float32)

        norm_prices = prices / (self.price_data.iloc[0].values + 1e-6)
        shares_norm = (
            (self.initial_balance / self.n_assets)
            / (self.price_data.iloc[0].values + 1e-6)
            + 1
        )
        norm_shares = self.shares_held / (shares_norm + 1e-6)
        norm_balance = np.array([self.balance / self.initial_balance], dtype=np.float32)

        state = np.concatenate([norm_prices, norm_shares, norm_balance], axis=0)
        return state.astype(np.float32)

    def _get_current_prices(self) -> np.ndarray:
        if self.current_step < self.n_steps:
            return self.price_data.iloc[self.current_step].values.astype(np.float32)
        return None

    def _calculate_portfolio_value(self, prices) -> float:
        if prices is None or len(prices) != self.n_assets:
            prices = self.price_data.iloc[min(self.current_step, self.n_steps - 1)].values

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


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Initialize log_std near zero
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        value = self.value_head(x)

        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std)

        return mean, std, value


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = PPO_LR,
        gamma: float = GAMMA,
        eps_clip: float = EPS_CLIP,
        K_epochs: int = K_EPOCHS,
        value_coef: float = VALUE_COEF,
        entropy_coef: float = ENTROPY_COEF,
        device: str = 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.policy = PPOActorCritic(state_dim, action_dim).to(device)
        self.policy_old = PPOActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        for p in self.policy_old.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)

    def select_action(self, state, evaluate: bool = False):
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32)
            tensor = tensor.unsqueeze(0).to(self.device)

            mean, std, value = self.policy_old(tensor)
            dist = Normal(mean, std)
            action = mean if evaluate else dist.sample()

            logprob = dist.log_prob(action).sum(dim=-1)
            return action.cpu().numpy().flatten(), logprob.item(), value.item()

    def update(self, memory: list):
        if not memory:
            return

        states = torch.tensor(
            np.stack([m['state'] for m in memory]),
            dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(
            np.stack([m['action'] for m in memory]),
            dtype=torch.float32
        ).to(self.device)

        old_logprobs = torch.tensor(
            [m['logprob'] for m in memory],
            dtype=torch.float32
        ).to(self.device)

        rewards = torch.tensor(
            [m['reward'] for m in memory],
            dtype=torch.float32
        ).to(self.device)

        dones = torch.tensor(
            [m['done'] for m in memory],
            dtype=torch.float32
        ).to(self.device)

        old_values = torch.tensor(
            [m['value'] for m in memory],
            dtype=torch.float32
        ).to(self.device)

        # returns and advantages
        returns = []
        discounted = 0.0
        for r, d in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
            discounted = r + self.gamma * discounted * (1.0 - d)
            returns.insert(0, discounted)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            mean, std, vals = self.policy(states)
            dist = Normal(mean, std)

            new_logprobs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            s1 = ratios * advantages
            s2 = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantages

            pg_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(vals.squeeze(-1), returns)
            loss = pg_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Sync old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear()

# Training 

def collect_rollout(agent: PPOAgent, env: PortfolioEnv, memory: list):
    state = env.reset()
    done = False
    ep_reward = 0.0
    steps = 0
    max_steps = len(env.price_data) - 2

    while not done and steps < max_steps:
        action, logp, val = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        memory.append({
            'state': state,
            'action': action,
            'logprob': logp,
            'value': val,
            'reward': reward,
            'done': float(done)
        })

        state = next_state
        ep_reward += reward
        steps += 1

    return ep_reward, steps


def train_ppo(
    agent: PPOAgent,
    num_updates: int = NUM_UPDATES,
    trajectories_per_update: int = TRAJECTORIES_PER_UPDATE,
    env_fn=get_simulated_episode,
    device: str = 'cpu'
):
    all_rewards = []
    memory = []

    print(f"Starting PPO training for {num_updates} updates...")

    for u in range(num_updates):
        rewards_batch = []
        steps_batch = 0
        memory.clear()

        for _ in range(trajectories_per_update):
            sim_data = env_fn()
            if sim_data.empty:
                continue

            env = PortfolioEnv(sim_data)
            if env.done:
                continue

            r, s = collect_rollout(agent, env, memory)
            rewards_batch.append(r)
            steps_batch += s

        if memory:
            agent.update(memory)
            avg_r = np.mean(rewards_batch) if rewards_batch else 0.0
            print(f"Update {u+1}/{num_updates} | Avg Reward: {avg_r:.2f} | Steps: {steps_batch}")
            all_rewards.extend(rewards_batch)
        else:
            print(f"Update {u+1}/{num_updates}: No data collected, skipping.")

    print("Training finished.")
    return all_rewards


def evaluate_agent_ppo(
    agent: PPOAgent,
    historical_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    device: str = 'cpu'
):
    test_df = historical_data.loc[start_date:end_date]
    if test_df.empty:
        return INITIAL_BALANCE, [INITIAL_BALANCE]

    env = PortfolioEnv(test_df)
    state = env.reset()
    if env.done:
        return env.initial_balance, [env.initial_balance]

    while not env.done:
        action, _, _ = agent.select_action(state, evaluate=True)
        state, _, done, _ = env.step(action)

    return env.portfolio_value, env.history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine dims via dummy
    try:
        dummy = get_simulated_episode()
        if dummy.empty:
            raise ValueError
        env = PortfolioEnv(dummy)
        state_dim = env.state_dim
        action_dim = env.action_dim
        print(f"State dim: {state_dim}, Action dim: {action_dim}")
    except Exception:
        print("Using fallback dimensions.")
        n_assets_global = merged_df.shape[1]
        state_dim = n_assets_global * 2 + 1
        action_dim = n_assets_global

    agent = PPOAgent(state_dim, action_dim, device=device)

    rewards = train_ppo(agent, env_fn=get_simulated_episode, device=device)

