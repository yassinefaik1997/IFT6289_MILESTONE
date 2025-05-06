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


INITIAL_BALANCE = 1000000
COMMISSION_RATE = 0.001
MAX_SHARES_TRADE = 100  # Example: Max shares to trade per asset per step. Tune this!
REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 128 # Increased batch size often helps
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
NOISE_SCALE = 0.1 # Noise added to actions for exploration
NUM_TRAIN_EPISODES = 500

class PortfolioEnv:
    def __init__(self, price_data, window_obs=1, # window_obs kept for potential features
                 initial_balance=INITIAL_BALANCE,
                 commission_rate=COMMISSION_RATE):

        if not isinstance(price_data, pd.DataFrame):
             raise ValueError("price_data must be a pandas DataFrame")
        # Ensure data is numeric and handle NaNs robustly
        price_data = price_data.apply(pd.to_numeric, errors='coerce')
        if price_data.isnull().values.any():
            # print("Warning: Price data contains NaNs. Filling forward/backward.") # Less verbose
            price_data = price_data.fillna(method='ffill').fillna(method='bfill')
        if price_data.isnull().values.any():
             # print("Warning: NaNs remain after fill. Filling with 0.") # Less verbose
             price_data = price_data.fillna(0) # Fill any remaining NaNs
        if price_data.empty or price_data.shape[0] < 2: # Need at least 2 steps
            raise ValueError("Price data is empty or too short after cleaning.")

        self.price_data = price_data.reset_index(drop=True)
        self.n_assets = price_data.shape[1]
        self.n_steps = len(self.price_data)
        self.asset_names = price_data.columns.tolist()

        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.window_obs = window_obs

        # State dimension: Prices (n_assets) + Shares Held (n_assets) + Balance (1)
        self.state_dim = self.n_assets * 2 + 1
        self.action_dim = self.n_assets # Outputs number of shares to trade

        # Initialize state variables in reset
        self.current_step = 0
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.balance = float(self.initial_balance)
        self.portfolio_value = float(self.initial_balance)
        self.history = []
        self.done = False
        self.reset()

    def reset(self):
        self.current_step = 0
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.balance = float(self.initial_balance)
        self.portfolio_value = float(self.initial_balance)
        self.history = [self.portfolio_value]
        self.done = False

        if self.n_steps <= self.current_step + 1:
             self.done = True
             # print("Warning: Price data too short for environment initialization.") # Less verbose
             return np.zeros(self.state_dim) # Return only state for reset

        return self._get_state() # Return only state for reset

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, self.done, {"error": "Episode already finished"}

        current_prices = self._get_current_prices()
        if current_prices is None:
             self.done = True
             return self._get_state(), 0.0, self.done, {"error": "Reached end of price data"}

        start_portfolio_value = self._calculate_portfolio_value(current_prices)
        action = np.array(action, dtype=np.float32)

        # Clip actions to prevent extreme values if MAX_SHARES_TRADE is defined
        # action = np.clip(action, -MAX_SHARES_TRADE, MAX_SHARES_TRADE)

        sell_indices = np.where(action < 0)[0]; buy_indices = np.where(action > 0)[0]

        # Execute Sells
        for idx in sell_indices:
            shares_to_sell = min(abs(action[idx]), self.shares_held[idx])
            if shares_to_sell > 1e-6:
                 sell_value = shares_to_sell * current_prices[idx]
                 commission = sell_value * self.commission_rate
                 self.balance += sell_value - commission
                 self.shares_held[idx] -= shares_to_sell

        # Execute Buys
        available_balance = self.balance; total_buy_cost_needed = 0; buy_costs = {}
        for idx in buy_indices:
            shares_to_buy = abs(action[idx])
            if shares_to_buy > 1e-6:
                buy_value = shares_to_buy * current_prices[idx]
                commission = buy_value * self.commission_rate; cost = buy_value + commission
                buy_costs[idx] = {'shares': shares_to_buy, 'cost': cost}; total_buy_cost_needed += cost
        scale_factor = (available_balance / total_buy_cost_needed) if total_buy_cost_needed > available_balance > 0 else (0 if available_balance <= 0 else 1.0)
        for idx in buy_indices:
             if idx in buy_costs:
                 scaled_shares = buy_costs[idx]['shares'] * scale_factor; scaled_cost = buy_costs[idx]['cost'] * scale_factor
                 if scaled_shares > 1e-6 and scaled_cost <= self.balance + 1e-6:
                     self.balance -= scaled_cost; self.shares_held[idx] += scaled_shares

        self.balance = max(self.balance, 0.0); self.current_step += 1

        # Check for end of episode
        if self.current_step >= self.n_steps -1:
            self.done = True
            final_prices = self._get_current_prices()
            if final_prices is None: final_prices = self.price_data.iloc[-1].values
            end_portfolio_value = self._calculate_portfolio_value(final_prices)
            reward = end_portfolio_value - start_portfolio_value
            self.history.append(end_portfolio_value)
            return self._get_state(), reward, self.done, {}

        # Calculate reward for non-terminal step
        next_prices = self._get_current_prices()
        if next_prices is None: # Should not happen normally
            self.done = True; end_portfolio_value = start_portfolio_value; reward = 0.0
            # print("Warning: Could not get next prices in step.") # Less verbose
            self.history.append(end_portfolio_value)
            return self._get_state(), reward, self.done, {"error": "Could not get next prices"}

        end_portfolio_value = self._calculate_portfolio_value(next_prices)
        reward = end_portfolio_value - start_portfolio_value # Reward is change in value
        self.portfolio_value = end_portfolio_value; self.history.append(self.portfolio_value)
        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        step = min(self.current_step, self.n_steps - 1)
        current_prices = self.price_data.iloc[step].values.astype(np.float32)
        # Simple normalization (can be improved, e.g., using rolling windows)
        norm_prices = current_prices / (self.price_data.iloc[0].values + 1e-6)
        # Normalize shares by a heuristic: potential max shares if all balance invested at start
        shares_normalization = (self.initial_balance / (self.price_data.iloc[0].values + 1e-6)) + 1.0
        norm_shares = self.shares_held / shares_normalization
        norm_balance = np.array([self.balance / self.initial_balance], dtype=np.float32)
        state = np.concatenate([norm_prices, norm_shares, norm_balance], axis=0).astype(np.float32)
        return state

    def _get_current_prices(self):
        if self.current_step < self.n_steps: return self.price_data.iloc[self.current_step].values.astype(np.float32)
        else: return None

    def _calculate_portfolio_value(self, prices):
         if prices is None or len(prices) != self.n_assets:
             
             prices = self.price_data.iloc[min(self.current_step, self.n_steps - 1)].values
         prices = np.array(prices, dtype=np.float32)
         return np.sum(self.shares_held * prices) + self.balance

    def render(self, title="Portfolio Value Over Time"):
        plt.figure(figsize=(10, 5)); plt.plot(self.history); plt.title(title)
        plt.xlabel("Time Steps"); plt.ylabel("Portfolio Value"); plt.grid(True); plt.show()


train_data = merged_df.loc['2010-01-01':'2016-01-01'].dropna(axis=1, how='all') 
train_data = train_data.fillna(method='ffill').fillna(method='bfill').fillna(0) 
if train_data.empty: raise ValueError("Training data is empty after filtering and cleaning.")
train_log_returns = np.log(train_data / train_data.shift(1)).dropna()
if train_log_returns.empty: raise ValueError("Log returns are empty. Check training data.")
n_assets = train_log_returns.shape[1]
asset_names = train_data.columns.tolist()

window_size = 120
correlation_matrices = []
if len(train_log_returns) >= window_size:
    for start in range(0, len(train_log_returns) - window_size + 1, window_size):
        window_data = train_log_returns.iloc[start:start+window_size]
        corr_matrix = window_data.corr().fillna(0).values
        min_eig = np.min(np.linalg.eigvalsh(corr_matrix))
        if min_eig < 0: corr_matrix -= 1.01 * min_eig * np.eye(corr_matrix.shape[0])
        correlation_matrices.append(corr_matrix)
else:
    print("Warning: Not enough data for rolling correlation. Using overall correlation.")
    corr_matrix = train_log_returns.corr().fillna(0).values
    min_eig = np.min(np.linalg.eigvalsh(corr_matrix))
    if min_eig < 0: corr_matrix -= 1.01 * min_eig * np.eye(corr_matrix.shape[0])
    correlation_matrices = [corr_matrix] # Use single overall correlation

if not correlation_matrices: correlation_matrices = [np.eye(n_assets)] # Fallback

features = [mat[np.triu_indices(n_assets, k=1)] for mat in correlation_matrices]
features = np.array(features)
if features.shape[0] <= 1: # linkage requires > 1 observation
    print("Warning: Only one correlation matrix found. Skipping clustering.")
    rep_corr_matrices = correlation_matrices
else:
    Z = linkage(features, method='ward')
    n_clusters = min(4, features.shape[0]) # Cannot have more clusters than samples
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    rep_corr_matrices = []
    for cluster_id in range(1, n_clusters + 1):
        indices = np.where(clusters == cluster_id)[0]
        if indices.size > 0:
            avg_corr = np.mean([correlation_matrices[i] for i in indices], axis=0)
            min_eig = np.min(np.linalg.eigvalsh(avg_corr))
            if min_eig < 0: avg_corr -= 1.01 * min_eig * np.eye(avg_corr.shape[0])
            rep_corr_matrices.append(avg_corr)
if not rep_corr_matrices: rep_corr_matrices = [np.eye(n_assets)]

annual_factor = 252
annual_drift = train_log_returns.mean() * annual_factor
annual_vol = train_log_returns.std() * np.sqrt(annual_factor)
mu = annual_drift.values; sigmas = annual_vol.values
S0s = train_data.iloc[-1].values
T = 1; dt = 1/252

def simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix):
    n_assets = len(S0s); n_steps = int(T/dt)
    prices = np.zeros((n_steps+1, n_assets)); prices[0] = S0s
    try: L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError: L = np.eye(n_assets)
    for t in range(1, n_steps+1):
        Z = np.random.normal(0, 1, size=n_assets); correlated_Z = L @ Z
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigmas**2)*dt + sigmas*np.sqrt(dt)*correlated_Z)
        prices[t] = np.maximum(prices[t], 1e-6)
    return prices

def simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr_matrix, num_paths=50): # Reduced paths
    sims = [];
    for _ in range(num_paths): sims.append(simulate_correlated_prices(S0s, mu, sigmas, T, dt, corr_matrix))
    return np.array(sims)

all_simulations = {}
num_paths_per_cluster = 50 # Reduced paths
for idx, corr in enumerate(rep_corr_matrices):
    sims = simulate_multiple_paths(S0s, mu, sigmas, T, dt, corr, num_paths=num_paths_per_cluster)
    all_simulations[idx] = sims
if not all_simulations: raise ValueError("No simulations generated.")
combined_simulations = np.concatenate(list(all_simulations.values()), axis=0)
dummy_dates = pd.date_range(start="2018-01-01", periods=combined_simulations.shape[1], freq="B")

def get_simulated_episode():
    episode_idx = np.random.choice(combined_simulations.shape[0])
    df = pd.DataFrame(combined_simulations[episode_idx], index=dummy_dates, columns=asset_names) # Use consistent asset names
    df = df.clip(lower=1e-6)
    return df


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in batch_indices))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


# DDPG Actor and Critic Networks 
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=MAX_SHARES_TRADE):
        super(DDPGActor, self).__init__()
        self.max_action = max_action 
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

#  DDPG Agent Class 

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=MAX_SHARES_TRADE, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                 gamma=GAMMA, tau=TAU, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.actor = DDPGActor(state_dim, action_dim, max_action=self.max_action).to(device)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action=self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for param in self.actor_target.parameters(): param.requires_grad = False


        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters(): param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state, noise_scale=NOISE_SCALE, evaluate=False):
        """ Selects action based on state, adds noise for exploration if not evaluating. """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.actor.eval() 
        with torch.no_grad():
            action = self.actor(state_tensor)
        self.actor.train() 

        action = action.cpu().data.numpy().flatten()

        if not evaluate:
            noise = noise_scale * self.max_action * np.random.randn(len(action))
            action = action + noise
            
        action = np.clip(action, -self.max_action, self.max_action)

        # removed weight normalization : action now represents shares to trade
        return action

    def update(self, replay_buffer, batch_size=BATCH_SIZE):
        if len(replay_buffer) < batch_size:
            return 

        state_np, action_np, reward_np, next_state_np, done_np = replay_buffer.sample(batch_size)
        state_batch = torch.tensor(state_np, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(next_state_np, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_np, dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(reward_np, dtype=torch.float32).unsqueeze(1).to(self.device)
        done_batch = torch.tensor(done_np, dtype=torch.float32).unsqueeze(1).to(self.device) # Ensure dones are float for multiplication

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            # target Q value
            target_q = self.critic_target(next_state_batch, next_actions)
            # Bellman equation
            y = reward_batch + self.gamma * (1.0 - done_batch) * target_q 

        current_q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for param in self.critic.parameters(): param.requires_grad = False
        pred_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, pred_actions).mean()
        for param in self.critic.parameters(): param.requires_grad = True


        # optimizing the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, net, net_target):
        """ Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target """
        for target_param, local_param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        print(f"Agent saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Agent loaded from {filename}")


# Training 

def train_ddpg(agent, num_episodes=NUM_TRAIN_EPISODES, replay_buffer=None, batch_size=BATCH_SIZE, env_fn=get_simulated_episode, device='cpu'):
    if replay_buffer is None:
         replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    episode_rewards = []
    total_steps = 0
    print(f"Starting DDPG training for {num_episodes} episodes...")

    for ep in range(num_episodes):
        sim_data = env_fn()
        if sim_data.empty:
            print(f"Warning: Skipping episode {ep+1} due to empty simulation data.")
            continue
        env = PortfolioEnv(sim_data) 
        state = env.reset() 
        ep_reward = 0
        ep_steps = 0

        while not env.done:
            action = agent.select_action(state, evaluate=False) 
            next_state, reward, done, _ = env.step(action) 
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            # agent update
            agent.update(replay_buffer, batch_size)


        episode_rewards.append(ep_reward)
        if (ep+1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Episode {ep+1}/{num_episodes} | Steps: {ep_steps} | Reward: {ep_reward:.2f} | Avg Reward (last 10): {avg_reward:.2f}")


    print(f"Training finished. Total steps: {total_steps}")
    return episode_rewards

def evaluate_agent(agent, historical_data, start_date, end_date, device='cpu'):
    test_df = historical_data.loc[start_date:end_date]
    if test_df.empty:
        print(f"Warning: No historical data for evaluation period {start_date}-{end_date}")
        return INITIAL_BALANCE, [INITIAL_BALANCE] 

    env = PortfolioEnv(test_df) 
    state = env.reset() 
    if env.done:
        print("Warning: Evaluation environment started in 'done' state.")
        return INITIAL_BALANCE, [INITIAL_BALANCE]

    while not env.done:
        action = agent.select_action(state, evaluate=True) 
        state, _, done, _ = env.step(action)

    final_value = env.portfolio_value
    return final_value, env.history


# Execution

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("Initializing dummy environment for dimensions...")
        dummy_sim_data = get_simulated_episode()
        if dummy_sim_data.empty: raise ValueError("Failed dummy sim data gen.")
        dummy_env = PortfolioEnv(dummy_sim_data)
        state_dim = dummy_env.state_dim
        action_dim = dummy_env.action_dim
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    except Exception as e:
        print(f"Error initializing dummy environment: {e}")
        
        try:
             merged_df_global = load_stock_data()
             n_assets_global = merged_df_global.shape[1]
             state_dim = n_assets_global * 2 + 1
             action_dim = n_assets_global
             print(f"Using fallback dimensions from loaded data: State={state_dim}, Action={action_dim}")
        except Exception as e2:
             print(f"Fallback failed: Could not load data to get dimensions: {e2}")
                
             state_dim = 61 
             action_dim = 30 
             print(f"Using arbitrary fallback dimensions: State={state_dim}, Action={action_dim}")


    print("Initializing DDPG agent...")
    ddpg_agent = DDPGAgent(state_dim, action_dim, device=device)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    # Train on simulated episodes
    print("Starting DDPG training on simulated data...")
    train_rewards = train_ddpg(ddpg_agent, env_fn=get_simulated_episode, replay_buffer=replay_buffer, device=device)

