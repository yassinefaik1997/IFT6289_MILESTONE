#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

