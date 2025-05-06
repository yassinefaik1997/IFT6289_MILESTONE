# IFT6289_MILESTONE

This is an implementation that combines lexicon-based market sentiment with deep reinforcement learning ensembles for automated stock trading. The repo includes:

load_data.py: loads historical price CSVs and generates simulated episodes via Monte Carlo and Black-Scholes;
sentiment_calculations.py: parses your headlines CSV and computes period-by-period sentiment using the AFINN-165 lexicon;
PortfolioEnv.py: defines the Gym-style multi-asset trading environment with transaction costs;
PPO_agent.py, DDPG_agent.py, and A2C_agent.py: each builds its respective agent’s network architecture, optimizer, training loop, and checkpointing;
chi_score.py: implements the Chi metric as in the paper, combining return and risk;
traditional_ensemble.py: runs the fixed-interval (every N days) agent-switching baseline;
Sentiment_dynamic.py: performs real-time switching whenever sentiment shifts beyond a threshold, selecting the highest Sharpe-Sortino agent;
individual_agents.py: backtests each standalone A2C, DDPG, and PPO agent and records their P&L histories;
metrics_calculation.py: aggregates all P&L trajectories into performance tables and plots (cumulative return, annualized return, volatility, Sharpe, Sortino, Calmar, etc.).
