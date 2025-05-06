#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class BaseAgent:
    def __init__(self, state_dim, action_dim, device='cpu'): self.action_dim = action_dim
    def load(self, path): print(f"Dummy load {path}"); self.eval()
    def select_action(self, state, evaluate=True): return np.zeros(self.action_dim)
    def eval(self): pass
class DDPGAgent(BaseAgent): pass
class PPOAgent(BaseAgent): pass
class A2CAgent(BaseAgent): pass
def calculate_chi_score(vals, alpha=0.25, **kwargs): return np.random.rand() 
class PortfolioEnvHistorical:
     def __init__(self, data, **kwargs):
         self.n_steps = len(data)
         self.state_dim = 10 
         self.action_dim = 3 
         self.current_step = 0
         self.portfolio_value = 1_000_000
         self.history = [self.portfolio_value]
         self.done = False
     def reset(self): self.current_step = 0; self.done = False; self.history = [1_000_000]; return np.zeros(self.state_dim)
     def step(self, action):
         if self.current_step >= self.n_steps - 1: self.done = True
         reward = np.random.randn() * 100
         self.portfolio_value += reward
         self.history.append(self.portfolio_value)
         self.current_step += 1
         return np.zeros(self.state_dim), reward, self.done, {}
     def _get_state(self): return np.zeros(self.state_dim) 


# parameters
ALPHA = 0.25           # for Chi score calculation
BETA = 2    # sentiment change threshold for switching
SENTIMENT_PERIOD = 62  # days for sentiment evaluation period
DEVICE = 'cpu'         # 'cuda' if available

agent_paths = {
    "DDPG": "trained_models/ddpg_sim_trained.pth",
    "PPO": "trained_models/ppo_sim_trained.pth",
    "A2C": "trained_models/a2c_sim_trained.pth"
}

save_directory = "trained_models"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
for p in agent_paths.values():
     if not os.path.exists(p):
         with open(p, 'w') as f: f.write('dummy model data')


def run_dynamic_backtest(evaluation_data: pd.DataFrame,
                         agent_classes: Dict[str, Type[BaseAgent]],
                         agent_paths: Dict[str, str],
                         validation_metric_fn: callable,
                         env_class: Type[PortfolioEnvHistorical],
                         alpha: float = ALPHA,
                         beta: float = BETA,
                         period: int = SENTIMENT_PERIOD,
                         device: str = DEVICE) -> Optional[pd.DataFrame]:
    """
    Runs the backtest with dynamic agent switching based on sentiment.

    Args:
        evaluation_data (pd.DataFrame): Prepared historical data with prices, returns,
                                         'Sentiment', 'PeriodSentiment'. Index must be DatetimeIndex.
                                         Stock features must have MultiIndex columns.
        agent_classes (dict): Dictionary mapping agent names (e.g., "DDPG") to their classes.
        agent_paths (dict): Dictionary mapping agent names to saved model file paths.
        validation_metric_fn (callable): Function like calculate_chi_score(portfolio_values, alpha=...).
        env_class (class): The PortfolioEnvHistorical class.
        alpha (float): Weight for Sharpe in Chi score.
        beta (float): Sentiment change threshold.
        period (int): Length of sentiment evaluation period.
        device (str): Device ('cpu' or 'cuda').

    Returns:
        pd.DataFrame: DataFrame with portfolio history and active agent info, indexed by Date.
                      Returns None if errors occur during setup.
    """

    print("--- Starting Dynamic Backtest ---")

    # initialization
    print("Initializing environment...")
    required_cols = ['Sentiment', 'PeriodSentiment']
    if not all(col in evaluation_data.columns for col in required_cols):
         missing = [col for col in required_cols if col not in evaluation_data.columns]
         raise ValueError(f"Evaluation data missing required columns: {missing}")
    if not isinstance(evaluation_data.index, pd.DatetimeIndex):
        raise ValueError("Evaluation data must have a DatetimeIndex.")


    try:
        hist_env = env_class(evaluation_data) # full data
        state_dim = hist_env.state_dim
        action_dim = hist_env.action_dim
    except Exception as e:
        print(f"Error initializing PortfolioEnvHistorical: {e}")
        import traceback
        traceback.print_exc() 
        return None

    print(f"Environment initialized. State Dim: {state_dim}, Action Dim: {action_dim}")

    # load agents
    agents = {}
    print("Loading trained agents...")
    for name, agent_class in agent_classes.items():
        if name not in agent_paths or not agent_paths[name]:
            print(f"Warning: Path not found or empty for agent {name}. Skipping.")
            continue
        try:
            agent = agent_class(state_dim, action_dim, device=device)
            agent.load(agent_paths[name])
            agent.eval()
            agents[name] = agent
            print(f" -> Loaded {name} from {agent_paths[name]}")
        except FileNotFoundError:
             print(f"Error: Model file not found for agent {name} at {agent_paths[name]}")
        except Exception as e:
            print(f"Error loading agent {name} from {agent_paths[name]}: {e}")

    if not agents:
        print("Error: No agents were loaded successfully.")
        return None
    agent_names = list(agents.keys())

    # calculate previous period sentiment for change calculation
  
    if 'PeriodSentiment' not in evaluation_data.columns:
         raise ValueError("'PeriodSentiment' column required in evaluation_data.")
    evaluation_data['PrevPeriodSentiment'] = evaluation_data['PeriodSentiment'].shift(period)

    # initial agent selection
    # we simulate the first 'period' to get a history for validation
    print(f"Performing initial agent validation (simulating first {period} steps)...")
    best_initial_agent_name = None
    best_initial_score = -np.inf

    # here, we create a separate env instance just for initial validation runs
    init_env = env_class(evaluation_data)

    for agent_name, agent_instance in agents.items():
         print(f"  Validating initial: {agent_name}")
         init_state = init_env.reset()
         if init_env.n_steps < period:
              print(f"Warning: Data length ({init_env.n_steps}) shorter than period ({period}). Validating on available steps.")
              validation_steps = init_env.n_steps
         else:
              validation_steps = period

         # simulation for initial period
         init_done = False
         for _ in range(validation_steps):
             if init_done or init_env.current_step >= init_env.n_steps: break
             init_action, _, _ = agent_instance.select_action(init_state, evaluate=True)
             init_state, _, init_done, _ = init_env.step(init_action)

         # calculate score on the recorded history
         init_history = init_env.history
         if len(init_history) < 2:
              print(f"    Skipping {agent_name}: Not enough history ({len(init_history)}) for initial validation score.")
              score = -np.inf
         else:
              score = validation_metric_fn(np.array(init_history), alpha=alpha)
              print(f"    Initial Score (Chi): {score:.4f}")

         if score > best_initial_score:
             best_initial_score = score
             best_initial_agent_name = agent_name

    if best_initial_agent_name is None:
         print("Error: Initial validation failed for all agents. Cannot determine starting agent.")
         
         if agent_names:
              print("Warning: Using first loaded agent as fallback.")
              best_initial_agent_name = agent_names[0]
         else: 
              return None

    current_agent_name = best_initial_agent_name
    current_agent = agents[current_agent_name]
    print(f"--- Initial Agent Selected: {current_agent_name} (Score: {best_initial_score:.4f}) ---")

    #  Main Backtest Loop 
    results = []
    # We start trading after the initial validation period used to select the agent
    start_index = period
    total_steps = len(evaluation_data)

    state = hist_env.reset()
    
    print(f"Fast-forwarding main environment to start index {start_index}...")
    for _ in range(start_index):
         if hist_env.current_step >= hist_env.n_steps: break 
         ff_action = np.zeros(action_dim)
         state, _, ff_done, _ = hist_env.step(ff_action)
         if ff_done:
              print("Warning: Environment finished during fast-forward before main loop.")
              return pd.DataFrame(results) 
    print("Fast-forward complete. Starting main trading loop.")

    for step in range(start_index, total_steps):
        date = evaluation_data.index[step] 
        if hist_env.done:
            print(f"Environment marked done at start of step {step}. Ending loop.")
            break

        steps_elapsed_in_eval = step - start_index

        if steps_elapsed_in_eval > 0 and steps_elapsed_in_eval % period == 0:
            print(f"\n--- Period End Check: Date {date} (Step {step}, Steps in Eval {steps_elapsed_in_eval}) ---")
            current_period_sentiment = evaluation_data.loc[date, 'PeriodSentiment']
            prev_period_sentiment = evaluation_data.loc[date, 'PrevPeriodSentiment']

            if isinstance(current_period_sentiment, pd.Series):
                current_period_sentiment = current_period_sentiment.iloc[0]
            if isinstance(prev_period_sentiment, pd.Series):
                prev_period_sentiment = prev_period_sentiment.iloc[0]
            if pd.isna(current_period_sentiment) or pd.isna(prev_period_sentiment):
                print("Sentiment or previous sentiment is NaN. Skipping switch check.")
            else:
                sentiment_change = abs(current_period_sentiment - prev_period_sentiment)
                print(f"Current Period Sentiment: {current_period_sentiment:.4f}")
                print(f"Previous Period Sentiment: {prev_period_sentiment:.4f}")
                print(f"Absolute Sentiment Change: {sentiment_change:.4f} (Threshold: {beta})")

                if sentiment_change > beta:
                    print(f"Sentiment change > {beta}. Re-validating agents...")
                    validation_start_idx = step - period
                    validation_end_idx = step

                    best_agent_name_in_validation = None
                    best_score_in_validation = -np.inf

                    print(f" -> Validating {len(agents)} agents on window index {validation_start_idx} to {validation_end_idx}...")

                    for agent_name, agent_instance in agents.items():
                        print(f"    Validating agent: {agent_name}...")
                        try: 
                            temp_env = env_class(evaluation_data)
                            if temp_env.n_steps <= validation_start_idx:
                                print(f"      Skipping {agent_name}: Validation window starts beyond data length.")
                                continue

                            temp_state = temp_env.reset()
                            for _ in range(validation_start_idx):
                                if temp_env.current_step >= temp_env.n_steps: break
                                ff_action = np.zeros(action_dim)
                                temp_state, _, ff_done, _ = temp_env.step(ff_action)
                                if ff_done: break
                            if temp_env.current_step != validation_start_idx:
                                 print(f"      Warning: Failed to fast-forward temp_env accurately for {agent_name} validation.")

                            # validation simulation
                            steps_in_window = validation_end_idx - validation_start_idx
                            temp_state = temp_env._get_state() 
                            done_in_validation = False
                            for _ in range(steps_in_window):
                                 if done_in_validation or temp_env.current_step >= temp_env.n_steps: break
                                 val_action, _, _ = agent_instance.select_action(temp_state, evaluate=True)
                                 temp_state, _, done_in_validation, _ = temp_env.step(val_action)

                            validation_portfolio_history = temp_env.history[validation_start_idx : validation_end_idx + 1]

                            if len(validation_portfolio_history) < 2:
                                print(f"      Skipping {agent_name}: Not enough history points ({len(validation_portfolio_history)}) in validation window.")
                                score = -np.inf
                            else:
                                score = validation_metric_fn(np.array(validation_portfolio_history), alpha=alpha)
                                print(f"      Agent {agent_name} validation score (Chi): {score:.4f}")

                            if score > best_score_in_validation:
                                best_score_in_validation = score
                                best_agent_name_in_validation = agent_name

                        except Exception as e_val:
                             print(f"      Error during validation for agent {agent_name}: {e_val}")
                             score = -np.inf # we penalize agent if validation fails

                    # check if a best agent was found and if it's different
                    if best_agent_name_in_validation is not None and best_agent_name_in_validation != current_agent_name:
                        print(f"Switching agent from {current_agent_name} to {best_agent_name_in_validation} (Score: {best_score_in_validation:.4f})")
                        current_agent_name = best_agent_name_in_validation
                        current_agent = agents[current_agent_name]
                    elif best_agent_name_in_validation is not None:
                         print(f"Keeping current agent: {current_agent_name} (Best score: {best_score_in_validation:.4f})")
                    else:
                         print("Warning: Validation failed for all agents? Keeping current agent.")

                else: # sentiment_change <= beta
                    print("Sentiment change <= threshold. Keeping current agent.")


        # daily trading actio
        
        if state is None: 
             print(f"Error: State is None at step {step}. Breaking.")
             break

        action, _, _ = current_agent.select_action(state, evaluate=True)

        next_state, reward, done, info = hist_env.step(action)

        results.append({
            'Date': date,
            'PortfolioValue': hist_env.portfolio_value,
            'DailyReward': reward,
            'ActiveAgent': current_agent_name
        })

        state = next_state 

        if done:
            print(f"Environment finished normally at step {step}. Date: {date}")
            break 

    print("--- Backtest Loop Finished ---")
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.set_index('Date', inplace=True)
    else:
         print("Warning: No results were generated during the backtest.")

    return results_df

agent_classes = {
    "DDPG": DDPGAgent,
    "PPO": PPOAgent,
    "A2C": A2CAgent
}

agent_paths = {
    "DDPG": "trained_models/ddpg_sim_trained.pth", 
    "PPO": "trained_models/ppo_sim_trained.pth",   
    "A2C": "trained_models/a2c_sim_trained.pth"    
}

dynamic_results_df = run_dynamic_backtest(
    evaluation_data=test_df,        
    agent_classes=agent_classes,          
    agent_paths=agent_paths,              
    validation_metric_fn=calculate_chi_score, 
    env_class=PortfolioEnvHistorical,     
    alpha=ALPHA,                          
    beta=BETA,                            
    period=SENTIMENT_PERIOD,              
    device=DEVICE                         
)

if dynamic_results_df is not None:
    print("\n--- Dynamic Backtest Execution Finished ---")
    print("Results DataFrame head:")
    print(dynamic_results_df.head())
    print("\nResults DataFrame tail:")
    print(dynamic_results_df.tail())
    
else:
    print("\n--- Dynamic Backtest Failed ---")

