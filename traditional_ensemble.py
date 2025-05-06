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
def calculate_chi_score(vals, alpha=0.25, **kwargs): return np.random.rand() # Dummy metric
class PortfolioEnvHistorical: # Dummy Env
     def __init__(self, data, **kwargs):
         self.n_steps = len(data); self.state_dim = 10; self.action_dim = 3
         self.current_step = 0; self.portfolio_value = 1_000_000
         self.history = [self.portfolio_value]; self.done = False
     def reset(self): self.current_step = 0; self.done = False; self.history=[1_000_000]; return np.zeros(self.state_dim)
     def step(self, action):
         if self.current_step >= self.n_steps - 1: self.done = True
         reward = np.random.randn() * 100; self.portfolio_value += reward
         self.history.append(self.portfolio_value); self.current_step += 1
         return np.zeros(self.state_dim), reward, self.done, {}
     def _get_state(self): return np.zeros(self.state_dim)


ALPHA = 0.25           
FIXED_PERIOD = 62      
DEVICE = 'cpu'

agent_paths = {
    "DDPG": "trained_models/ddpg_sim_trained.pth",
    "PPO": "trained_models/ppo_sim_trained.pth",
    "A2C": "trained_models/a2c_sim_trained.pth"
}

save_directory = "trained_models"
if not os.path.exists(save_directory): os.makedirs(save_directory)
for p in agent_paths.values():
     if not os.path.exists(p):
         with open(p, 'w') as f: f.write('dummy model data')


def run_traditional_ensemble_backtest(evaluation_data: pd.DataFrame,
                                      agent_classes: Dict[str, Type[BaseAgent]],
                                      agent_paths: Dict[str, str],
                                      validation_metric_fn: callable,
                                      env_class: Type[PortfolioEnvHistorical],
                                      alpha: float = ALPHA,
                                      period: int = FIXED_PERIOD,
                                      device: str = DEVICE) -> Optional[pd.DataFrame]:
    """
    Runs the backtest with traditional ensemble switching (fixed period).

    Args:
        evaluation_data (pd.DataFrame): Prepared historical data (needs stock features).
                                         Sentiment columns are not used by this function.
        agent_classes (dict): Dictionary mapping agent names to their classes.
        agent_paths (dict): Dictionary mapping agent names to saved model file paths.
        validation_metric_fn (callable): Function calculate_chi_score(portfolio_values, alpha=...).
        env_class (class): The PortfolioEnvHistorical class.
        alpha (float): Weight for Sharpe in Chi score.
        period (int): Length of fixed switching/validation period.
        device (str): Device ('cpu' or 'cuda').

    Returns:
        pd.DataFrame: DataFrame with portfolio history and active agent info, indexed by Date.
                      Returns None if errors occur during setup.
    """
    print("--- Starting Traditional Ensemble Backtest ---")

    print("Initializing environment...")
    if not isinstance(evaluation_data.index, pd.DatetimeIndex):
        raise ValueError("Evaluation data must have a DatetimeIndex.")

    try:
        hist_env = env_class(evaluation_data)
        state_dim = hist_env.state_dim
        action_dim = hist_env.action_dim
    except Exception as e:
        print(f"Error initializing PortfolioEnvHistorical: {e}")
        import traceback
        traceback.print_exc()
        return None
    print(f"Environment initialized. State Dim: {state_dim}, Action Dim: {action_dim}")

    # Load agents
    agents = {}
    print("Loading trained agents...")
    for name, agent_class in agent_classes.items():
        if name not in agent_paths or not agent_paths[name]: continue
        try:
            agent = agent_class(state_dim, action_dim, device=device)
            agent.load(agent_paths[name])
            agent.eval()
            agents[name] = agent
            print(f" -> Loaded {name}")
        except FileNotFoundError: print(f"Error: File not found for {name} at {agent_paths[name]}")
        except Exception as e: print(f"Error loading agent {name}: {e}")

    if not agents: print("Error: No agents loaded."); return None
    agent_names = list(agents.keys())

    # initial agent selection
    print(f"Performing initial agent validation (simulating first {period} steps)...")
    best_initial_agent_name = None; best_initial_score = -np.inf
    init_env = env_class(evaluation_data)
    for agent_name, agent_instance in agents.items():
         print(f"  Validating initial: {agent_name}")
         init_state = init_env.reset()
         validation_steps = min(period, init_env.n_steps)
         init_done = False
         for _ in range(validation_steps):
             if init_done or init_env.current_step >= init_env.n_steps: break
             init_action, _, _ = agent_instance.select_action(init_state, evaluate=True)
             init_state, _, init_done, _ = init_env.step(init_action)
         init_history = init_env.history
         if len(init_history) < 2: score = -np.inf
         else: score = validation_metric_fn(np.array(init_history), alpha=alpha)
         print(f"    Initial Score (Chi): {score:.4f}")
         if score > best_initial_score: best_initial_score = score; best_initial_agent_name = agent_name

    if best_initial_agent_name is None:
        print("Error: Initial validation failed. Using first loaded agent as fallback.")
        if not agent_names: return None 
        best_initial_agent_name = agent_names[0]

    current_agent_name = best_initial_agent_name
    current_agent = agents[current_agent_name]
    print(f"--- Initial Agent Selected: {current_agent_name} (Score: {best_initial_score:.4f}) ---")

    # main Backtest Loop Setup
    results = []; start_index = period; total_steps = len(evaluation_data)
    state = hist_env.reset()
    print(f"Fast-forwarding main environment to start index {start_index}...")
    for _ in range(start_index):
        if hist_env.current_step >= hist_env.n_steps: break
        ff_action = np.zeros(action_dim); state, _, ff_done, _ = hist_env.step(ff_action)
        if ff_done: print("Warning: Env finished during fast-forward."); return pd.DataFrame(results)
    print("Fast-forward complete. Starting main trading loop.")

    # main backtest Loop
    for step in range(start_index, total_steps):
        date = evaluation_data.index[step]
        if hist_env.done: print(f"Env done at start of step {step}."); break
            
        steps_elapsed_in_eval = step - start_index
        if steps_elapsed_in_eval > 0 and steps_elapsed_in_eval % period == 0:
            print(f"\n--- Fixed Period End: Date {date} (Step {step}). Validating Agents ---")
            validation_start_idx = step - period
            validation_end_idx = step 

            best_agent_name_in_validation = None
            best_score_in_validation = -np.inf

            # Agent revalidation logic (runs every period) 
            for agent_name, agent_instance in agents.items():
                print(f"    Validating agent: {agent_name}...")
                try:
                    temp_env = env_class(evaluation_data)
                    if temp_env.n_steps <= validation_start_idx: continue

                    temp_state = temp_env.reset()
                    for _ in range(validation_start_idx): 
                         if temp_env.current_step >= temp_env.n_steps: break
                         ff_action = np.zeros(action_dim); temp_state, _, ff_done, _ = temp_env.step(ff_action)
                         if ff_done: break
                    if temp_env.current_step != validation_start_idx: print(f"Warning: Failed fast-forward for {agent_name}.")

                    # validation simulation
                    steps_in_window = validation_end_idx - validation_start_idx
                    temp_state = temp_env._get_state()
                    done_in_validation = False
                    for _ in range(steps_in_window):
                         if done_in_validation or temp_env.current_step >= temp_env.n_steps: break
                         val_action, _, _ = agent_instance.select_action(temp_state, evaluate=True)
                         temp_state, _, done_in_validation, _ = temp_env.step(val_action)

                    validation_portfolio_history = temp_env.history[validation_start_idx : validation_end_idx + 1]
                    if len(validation_portfolio_history) < 2: score = -np.inf
                    else: score = validation_metric_fn(np.array(validation_portfolio_history), alpha=alpha)
                    print(f"      Agent {agent_name} validation score (Chi): {score:.4f}")

                    if score > best_score_in_validation:
                        best_score_in_validation = score
                        best_agent_name_in_validation = agent_name

                except Exception as e_val: print(f"Error validating {agent_name}: {e_val}"); score = -np.inf

            # Select the best agent for the next period
            if best_agent_name_in_validation is not None and best_agent_name_in_validation != current_agent_name:
                print(f"Switching agent from {current_agent_name} to {best_agent_name_in_validation} (Score: {best_score_in_validation:.4f})")
                current_agent_name = best_agent_name_in_validation
                current_agent = agents[current_agent_name]
            elif best_agent_name_in_validation is not None:
                 print(f"Keeping current agent: {current_agent_name} (Best score: {best_score_in_validation:.4f})")
            else:
                 print("Warning: Validation failed for all agents? Keeping current agent.")


        # daily trading Action
        action, _, _ = current_agent.select_action(state, evaluate=True)
        next_state, reward, done, info = hist_env.step(action)

        # log results
        results.append({
            'Date': date, 'PortfolioValue': hist_env.portfolio_value,
            'DailyReward': reward, 'ActiveAgent': current_agent_name
        })
        state = next_state
        if done: print(f"Env finished at step {step}. Date: {date}"); break

    print("--- Traditional Ensemble Backtest Finished ---")
    results_df = pd.DataFrame(results)
    if not results_df.empty: results_df.set_index('Date', inplace=True)
    else: print("Warning: No results generated.")
    return results_df

print("\n--- Running Traditional Ensemble Backtest ---")
# Example Call:
traditional_results_df = run_traditional_ensemble_backtest(
    evaluation_data=test_df,      
    agent_classes=agent_classes,        
    agent_paths=agent_paths,             
    validation_metric_fn=calculate_chi_score,
    env_class=PortfolioEnvHistorical,   
    alpha=ALPHA,                         
    period=FIXED_PERIOD,                
    device=DEVICE              
)

if traditional_results_df is not None:
    print("\n--- Traditional Ensemble Backtest Execution Finished ---")
    print("Results DataFrame head:")
    print(traditional_results_df.head())

else:
    print("\n--- Traditional Ensemble Backtest Failed ---")

