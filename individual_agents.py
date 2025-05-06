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
class PortfolioEnvHistorical: # Dummy Env
     def __init__(self, data, **kwargs):
         self.n_steps = len(data); self.state_dim = 10; self.action_dim = 3
         self.current_step = 0; self.portfolio_value = 1_000_000
         self.history = [self.portfolio_value]; self.done = False; self.initial_balance = 1_000_000
     def reset(self): self.current_step = 0; self.done = False; self.history=[1_000_000]; return np.zeros(self.state_dim)
     def step(self, action):
         if self.current_step >= self.n_steps - 1: self.done = True
         reward = np.random.randn() * 100; self.portfolio_value += reward
         self.history.append(self.portfolio_value); self.current_step += 1
         return np.zeros(self.state_dim), reward, self.done, {}
     def _get_state(self): return np.zeros(self.state_dim)

INITIAL_BALANCE = 1_000_000 

# --- Corrected Evaluation Function for DDPG ---
def evaluate_agent_ddpg(agent: DDPGAgent,
                      evaluation_data: pd.DataFrame, 
                      start_date: str,
                      end_date: str,
                      env_class: type = PortfolioEnvHistorical, # Use the correct class
                      device='cpu') -> Tuple[float, List[float]]:
    print(f"Evaluating DDPG from {start_date} to {end_date}...")
    test_df = evaluation_data # Pass potentially larger df if env handles range

    if test_df.empty:
        print(f"Warning: No historical data for evaluation period {start_date}-{end_date}")
        return INITIAL_BALANCE, [INITIAL_BALANCE]

    try:
        env = env_class(test_df)
    except Exception as e:
         print(f"Error initializing environment for DDPG eval: {e}")
         return INITIAL_BALANCE, [INITIAL_BALANCE]

    state = env.reset()
    if env.done:
        print("Warning: Evaluation environment started in 'done' state.")
        return env.initial_balance, [env.initial_balance]

    while not env.done:
        agent.eval()
        action_data = agent.select_action(state, evaluate=True)
        if isinstance(action_data, tuple): action = action_data[0]
        else: action = action_data
        state, _, done, _ = env.step(action)
        if env.current_step >= env.n_steps * 1.1: # Safety break
             print("Warning: Evaluation loop exceeded expected steps. Breaking.")
             break

    final_value = env.portfolio_value
    print(f"DDPG Evaluation complete. Final Value: {final_value:.2f}")
    return final_value, env.history

# Evaluation PPO 
def evaluate_agent_ppo(agent: PPOAgent,
                     evaluation_data: pd.DataFrame,
                     start_date: str,
                     end_date: str,
                     env_class: type = PortfolioEnvHistorical, # Use the correct class
                     device='cpu') -> Tuple[float, List[float]]:
    print(f"Evaluating PPO from {start_date} to {end_date}...")
    test_df = evaluation_data
    if test_df.empty: return INITIAL_BALANCE, [INITIAL_BALANCE]

    try:
        # *** Uses env_class (PortfolioEnvHistorical) ***
        env = env_class(test_df)
    except Exception as e:
         print(f"Error initializing environment for PPO eval: {e}")
         return INITIAL_BALANCE, [INITIAL_BALANCE]

    state = env.reset()
    if env.done: return env.initial_balance, [env.initial_balance]

    while not env.done:
        agent.eval()
        action, _, _ = agent.select_action(state, evaluate=True)
        state, _, done, _ = env.step(action)
        if env.current_step >= env.n_steps * 1.1: break # Safety break

    final_value = env.portfolio_value
    print(f"PPO Evaluation complete. Final Value: {final_value:.2f}")
    return final_value, env.history

# Evaluation for A2C 
def evaluate_agent_a2c(agent: A2CAgent,
                     evaluation_data: pd.DataFrame,
                     start_date: str,
                     end_date: str,
                     env_class: type = PortfolioEnvHistorical, 
                     device='cpu') -> Tuple[float, List[float]]:
    print(f"Evaluating A2C from {start_date} to {end_date}...")
    test_df = evaluation_data
    if test_df.empty: return INITIAL_BALANCE, [INITIAL_BALANCE]

    try:

        env = env_class(test_df)
    except Exception as e:
         print(f"Error initializing environment for A2C eval: {e}")
         return INITIAL_BALANCE, [INITIAL_BALANCE]

    state = env.reset()
    if env.done: return env.initial_balance, [env.initial_balance]

    while not env.done:
        agent.eval()
        action, _, _ = agent.select_action(state, evaluate=True)
        state, _, done, _ = env.step(action)
        if env.current_step >= env.n_steps * 1.1: break 

    final_value = env.portfolio_value
    print(f"A2C Evaluation complete. Final Value: {final_value:.2f}")
    return final_value, env.history

# Store results 
individual_agent_results = {}
individual_agent_histories = {}

# Evaluate DDPG 
if "DDPG" in agent_classes:
    print("\n--- Running DDPG Baseline ---")
    if 'state_dim' not in locals() or 'action_dim' not in locals():
         print("Error: state_dim or action_dim not defined.")
         try:
              dummy_env_eval = PortfolioEnvHistorical(test_df)
              state_dim = dummy_env_eval.state_dim
              action_dim = dummy_env_eval.action_dim
              print(f"Determined dims: State={state_dim}, Action={action_dim}")
         except Exception as e_dim:
              print(f"Could not determine dims for eval: {e_dim}")
    if 'state_dim' in locals() and 'action_dim' in locals(): 
        ddpg_agent_eval = agent_classes["DDPG"](state_dim, action_dim, device=DEVICE) 
        try:
            ddpg_agent_eval.load(agent_paths["DDPG"])
            final_val, history = evaluate_agent_ddpg( 
                agent=ddpg_agent_eval,
                evaluation_data=test_df,
                start_date=test_start_date,
                end_date=test_end_date,
                env_class=PortfolioEnvHistorical,
                device=DEVICE
            )
            individual_agent_results["DDPG"] = final_val
            individual_agent_histories["DDPG"] = history
        except Exception as e:
            print(f"Error during DDPG evaluation: {e}")

#  Evaluate PPO 
if "PPO" in agent_classes:
    print("\n--- Running PPO Baseline ---")
    if 'state_dim' in locals() and 'action_dim' in locals(): 
        ppo_agent_eval = agent_classes["PPO"](state_dim, action_dim, device=DEVICE)
        try:
            ppo_agent_eval.load(agent_paths["PPO"])
            final_val, history = evaluate_agent_ppo( 
                agent=ppo_agent_eval,
                evaluation_data=test_df,
                start_date=test_start_date,
                end_date=test_end_date,
                env_class=PortfolioEnvHistorical,
                device=DEVICE
            )
            individual_agent_results["PPO"] = final_val
            individual_agent_histories["PPO"] = history
        except Exception as e:
            print(f"Error during PPO evaluation: {e}")

# Evaluate A2C 
if "A2C" in agent_classes:
    print("\n--- Running A2C Baseline ---")
    if 'state_dim' in locals() and 'action_dim' in locals(): 
        a2c_agent_eval = agent_classes["A2C"](state_dim, action_dim, device=DEVICE)
        try:
            a2c_agent_eval.load(agent_paths["A2C"])
            final_val, history = evaluate_agent_a2c( 
                agent=a2c_agent_eval,
                evaluation_data=test_df,
                start_date=test_start_date,
                end_date=test_end_date,
                env_class=PortfolioEnvHistorical,
                device=DEVICE
            )
            individual_agent_results["A2C"] = final_val
            individual_agent_histories["A2C"] = history
        except Exception as e:
            print(f"Error during A2C evaluation: {e}")

# Final Values
print("\n--- Individual Agent Baseline Final Values ---")
print(individual_agent_results)

