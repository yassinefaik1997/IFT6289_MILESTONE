#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class PortfolioEnvHistorical:
    """
    Environment for portfolio management adapted for historical backtesting.
    Expects a pandas DataFrame with:
    - DatetimeIndex
    - MultiIndex columns (StockSymbol, Feature) where Feature includes 'AdjClose'
    - Potentially other top-level columns like 'Sentiment', 'PeriodSentiment' (ignored by env).
    """
    INITIAL_BALANCE=1000000
    COMMISSION_RATE=0
    
    def __init__(
        self,
        evaluation_data: pd.DataFrame,
        initial_balance: float = INITIAL_BALANCE,
        commission_rate: float = COMMISSION_RATE,
        price_col_name: str = 'AdjClose' # price column
    ):
        if not isinstance(evaluation_data, pd.DataFrame):
            raise ValueError("evaluation_data must be a pandas DataFrame")

        #  stock symbols
        if isinstance(evaluation_data.columns, pd.MultiIndex):
             stock_symbols = evaluation_data.columns.levels[0].unique().tolist()
             stock_symbols = [s for s in stock_symbols if isinstance(evaluation_data[s].columns, pd.MultiIndex)]
        else:
             stock_cols_multi = [col[0] for col in evaluation_data.columns if isinstance(col, tuple)]
             if not stock_cols_multi:
                   raise ValueError("No MultiIndex columns found for stock data.")
             stock_symbols = list(set(stock_cols_multi))

        for col_name in ['Sentiment', 'PeriodSentiment']:
             if col_name in stock_symbols:
                 stock_symbols.remove(col_name)

        if not stock_symbols:
             raise ValueError("Could not identify stock symbols in DataFrame columns.")

        try:
            stock_data = evaluation_data[stock_symbols].copy()
            if not isinstance(stock_data.columns, pd.MultiIndex):
                 raise ValueError("Selected stock data columns are not MultiIndex.")
        except KeyError:
             raise ValueError("Could not select stock data using identified symbols. Check DataFrame structure.")

        required_price_cols = [(symbol, price_col_name) for symbol in stock_symbols]
        missing_cols = [col for col in required_price_cols if col not in stock_data.columns]
        if missing_cols:
             raise ValueError(f"Price column '{price_col_name}' missing for symbols: {missing_cols}")

        stock_data = stock_data.apply(pd.to_numeric, errors='coerce') 

        if stock_data.isnull().values.any():
            print("Warning: Stock data contains NaNs. Filling forward/backward.")
            stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
        if stock_data.isnull().values.any():
            print("Warning: NaNs remain after fill in stock data. Filling with 0 (use caution).")
            stock_data = stock_data.fillna(0) 

        if stock_data.empty or stock_data.shape[0] < 2:
            raise ValueError("Stock data is empty or too short after cleaning.")


        self.price_data = stock_data # only stock features internally
        self.price_col_name = price_col_name 
        self.n_assets = len(stock_symbols)
        self.n_steps = len(self.price_data)
        self.asset_names = stock_symbols # actual stock symbols

        self.initial_balance = initial_balance
        self.commission_rate = commission_rate

        # state dimension: [normalized prices (n_assets) + normalized holdings (n_assets) + normalized balance (1)]
        self.state_dim = self.n_assets * 2 + 1
        self.action_dim = self.n_assets # Outputs number of shares to trade per asset

        # storing initial prices for normalization
        self.initial_prices = self._get_prices_at_step(0)
        if self.initial_prices is None:
             raise ValueError("Could not retrieve initial prices for normalization.")
                
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
             print("Warning: Price data too short for environment initialization.")
             return np.zeros(self.state_dim, dtype=np.float32)

        return self._get_state() 

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            step_idx = min(self.current_step, self.n_steps - 1)
            final_state = self._get_state_at_step(step_idx) 
            return final_state, 0.0, True, {"error": "Episode already finished"}

        current_step_idx = self.current_step
        current_prices = self._get_prices_at_step(current_step_idx)

        if current_prices is None:
             self.done = True
             print(f"Error: Could not get current prices at step {current_step_idx}.")
             last_valid_step_idx = max(0, current_step_idx - 1)
             last_state = self._get_state_at_step(last_valid_step_idx) # state from last valid step
             self.current_step = self.n_steps # ensuring step counter is out of bounds
             return last_state, 0.0, True, {"error": "Reached end of price data unexpectedly"}

        start_portfolio_value = self._calculate_portfolio_value(current_prices)
        action = np.array(action, dtype=np.float32)

        # trades execution
        sell_indices = np.where(action < 0)[0]; buy_indices = np.where(action > 0)[0]
        # sells
        for idx in sell_indices:
            shares_to_sell = min(abs(action[idx]), self.shares_held[idx])
            if shares_to_sell > 1e-6:
                 sell_value = shares_to_sell * current_prices[idx]
                 commission = sell_value * self.commission_rate
                 self.balance += sell_value - commission
                 self.shares_held[idx] -= shares_to_sell
        # buys
        available_balance = self.balance; total_buy_cost_needed = 0; buy_costs = {}
        for idx in buy_indices:
            shares_to_buy = abs(action[idx])
            if shares_to_buy > 1e-6:
                buy_value = shares_to_buy * current_prices[idx]
                commission = buy_value * self.commission_rate; cost = buy_value + commission
                if cost > 1e-9: buy_costs[idx] = {'shares': shares_to_buy, 'cost': cost}; total_buy_cost_needed += cost
        scale_factor = 1.0
        if total_buy_cost_needed > available_balance: scale_factor = available_balance / total_buy_cost_needed if total_buy_cost_needed > 1e-9 else 0.0
        if available_balance <= 1e-9: scale_factor = 0.0
        for idx in buy_indices:
             if idx in buy_costs:
                 scaled_shares = buy_costs[idx]['shares'] * scale_factor; scaled_cost = buy_costs[idx]['cost'] * scale_factor
                 if scaled_shares > 1e-6 and scaled_cost <= self.balance + 1e-6:
                     self.balance -= scaled_cost; self.shares_held[idx] += scaled_shares
        self.balance = max(self.balance, 0.0)

        next_step_idx = current_step_idx + 1
        self.current_step = next_step_idx 
        next_prices = self._get_prices_at_step(next_step_idx)

        if self.current_step >= self.n_steps:
            self.done = True
            end_portfolio_value = self._calculate_portfolio_value(current_prices) 
            reward = end_portfolio_value - start_portfolio_value
        else:
            self.done = False
            if next_prices is None:
                 print(f"Warning: next_prices is None at step {self.current_step}. Using current prices for value.")
                 next_prices = current_prices 
            end_portfolio_value = self._calculate_portfolio_value(next_prices) 
            reward = end_portfolio_value - start_portfolio_value 

        self.portfolio_value = end_portfolio_value
        self.history.append(self.portfolio_value)

        next_state = self._get_state()

        return next_state, reward, self.done, {} 

    def _get_state(self) -> np.ndarray:
        """ Gets the state representation for the *current* step index. """
        return self._get_state_at_step(self.current_step)


    def _get_state_at_step(self, step_idx: int) -> np.ndarray:
         """ Helper function to get state at a specific step index. """
            
         step_idx = min(step_idx, self.n_steps - 1)
         current_prices = self._get_prices_at_step(step_idx)

         if current_prices is None:
              print(f"Warning: Could not get prices at step {step_idx} in _get_state_at_step. Using last known or zero state.")
              prev_step_idx = max(0, step_idx - 1)
              current_prices = self._get_prices_at_step(prev_step_idx)
              if current_prices is None:
                   return np.zeros(self.state_dim, dtype=np.float32)

         # state calculation
         # normalize prices by initial prices of the evaluation period
         norm_prices = (current_prices / (self.initial_prices + 1e-9)).astype(np.float32)

         # normalize shares by a heuristic based on initial balance and prices
         shares_normalization_factor = (self.initial_balance / self.n_assets) / (self.initial_prices + 1e-9) + 1.0 # Example heuristic
         norm_shares = (self.shares_held / (shares_normalization_factor + 1e-9)).astype(np.float32)

         # normalize balance
         norm_balance = np.array([self.balance / self.initial_balance], dtype=np.float32)

         # concatenate features
         state = np.concatenate([norm_prices, norm_shares, norm_balance], axis=0)
         # 

         return state.astype(np.float32)


    def _get_prices_at_step(self, step: int) -> Optional[np.ndarray]:
        """Safely gets the relevant price vector for a given step using iloc."""
        if step < 0 or step >= self.n_steps:
            return None
        try:
            prices_at_step_series = self.price_data.iloc[step]
            price_vector = prices_at_step_series.xs(self.price_col_name, level='Feature').values
            return price_vector.astype(np.float32)
        except IndexError:
            return None 
        except KeyError:
             print(f"Error: Price column '{self.price_col_name}' not found in Series index levels at step {step}.")
             return None
        except Exception as e:
            print(f"Error getting prices at step {step}: {e}")
            return None


    def _calculate_portfolio_value(self, prices: Optional[np.ndarray]) -> float:
         """Calculates total portfolio value using provided prices."""
         if prices is None:
             prices = self._get_prices_at_step(min(self.current_step, self.n_steps - 1))
             if prices is None:
                 prices = self._get_prices_at_step(max(0, self.current_step - 1))
             if prices is None:
                 print("Error: Cannot calculate portfolio value, price unavailable.")
                 return self.balance

         prices = np.array(prices, dtype=np.float32)
         if len(prices) != self.n_assets:
             print(f"Error: Price vector length mismatch. Retrying get_prices_at_step.")
             prices = self._get_prices_at_step(min(self.current_step, self.n_steps - 1))
             if prices is None or len(prices) != self.n_assets:
                  prices = self._get_prices_at_step(max(0, self.current_step - 1))
             if prices is None or len(prices) != self.n_assets:
                  print("Error: Fallback failed. Cannot calculate portfolio value accurately.")
                  return self.balance

         return float(np.sum(self.shares_held * prices) + self.balance)

    def render(self, title: str = "Portfolio Value Over Time (Historical Eval)") -> None:
        plt.figure(figsize=(10, 5)); plt.plot(self.history); plt.title(title)
        plt.xlabel("Time Steps"); plt.ylabel("Portfolio Value"); plt.grid(True); plt.show()

