#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def calculate_sharpe_ratio(portfolio_values: np.ndarray | pd.Series,
                           risk_free_rate: float = 0.0,
                           periods_per_year: int = 252) -> float:
    """Calculates the annualized Sharpe ratio for a portfolio value series."""
    if isinstance(portfolio_values, pd.Series):
        portfolio_values = portfolio_values.sort_index().values
    elif isinstance(portfolio_values, list):
         portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) < 2: return 0.0
    valid_mask = portfolio_values[:-1] != 0
    daily_returns = np.full_like(portfolio_values[:-1], fill_value=np.nan, dtype=float)
    if np.any(valid_mask): daily_returns[valid_mask] = np.diff(portfolio_values)[valid_mask] / portfolio_values[:-1][valid_mask]
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    if len(daily_returns) == 0: return 0.0
    std_dev = np.std(daily_returns)
    if std_dev < 1e-9:
        mean_return = np.mean(daily_returns); target_return_daily = risk_free_rate / periods_per_year
        return 0.0 if np.isclose(mean_return, target_return_daily) else np.sign(mean_return - target_return_daily) * 1e9
    excess_returns = daily_returns - (risk_free_rate / periods_per_year)
    sharpe = np.mean(excess_returns) / std_dev
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)
    return annualized_sharpe if np.isfinite(annualized_sharpe) else 0.0

def calculate_sortino_ratio(portfolio_values: np.ndarray | pd.Series,
                            risk_free_rate: float = 0.0,
                            periods_per_year: int = 252) -> float:
    """Calculates the annualized Sortino ratio for a portfolio value series."""
    if isinstance(portfolio_values, pd.Series):
        portfolio_values = portfolio_values.sort_index().values
    elif isinstance(portfolio_values, list):
         portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) < 2: return 0.0
    valid_mask = portfolio_values[:-1] != 0
    daily_returns = np.full_like(portfolio_values[:-1], fill_value=np.nan, dtype=float)
    if np.any(valid_mask): daily_returns[valid_mask] = np.diff(portfolio_values)[valid_mask] / portfolio_values[:-1][valid_mask]
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    if len(daily_returns) == 0: return 0.0
    target_return = risk_free_rate / periods_per_year
    excess_returns = daily_returns - target_return
    mean_excess_return = np.mean(excess_returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 1: return np.inf if mean_excess_return > 0 else 0.0
    downside_deviation = np.std(downside_returns)
    if downside_deviation < 1e-9: return np.inf if mean_excess_return > 0 else 0.0
    sortino = mean_excess_return / downside_deviation
    annualized_sortino = sortino * np.sqrt(periods_per_year)
    return annualized_sortino if (np.isfinite(annualized_sortino) or (np.isinf(annualized_sortino) and annualized_sortino > 0)) else 0.0

def calculate_annualized_return(portfolio_values: np.ndarray | pd.Series,
                                periods_per_year: int = 252) -> float:
    """Calculates the annualized return from a series of portfolio values."""
    if isinstance(portfolio_values, pd.Series):
        portfolio_values = portfolio_values.sort_index().values
    elif isinstance(portfolio_values, list):
         portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) < 2: return 0.0
    num_periods = len(portfolio_values) - 1
    if num_periods == 0: return 0.0
    start_value = portfolio_values[0]
    end_value = portfolio_values[-1]
    if start_value <= 1e-9: return np.inf if end_value > 0 else 0.0
    total_return_factor = end_value / start_value
    annualized_return = (total_return_factor ** (periods_per_year / num_periods)) - 1
    return annualized_return if np.isfinite(annualized_return) else 0.0

def calculate_max_drawdown(portfolio_values: np.ndarray | pd.Series) -> float:
    """Calculates the maximum drawdown from a series of portfolio values."""
    if isinstance(portfolio_values, pd.Series):
        portfolio_values = portfolio_values.sort_index().values
    elif isinstance(portfolio_values, list):
         portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) < 1: return 0.0
    running_peak = np.maximum.accumulate(portfolio_values)
    running_peak[running_peak < 1e-9] = 1e-9 # Avoid division by zero 
    drawdown = (portfolio_values - running_peak) / running_peak
    max_dd = np.min(drawdown)
    return min(max_dd, 0.0) if np.isfinite(max_dd) else 0.0

def calculate_calmar_ratio(portfolio_values: np.ndarray | pd.Series,
                           periods_per_year: int = 252) -> float:
    """Calculates the Calmar ratio (Annualized Return / Absolute Max Drawdown)."""
    annualized_return = calculate_annualized_return(portfolio_values, periods_per_year)
    max_drawdown = calculate_max_drawdown(portfolio_values)
    if abs(max_drawdown) < 1e-9: return np.inf if annualized_return > 0 else 0.0
    calmar = annualized_return / abs(max_drawdown)
    return calmar if np.isfinite(calmar) else 0.0

#  Wrapper Function 
def calculate_all_metrics(portfolio_values, risk_free_rate=0.0, periods_per_year=252):
    """ Calculates a dictionary of performance metrics. """
    if not isinstance(portfolio_values, pd.Series):
        portfolio_values = pd.Series(portfolio_values)

    if len(portfolio_values) < 2:
        print("Warning: Portfolio history too short for metrics calculation.")

        return {
            'Cumulative Return': np.nan, 'Annualized Return': np.nan,
            'Maximum Drawdown': np.nan, 'Annualized Volatility': np.nan,
            'Sharpe Ratio': np.nan, 'Sortino Ratio': np.nan, 'Calmar Ratio': np.nan
        }

    metrics = {}
    metrics['Cumulative Return'] = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1

    # daily returns 
    daily_returns = portfolio_values.pct_change().dropna()

    if len(daily_returns) > 0:
         metrics['Annualized Volatility'] = np.std(daily_returns) * np.sqrt(periods_per_year)
    else:
         metrics['Annualized Volatility'] = 0.0

    metrics['Annualized Return'] = calculate_annualized_return(portfolio_values.values, periods_per_year)
    metrics['Maximum Drawdown'] = calculate_max_drawdown(portfolio_values.values)
    metrics['Sharpe Ratio'] = calculate_sharpe_ratio(portfolio_values.values, risk_free_rate, periods_per_year)
    metrics['Sortino Ratio'] = calculate_sortino_ratio(portfolio_values.values, risk_free_rate, periods_per_year)
    metrics['Calmar Ratio'] = calculate_calmar_ratio(portfolio_values.values, periods_per_year)

    return metrics

dynamic_ensemble_history = dynamic_results_df['PortfolioValue']
traditional_ensemble_history = traditional_results_df['PortfolioValue']

if ('dynamic_ensemble_history' in locals() and isinstance(dynamic_ensemble_history, pd.Series) and
    'traditional_ensemble_history' in locals() and isinstance(traditional_ensemble_history, pd.Series)):


    print("\nCalculating metrics for Dynamic Ensemble...")
    dynamic_metrics = calculate_all_metrics(dynamic_ensemble_history)

    print("\nCalculating metrics for Traditional Ensemble...")
    traditional_metrics = calculate_all_metrics(traditional_ensemble_history)


    comparison_df = pd.DataFrame({
        "Dynamic Ensemble": dynamic_metrics,
        "Traditional Ensemble": traditional_metrics
    })

    print("\n--- Performance Metrics Comparison ---")

    print(comparison_df.T.round(4)) 

else:
    print("\nPlease ensure 'dynamic_ensemble_history' and 'traditional_ensemble_history' Series are defined.")
    
def calculate_all_metrics(portfolio_values, risk_free_rate=0.0, periods_per_year=252):
    """ Calculates a dictionary of performance metrics. """
    if not isinstance(portfolio_values, pd.Series):
        portfolio_values = pd.Series(portfolio_values) 

    if len(portfolio_values) < 2:
        print("Warning: Portfolio history too short for metrics calculation.")
        return { 
            'Cumulative Return': 0.0, 'Annualized Return': 0.0,
            'Maximum Drawdown': 0.0, 'Annualized Volatility': 0.0,
            'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0, 'Calmar Ratio': 0.0
        }

    metrics = {}
    metrics['Cumulative Return'] = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1

    daily_returns = portfolio_values.pct_change().dropna()
    if len(daily_returns) > 0:
         metrics['Annualized Volatility'] = np.std(daily_returns) * np.sqrt(periods_per_year)
    else:
         metrics['Annualized Volatility'] = 0.0

    metrics['Annualized Return'] = calculate_annualized_return(portfolio_values, periods_per_year)
    metrics['Maximum Drawdown'] = calculate_max_drawdown(portfolio_values)
    metrics['Sharpe Ratio'] = calculate_sharpe_ratio(portfolio_values, risk_free_rate, periods_per_year)
    metrics['Sortino Ratio'] = calculate_sortino_ratio(portfolio_values, risk_free_rate, periods_per_year)
    metrics['Calmar Ratio'] = calculate_calmar_ratio(portfolio_values, periods_per_year)

    return metrics

print("Calculating metrics for Dynamic Ensemble...")
dynamic_metrics = calculate_all_metrics(dynamic_ensemble_history)

print("Calculating metrics for Traditional Ensemble...")
traditional_metrics = calculate_all_metrics(traditional_ensemble_history)

comparison_df = pd.DataFrame({
    "Dynamic Ensemble": dynamic_metrics,
    "Traditional Ensemble": traditional_metrics
})

print("\n--- Performance Metrics Comparison ---")
print(comparison_df. T.round(4)) 

plt.figure(figsize=(12, 6))
plt.plot(dynamic_ensemble_history.index, dynamic_ensemble_history, label='Dynamic Ensemble (Sentiment)')
plt.plot(traditional_ensemble_history.index, traditional_ensemble_history, label='Traditional Ensemble (Fixed Period)')

plt.title('Portfolio Value Comparison: Dynamic vs. Traditional Ensemble')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.yscale('log') 
plt.tight_layout()
plt.show()

