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

    if len(portfolio_values) < 2:
        return 0.0

    # daily returns
    valid_mask = portfolio_values[:-1] != 0
    daily_returns = np.full_like(portfolio_values[:-1], fill_value=np.nan, dtype=float)
    if np.any(valid_mask): 
        daily_returns[valid_mask] = np.diff(portfolio_values)[valid_mask] / portfolio_values[:-1][valid_mask]
    daily_returns = daily_returns[~np.isnan(daily_returns)] 

    if len(daily_returns) == 0:
         return 0.0

    std_dev = np.std(daily_returns)
    if std_dev < 1e-9: 
        # zero volatility case
        mean_return = np.mean(daily_returns)
        target_return_daily = risk_free_rate / periods_per_year
        if np.isclose(mean_return, target_return_daily):
             return 0.0
        else:

             return np.sign(mean_return - target_return_daily) * 1e9 

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

    if len(portfolio_values) < 2:
        return 0.0

    valid_mask = portfolio_values[:-1] != 0
    daily_returns = np.full_like(portfolio_values[:-1], fill_value=np.nan, dtype=float)
    if np.any(valid_mask):
         daily_returns[valid_mask] = np.diff(portfolio_values)[valid_mask] / portfolio_values[:-1][valid_mask]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    if len(daily_returns) == 0:
         return 0.0

    target_return = risk_free_rate / periods_per_year
    excess_returns = daily_returns - target_return
    mean_excess_return = np.mean(excess_returns)

    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 1:
        return np.inf if mean_excess_return > 0 else 0.0

    downside_deviation = np.std(downside_returns)

    if downside_deviation < 1e-9: 
         return np.inf if mean_excess_return > 0 else 0.0

    sortino = mean_excess_return / downside_deviation
    annualized_sortino = sortino * np.sqrt(periods_per_year)

    return annualized_sortino if (np.isfinite(annualized_sortino) or (np.isinf(annualized_sortino) and annualized_sortino > 0)) else 0.0


def calculate_chi_score(portfolio_values: np.ndarray | pd.Series,
                        alpha: float = 0.25, 
                        risk_free_rate: float = 0.0,
                        periods_per_year: int = 252) -> float:
    """Calculates the combined Chi score (alpha*Sharpe + (1-alpha)*Sortino)."""
    sharpe = calculate_sharpe_ratio(portfolio_values, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(portfolio_values, risk_free_rate, periods_per_year)

    SORTINO_CAP = 10.0 
    if np.isinf(sortino) and sortino > 0:
        print(f"Warning: Sortino ratio is infinite, capping at {SORTINO_CAP}")
        sortino = SORTINO_CAP
    elif not np.isfinite(sortino):
        sortino = 0.0

    if not np.isfinite(sharpe):
        sharpe = 0.0

    chi = alpha * sharpe + (1.0 - alpha) * sortino
    return chi

