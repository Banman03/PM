#!/usr/bin/env python3
"""
validation.py

Model validation, diagnostics, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from typing import Tuple, Dict, List
import warnings

from model import integrate_ode_linear, integrate_ode_bounded


def compute_goodness_of_fit(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute R², RMSE, MAE, MAPE.

    Parameters
    ----------
    y_obs : observed values
    y_pred : predicted values

    Returns
    -------
    metrics : dict with keys 'r2', 'rmse', 'mae', 'mape'
    """
    # Remove NaNs
    valid = ~(np.isnan(y_obs) | np.isnan(y_pred))
    y_obs = y_obs[valid]
    y_pred = y_pred[valid]

    if len(y_obs) == 0:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}

    # Residuals
    residuals = y_obs - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)

    # R²
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))

    # MAE
    mae = np.mean(np.abs(residuals))

    # MAPE (avoid division by zero)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mape = np.mean(np.abs(residuals / (y_obs + 1e-12))) * 100

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
    }


def residual_autocorrelation(residuals: np.ndarray, max_lag: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function of residuals.

    Parameters
    ----------
    residuals : 1D array
    max_lag : maximum lag

    Returns
    -------
    lags : array of lag values
    acf : autocorrelation at each lag
    """
    from statsmodels.tsa.stattools import acf as sm_acf

    # Remove NaNs
    residuals = residuals[~np.isnan(residuals)]

    if len(residuals) < max_lag + 1:
        max_lag = max(1, len(residuals) - 1)

    acf_vals = sm_acf(residuals, nlags=max_lag, fft=True)
    lags = np.arange(max_lag + 1)

    return lags, acf_vals


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    """
    Ljung-Box test for autocorrelation in residuals.

    H0: residuals are independently distributed (no autocorrelation)

    Parameters
    ----------
    residuals : 1D array
    lags : number of lags to test

    Returns
    -------
    statistic : test statistic
    p_value : p-value (reject H0 if p < 0.05)
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    residuals = residuals[~np.isnan(residuals)]

    if len(residuals) < lags + 1:
        return np.nan, np.nan

    result = acorr_ljungbox(residuals, lags=lags, return_df=False)
    # result is (statistic, p_value) for each lag; take last
    statistic = result[0][-1]
    p_value = result[1][-1]

    return statistic, p_value


def shapiro_wilk_test(residuals: np.ndarray) -> Tuple[float, float]:
    """
    Shapiro-Wilk test for normality of residuals.

    H0: residuals are normally distributed

    Parameters
    ----------
    residuals : 1D array

    Returns
    -------
    statistic : W statistic
    p_value : p-value (reject H0 if p < 0.05)
    """
    residuals = residuals[~np.isnan(residuals)]

    if len(residuals) < 3:
        return np.nan, np.nan

    statistic, p_value = stats.shapiro(residuals)
    return statistic, p_value


def breusch_pagan_test(residuals: np.ndarray, fitted: np.ndarray) -> Tuple[float, float]:
    """
    Breusch-Pagan test for heteroskedasticity.

    H0: homoskedastic errors (constant variance)

    Parameters
    ----------
    residuals : 1D array of residuals
    fitted : 1D array of fitted values

    Returns
    -------
    statistic : LM statistic
    p_value : p-value (reject H0 if p < 0.05)
    """
    from statsmodels.stats.diagnostic import het_breuschpagan

    valid = ~(np.isnan(residuals) | np.isnan(fitted))
    residuals = residuals[valid]
    fitted = fitted[valid]

    if len(residuals) < 10:
        return np.nan, np.nan

    # BP test requires exog matrix; use fitted values as single predictor
    exog = np.column_stack([np.ones_like(fitted), fitted])

    try:
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, exog)
        return lm, lm_pvalue
    except Exception as e:
        warnings.warn(f"Breusch-Pagan test failed: {e}")
        return np.nan, np.nan


def run_all_diagnostics(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """
    Run full battery of diagnostic tests.

    Parameters
    ----------
    y_obs : observed values
    y_pred : predicted values

    Returns
    -------
    diagnostics : dict with goodness-of-fit, residual tests, etc.
    """
    # Goodness of fit
    gof = compute_goodness_of_fit(y_obs, y_pred)

    # Residuals
    residuals = y_obs - y_pred

    # Autocorrelation
    lags, acf_vals = residual_autocorrelation(residuals, max_lag=20)
    lb_stat, lb_pval = ljung_box_test(residuals, lags=10)

    # Normality
    sw_stat, sw_pval = shapiro_wilk_test(residuals)

    # Heteroskedasticity
    bp_stat, bp_pval = breusch_pagan_test(residuals, y_pred)

    return {
        'goodness_of_fit': gof,
        'acf_lags': lags,
        'acf_values': acf_vals,
        'ljung_box_statistic': lb_stat,
        'ljung_box_pvalue': lb_pval,
        'shapiro_wilk_statistic': sw_stat,
        'shapiro_wilk_pvalue': sw_pval,
        'breusch_pagan_statistic': bp_stat,
        'breusch_pagan_pvalue': bp_pval,
    }


def train_test_split_temporal(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    time_col: str = 'tau_sec',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train and test sets.

    Parameters
    ----------
    df : DataFrame
    train_fraction : fraction for training (0 < f < 1)
    time_col : column to sort by

    Returns
    -------
    df_train, df_test : train and test DataFrames
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    split_idx = int(n * train_fraction)

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    return df_train, df_test


def bootstrap_parameters(
    fit_func,
    tau: np.ndarray,
    ell_obs: np.ndarray,
    I_obs: np.ndarray,
    R_obs: np.ndarray,
    n_bootstrap: int = 100,
    block_size: int = 10,
    **fit_kwargs,
) -> pd.DataFrame:
    """
    Block-bootstrap parameter estimates to quantify uncertainty.

    Parameters
    ----------
    fit_func : fitting function (e.g., fit_linear_method_b)
    tau, ell_obs, I_obs, R_obs : data arrays
    n_bootstrap : number of bootstrap samples
    block_size : size of temporal blocks for resampling
    **fit_kwargs : additional arguments for fit_func

    Returns
    -------
    bootstrap_results : DataFrame with columns for each parameter, one row per bootstrap sample
    """
    n = len(tau)
    n_blocks = n // block_size

    if n_blocks < 5:
        warnings.warn("Too few blocks for reliable bootstrap; increase data or decrease block_size")

    bootstrap_params = []

    for b in range(n_bootstrap):
        # Randomly sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)

        # Reconstruct time series from blocks
        sample_indices = []
        for bi in block_indices:
            start = bi * block_size
            end = min(start + block_size, n)
            sample_indices.extend(range(start, end))

        sample_indices = sample_indices[:n]  # ensure same length

        tau_boot = tau[sample_indices]
        ell_boot = ell_obs[sample_indices]
        I_boot = I_obs[sample_indices]
        R_boot = R_obs[sample_indices]

        # Sort by tau (bootstrap may disorder)
        sort_idx = np.argsort(tau_boot)
        tau_boot = tau_boot[sort_idx]
        ell_boot = ell_boot[sort_idx]
        I_boot = I_boot[sort_idx]
        R_boot = R_boot[sort_idx]

        # Fit
        try:
            result = fit_func(tau_boot, ell_boot, I_boot, R_boot, **fit_kwargs)
            if result['success']:
                bootstrap_params.append(result['params'])
            else:
                bootstrap_params.append({k: np.nan for k in result['params'].keys()})
        except Exception as e:
            warnings.warn(f"Bootstrap iteration {b} failed: {e}")
            # Append NaNs
            if len(bootstrap_params) > 0:
                bootstrap_params.append({k: np.nan for k in bootstrap_params[0].keys()})

    df_boot = pd.DataFrame(bootstrap_params)
    return df_boot


def compute_parameter_confidence_intervals(
    bootstrap_df: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute percentile confidence intervals from bootstrap distribution.

    Parameters
    ----------
    bootstrap_df : DataFrame of bootstrap parameter samples (from bootstrap_parameters)
    confidence_level : e.g., 0.95 for 95% CI

    Returns
    -------
    ci_df : DataFrame with columns 'parameter', 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    results = []
    for col in bootstrap_df.columns:
        values = bootstrap_df[col].dropna()
        if len(values) == 0:
            results.append({
                'parameter': col,
                'mean': np.nan,
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
            })
        else:
            results.append({
                'parameter': col,
                'mean': values.mean(),
                'std': values.std(),
                'ci_lower': np.percentile(values, lower_percentile),
                'ci_upper': np.percentile(values, upper_percentile),
            })

    return pd.DataFrame(results)


def sensitivity_analysis_trade_size(
    fit_func,
    compute_observables_func,
    csv_path: str,
    jsonl_path: str,
    trade_sizes: List[float],
    **fit_kwargs,
) -> pd.DataFrame:
    """
    Test sensitivity of fitted parameters to choice of simulated trade size Δq.

    Parameters
    ----------
    fit_func : fitting function
    compute_observables_func : function to compute observables given trade_size
    csv_path, jsonl_path : data paths
    trade_sizes : list of trade sizes to test
    **fit_kwargs : kwargs for fit_func

    Returns
    -------
    sensitivity_df : DataFrame with columns 'trade_size', 'a', 'b', 'c', ...
    """
    results = []

    for ts in trade_sizes:
        print(f"  Testing trade_size={ts}...")

        # Recompute observables with this trade_size
        df_obs = compute_observables_func(csv_path, jsonl_path, trade_size=ts)

        # Extract data
        tau = df_obs['tau_sec'].values if 'tau_sec' in df_obs.columns else np.arange(len(df_obs))
        ell = df_obs['ell_avg'].values
        I = df_obs['I'].values
        R = df_obs['R'].values

        # Fit
        try:
            result = fit_func(tau, ell, I, R, **fit_kwargs)
            if result['success']:
                row = {'trade_size': ts, **result['params'], 'cost': result['cost']}
            else:
                row = {'trade_size': ts, **{k: np.nan for k in result['params'].keys()}, 'cost': np.nan}
        except Exception as e:
            warnings.warn(f"Fit failed for trade_size={ts}: {e}")
            row = {'trade_size': ts, 'cost': np.nan}

        results.append(row)

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Example: test diagnostics on synthetic data
    np.random.seed(42)

    y_obs = np.random.randn(100) + 5.0
    y_pred = y_obs + np.random.randn(100) * 0.1  # small noise

    diagnostics = run_all_diagnostics(y_obs, y_pred)

    print("Goodness of fit:")
    for k, v in diagnostics['goodness_of_fit'].items():
        print(f"  {k}: {v:.4f}")

    print(f"\nLjung-Box test: statistic={diagnostics['ljung_box_statistic']:.2f}, p={diagnostics['ljung_box_pvalue']:.4f}")
    print(f"Shapiro-Wilk test: statistic={diagnostics['shapiro_wilk_statistic']:.4f}, p={diagnostics['shapiro_wilk_pvalue']:.4f}")
    print(f"Breusch-Pagan test: statistic={diagnostics['breusch_pagan_statistic']:.2f}, p={diagnostics['breusch_pagan_pvalue']:.4f}")
