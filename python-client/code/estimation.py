#!/usr/bin/env python3
"""
estimation.py

Parameter estimation for liquidity ODE models.
Implements both Method A (derivative-residual) and Method B (trajectory fit).
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Callable
import warnings

from model import integrate_ode_linear, integrate_ode_bounded


def finite_difference_derivative(
    tau: np.ndarray,
    ell: np.ndarray,
    method: str = 'central',
) -> np.ndarray:
    """
    Compute numerical derivative dℓ/dτ via finite differences.

    Parameters
    ----------
    tau : 1D array of time-to-resolution (must be sorted)
    ell : 1D array of liquidity values
    method : 'forward', 'backward', or 'central'

    Returns
    -------
    dell_dtau : numerical derivative (same length as tau, with NaNs at boundaries)
    """
    n = len(tau)
    dell = np.full(n, np.nan)

    if method == 'central':
        for i in range(1, n - 1):
            dell[i] = (ell[i + 1] - ell[i - 1]) / (tau[i + 1] - tau[i - 1])
    elif method == 'forward':
        for i in range(n - 1):
            dell[i] = (ell[i + 1] - ell[i]) / (tau[i + 1] - tau[i])
    elif method == 'backward':
        for i in range(1, n):
            dell[i] = (ell[i] - ell[i - 1]) / (tau[i] - tau[i - 1])
    else:
        raise ValueError(f"Unknown method: {method}")

    return dell


def fit_linear_method_a(
    tau: np.ndarray,
    ell_obs: np.ndarray,
    I_obs: np.ndarray,
    R_obs: np.ndarray,
    initial_guess: Tuple[float, float, float] = (1e-5, 0.1, -0.01),
    bounds: Tuple[Tuple, Tuple] = ((0, 0, -10), (np.inf, np.inf, 10)),
) -> Dict:
    """
    Method A: Fit linear model by minimizing derivative residuals.

    min_{a,b,c} Σ [dℓ/dτ - (a·I - b·R + c·ell)]²

    Parameters
    ----------
    tau : 1D array, time-to-resolution
    ell_obs : observed liquidity
    I_obs : observed intensity
    R_obs : observed spread
    initial_guess : (a0, b0, c0)
    bounds : ((a_min, b_min, c_min), (a_max, b_max, c_max))

    Returns
    -------
    result : dict with keys 'params', 'success', 'message', 'residuals', 'cost'
    """
    # Compute derivative
    dell_obs = finite_difference_derivative(tau, ell_obs, method='central')

    # Filter out NaNs
    valid = ~(np.isnan(dell_obs) | np.isnan(ell_obs) | np.isnan(I_obs) | np.isnan(R_obs))
    tau_valid = tau[valid]
    dell_valid = dell_obs[valid]
    ell_valid = ell_obs[valid]
    I_valid = I_obs[valid]
    R_valid = R_obs[valid]

    if len(tau_valid) < 10:
        warnings.warn("Insufficient valid data points for Method A")
        return {
            'params': {'a': np.nan, 'b': np.nan, 'c': np.nan},
            'success': False,
            'message': 'Insufficient data',
            'residuals': np.array([]),
            'cost': np.nan,
        }

    # Residual function
    def residuals(theta):
        a, b, c = theta
        predicted_dell = a * I_valid - b * R_valid + c * ell_valid
        return dell_valid - predicted_dell

    # Optimize
    res = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        loss='soft_l1',  # robust to outliers
        verbose=0,
    )

    a_fit, b_fit, c_fit = res.x

    return {
        'params': {'a': a_fit, 'b': b_fit, 'c': c_fit},
        'success': res.success,
        'message': res.message,
        'residuals': res.fun,
        'cost': res.cost,
    }


def fit_linear_method_b(
    tau: np.ndarray,
    ell_obs: np.ndarray,
    I_obs: np.ndarray,
    R_obs: np.ndarray,
    initial_guess: Tuple[float, float, float] = (1e-5, 0.1, -0.01),
    bounds: Tuple[Tuple, Tuple] = ((0, 0, -10), (np.inf, np.inf, 10)),
) -> Dict:
    """
    Method B: Fit linear model by trajectory matching (integrate-then-fit).

    min_{a,b,c} Σ [ell_obs(τᵢ) - ell_model(τᵢ; a,b,c)]²

    Parameters
    ----------
    tau : 1D array (sorted)
    ell_obs : observed liquidity
    I_obs, R_obs : observed inputs
    initial_guess : (a0, b0, c0)
    bounds : parameter bounds

    Returns
    -------
    result : dict
    """
    # Filter NaNs
    valid = ~(np.isnan(ell_obs) | np.isnan(I_obs) | np.isnan(R_obs))
    tau_valid = tau[valid]
    ell_valid = ell_obs[valid]
    I_valid = I_obs[valid]
    R_valid = R_obs[valid]

    if len(tau_valid) < 10:
        warnings.warn("Insufficient valid data for Method B")
        return {
            'params': {'a': np.nan, 'b': np.nan, 'c': np.nan},
            'success': False,
            'message': 'Insufficient data',
            'residuals': np.array([]),
            'cost': np.nan,
        }

    # Create interpolators for I and R
    I_func = interp1d(tau_valid, I_valid, kind='linear', fill_value='extrapolate')
    R_func = interp1d(tau_valid, R_valid, kind='linear', fill_value='extrapolate')
    
    print("moved past interpolation")

    # Initial condition
    ell0 = ell_valid[0]

    # Residual function: integrate ODE and compare
    def residuals(theta):
        a, b, c = theta
        try:
            ell_model = integrate_ode_linear(tau_valid, ell0, a, b, c, I_func, R_func)
            return ell_valid - ell_model
        except Exception as e:
            # If integration fails, return large residuals
            warnings.warn(f"ODE integration failed: {e}")
            return np.full_like(ell_valid, 1e6)

    # Optimize
    res = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        loss='soft_l1',
        verbose=2,
        max_nfev=200,  # limit function evaluations
    )
    print("got past least squares")
    a_fit, b_fit, c_fit = res.x

    return {
        'params': {'a': a_fit, 'b': b_fit, 'c': c_fit},
        'success': res.success,
        'message': res.message,
        'residuals': res.fun,
        'cost': res.cost,
    }


def fit_bounded_method_b(
    tau: np.ndarray,
    ell_obs: np.ndarray,
    I_obs: np.ndarray,
    R_obs: np.ndarray,
    initial_guess: Tuple = (1e-5, 0.1, -0.01, 0.01, 0.01, 0.0001),
    bounds: Tuple[Tuple, Tuple] = ((0, 0, -10, 0, 1e-6, 0), (np.inf, np.inf, 10, 10, 1.0, 0.1)),
) -> Dict:
    """
    Method B for bounded model.

    Parameters: (a, b, c, d, ell_max, ell_min)

    Parameters
    ----------
    tau, ell_obs, I_obs, R_obs : data
    initial_guess : (a0, b0, c0, d0, ell_max0, ell_min0)
    bounds : parameter bounds

    Returns
    -------
    result : dict
    """
    # Filter NaNs
    valid = ~(np.isnan(ell_obs) | np.isnan(I_obs) | np.isnan(R_obs))
    tau_valid = tau[valid]
    ell_valid = ell_obs[valid]
    I_valid = I_obs[valid]
    R_valid = R_obs[valid]

    if len(tau_valid) < 10:
        warnings.warn("Insufficient valid data for bounded Method B")
        return {
            'params': {'a': np.nan, 'b': np.nan, 'c': np.nan, 'd': np.nan, 'ell_max': np.nan, 'ell_min': np.nan},
            'success': False,
            'message': 'Insufficient data',
            'residuals': np.array([]),
            'cost': np.nan,
        }

    I_func = interp1d(tau_valid, I_valid, kind='linear', fill_value='extrapolate')
    R_func = interp1d(tau_valid, R_valid, kind='linear', fill_value='extrapolate')

    ell0 = ell_valid[0]

    def residuals(theta):
        a, b, c, d, ell_max, ell_min = theta
        try:
            ell_model = integrate_ode_bounded(
                tau_valid, ell0, a, b, c, d, ell_max, ell_min, I_func, R_func
            )
            return ell_valid - ell_model
        except Exception as e:
            warnings.warn(f"Bounded ODE integration failed: {e}")
            return np.full_like(ell_valid, 1e6)

    res = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        loss='soft_l1',
        verbose=0,
        max_nfev=300,
    )

    a_fit, b_fit, c_fit, d_fit, ell_max_fit, ell_min_fit = res.x

    return {
        'params': {
            'a': a_fit, 'b': b_fit, 'c': c_fit, 'd': d_fit,
            'ell_max': ell_max_fit, 'ell_min': ell_min_fit
        },
        'success': res.success,
        'message': res.message,
        'residuals': res.fun,
        'cost': res.cost,
    }


def fit_with_multiple_restarts(
    fit_func: Callable,
    n_restarts: int = 5,
    **kwargs,
) -> Dict:
    """
    Run fitting function with multiple random initializations and return best result.

    Parameters
    ----------
    fit_func : fitting function (e.g., fit_linear_method_b)
    n_restarts : number of random restarts
    **kwargs : arguments to pass to fit_func (must include 'bounds' for randomization)

    Returns
    -------
    best_result : dict with best (lowest cost) fit
    """
    best_result = None
    best_cost = np.inf

    # Extract bounds for randomization
    bounds_lower, bounds_upper = kwargs.get('bounds', ((0,), (1,)))

    for i in range(n_restarts):
        if i == 0:
            # First iteration: use provided initial_guess
            print("using an initial guess")
            init_guess = kwargs.get('initial_guess', tuple((l + u) / 2 for l, u in zip(bounds_lower, bounds_upper)))
            print("initial guess: ", init_guess)
        else:
            # Random initialization within bounds
            init_guess = tuple(
                np.random.uniform(low, high) for low, high in zip(bounds_lower, bounds_upper)
            )

        kwargs_copy = kwargs.copy()
        kwargs_copy['initial_guess'] = init_guess

        result = fit_func(**kwargs_copy)

        if result['success'] and result['cost'] < best_cost:
            best_cost = result['cost']
            best_result = result

    if best_result is None:
        # All attempts failed; return last result
        warnings.warn("All restarts failed; returning last attempt")
        best_result = result

    return best_result


def compute_aic_bic(residuals: np.ndarray, n_params: int) -> Tuple[float, float]:
    """
    Compute Akaike and Bayesian Information Criteria.

    AIC = 2k - 2ln(L)
    BIC = k·ln(n) - 2ln(L)

    Assuming Gaussian errors, -2ln(L) ≈ n·ln(RSS/n)

    Parameters
    ----------
    residuals : array of residuals
    n_params : number of fitted parameters

    Returns
    -------
    AIC, BIC : float
    """
    n = len(residuals)
    rss = np.sum(residuals ** 2)

    if rss <= 0 or n <= n_params:
        return np.inf, np.inf

    # Log-likelihood (assuming Gaussian)
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)

    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood

    return aic, bic


if __name__ == '__main__':
    # Example: fit synthetic data
    np.random.seed(42)

    tau = np.linspace(0, 100, 200)
    a_true, b_true, c_true = 0.00001, 0.1, -0.01

    I_true = 50.0 + 10 * np.sin(tau / 20)
    R_true = 0.002 + 0.0005 * np.cos(tau / 15)

    I_func = interp1d(tau, I_true, kind='linear', fill_value='extrapolate')
    R_func = interp1d(tau, R_true, kind='linear', fill_value='extrapolate')

    ell_true = integrate_ode_linear(tau, 0.001, a_true, b_true, c_true, I_func, R_func)

    # Add noise
    ell_obs = ell_true + np.random.normal(0, 0.0001, size=len(tau))

    # Fit using Method B
    print("Fitting linear model (Method B) to synthetic data...")
    result_b = fit_linear_method_b(tau, ell_obs, I_true, R_true)

    print(f"True parameters: a={a_true}, b={b_true}, c={c_true}")
    print(f"Fitted parameters: a={result_b['params']['a']:.2e}, b={result_b['params']['b']:.2e}, c={result_b['params']['c']:.2e}")
    print(f"Success: {result_b['success']}, Cost: {result_b['cost']:.2e}")

    aic, bic = compute_aic_bic(result_b['residuals'], n_params=3)
    print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")
