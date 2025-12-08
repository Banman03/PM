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
    R: np.ndarray,
    method: str = 'central',
) -> np.ndarray:
    """
    Compute numerical derivative dliq/dtau via finite differences.

    Parameters
    ----------
    tau : 1D array of time-to-resolution (must be sorted)
    R : 1D array of liquidity values
    method : 'forward', 'backward', or 'central'

    Returns
    -------
    dR_dtau : numerical derivative (same length as tau, with NaNs at boundaries)
    """
    n = len(tau)
    dR = np.full(n, np.nan)

    if method == 'central':
        for i in range(1, n - 1):
            dR[i] = (R[i + 1] - R[i - 1]) / (tau[i + 1] - tau[i - 1])
    elif method == 'forward':
        for i in range(n - 1):
            dR[i] = (R[i + 1] - R[i]) / (tau[i + 1] - tau[i])
    elif method == 'backward':
        for i in range(1, n):
            dR[i] = (R[i] - R[i - 1]) / (tau[i] - tau[i - 1])
    else:
        raise ValueError(f"Unknown method: {method}")

    return dR


def fit_linear_method_a(
    tau: np.ndarray,
    R_obs: np.ndarray,
    I_obs: np.ndarray,
    D_obs: np.ndarray,
    initial_guess: Tuple[float, float, float] = (1e-5, 1e-6, -0.01),
    bounds: Tuple[Tuple, Tuple] = ((0, 0, -10), (np.inf, np.inf, 10)),
) -> Dict:
    """
    Method A: Fit linear spread model by minimizing derivative residuals.

    NEW MODEL: dR/dτ = a·I(τ) - b·D(τ) + c·R(τ)

    min_{a,b,c} Σ [dR/dτ - (a·I - b·D + c·R)]²

    Parameters
    ----------
    tau : 1D array, time-to-resolution
    R_obs : observed spread (DEPENDENT VARIABLE)
    I_obs : observed trading intensity
    D_obs : observed market depth
    initial_guess : (a0, b0, c0)
    bounds : ((a_min, b_min, c_min), (a_max, b_max, c_max))

    Returns
    -------
    result : dict with keys 'params', 'success', 'message', 'residuals', 'cost'
    """
    dR_obs = finite_difference_derivative(tau, R_obs, method='central')

    valid = ~(np.isnan(dR_obs) | np.isnan(R_obs) | np.isnan(I_obs) | np.isnan(D_obs))
    tau_valid = tau[valid]
    dR_valid = dR_obs[valid]
    R_valid = R_obs[valid]
    I_valid = I_obs[valid]
    D_valid = D_obs[valid]

    if len(tau_valid) < 10:
        warnings.warn("Insufficient valid data points for Method A")
        return {
            'params': {'a': np.nan, 'b': np.nan, 'c': np.nan},
            'success': False,
            'message': 'Insufficient data',
            'residuals': np.array([]),
            'cost': np.nan,
        }

    def residuals(theta):
        a, b, c = theta
        predicted_dR = a * I_valid - b * D_valid + c * R_valid
        return dR_valid - predicted_dR

    res = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
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
    R_obs: np.ndarray,
    I_obs: np.ndarray,
    D_obs: np.ndarray,
    initial_guess: Tuple[float, float, float] = (1e-5, 1e-6, -0.01),
    bounds: Tuple[Tuple, Tuple] = ((0, 0, -10), (np.inf, np.inf, 10)),
) -> Dict:
    """
    Method B: Fit linear spread model by trajectory matching (integrate-then-fit).

    NEW MODEL: dR/dτ = a·I(τ) - b·D(τ) + c·R(τ)

    min_{a,b,c} Σ [R_obs(τᵢ) - R_model(τᵢ; a,b,c)]²

    Parameters
    ----------
    tau : 1D array (sorted)
    R_obs : observed spread (DEPENDENT VARIABLE)
    I_obs : observed trading intensity
    D_obs : observed market depth
    initial_guess : (a0, b0, c0)
    bounds : parameter bounds

    Returns
    -------
    result : dict
    """
    valid = ~(np.isnan(R_obs) | np.isnan(I_obs) | np.isnan(D_obs))
    tau_valid = tau[valid]
    R_valid = R_obs[valid]
    I_valid = I_obs[valid]
    D_valid = D_obs[valid]

    if len(tau_valid) < 10:
        warnings.warn("Insufficient valid data for Method B")
        return {
            'params': {'a': np.nan, 'b': np.nan, 'c': np.nan},
            'success': False,
            'message': 'Insufficient data',
            'residuals': np.array([]),
            'cost': np.nan,
        }

    I_func = interp1d(tau_valid, I_valid, kind='linear', fill_value='extrapolate')
    D_func = interp1d(tau_valid, D_valid, kind='linear', fill_value='extrapolate')

    R0 = R_valid[0]

    def residuals(theta):
        a, b, c = theta
        try:
            R_model = integrate_ode_linear(tau_valid, R0, a, b, c, I_func, D_func)
            return R_valid - R_model
        except Exception as e:
            warnings.warn(f"ODE integration failed: {e}")
            return np.full_like(R_valid, 1e6)

    res = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        loss='soft_l1',
        verbose=2,
    )

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
    D_obs: np.ndarray,
    I_obs: np.ndarray,
    R_obs: np.ndarray,
    initial_guess: Tuple = (1e-5, 0.1, -0.01, 0.01, 0.01, 0.0001),
    bounds: Tuple[Tuple, Tuple] = ((0, 0, -10, 0, 1e-6, 0), (np.inf, np.inf, 10, 10, 1.0, 0.1)),
) -> Dict:
    """
    Method B for bounded model.

    Parameters: (a, b, c, d, R_max, R_min)

    Parameters
    ----------
    tau, D_obs, I_obs, R_obs : data
    initial_guess : (a0, b0, c0, d0, R_max0, R_min0)
    bounds : parameter bounds

    Returns
    -------
    result : dict
    """
    valid = ~(np.isnan(D_obs) | np.isnan(I_obs) | np.isnan(R_obs))
    tau_valid = tau[valid]
    D_valid = D_obs[valid]
    I_valid = I_obs[valid]
    R_valid = R_obs[valid]

    if len(tau_valid) < 10:
        warnings.warn("Insufficient valid data for bounded Method B")
        return {
            'params': {'a': np.nan, 'b': np.nan, 'c': np.nan, 'd': np.nan, 'R_max': np.nan, 'r_min': np.nan},
            'success': False,
            'message': 'Insufficient data',
            'residuals': np.array([]),
            'cost': np.nan,
        }

    I_func = interp1d(tau_valid, I_valid, kind='linear', fill_value='extrapolate')
    D_func = interp1d(tau_valid, D_valid, kind='linear', fill_value='extrapolate')

    R0 = R_valid[0]

    def residuals(theta):
        a, b, c, d, R_max, R_min = theta
        try:
            R_model = integrate_ode_bounded(
                tau_valid, R0, a, b, c, d, R_max, R_min, I_func, D_func
            )
            return R_valid - R_model
        except Exception as e:
            warnings.warn(f"Bounded ODE integration failed: {e}")
            return np.full_like(R_valid, 1e6)

    res = least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        loss='soft_l1',
        verbose=2,
        max_nfev=300,
    )

    a_fit, b_fit, c_fit, d_fit, r_max_fit, r_min_fit = res.x

    return {
        'params': {
            'a': a_fit, 'b': b_fit, 'c': c_fit, 'd': d_fit,
            'r_max': r_max_fit, 'r_min': r_min_fit
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

    bounds_lower, bounds_upper = kwargs.get('bounds', ((0,), (1,)))

    for i in range(n_restarts):
        if i == 0:
            print("using an initial guess")
            init_guess = kwargs.get('initial_guess', tuple((l + u) / 2 for l, u in zip(bounds_lower, bounds_upper)))
            print("initial guess: ", init_guess)
        else:
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

    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)

    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood

    return aic, bic


if __name__ == '__main__':
    np.random.seed(0)

    tau = np.linspace(0, 100, 200)
    a_true, b_true, c_true = 0.00001, 0.1, -0.01

    I_true = 50.0 + 10 * np.sin(tau / 20)
    R_true = 0.002 + 0.0005 * np.cos(tau / 15)

    I_func = interp1d(tau, I_true, kind='linear', fill_value='extrapolate')
    R_func = interp1d(tau, R_true, kind='linear', fill_value='extrapolate')

    ell_true = integrate_ode_linear(tau, 0.001, a_true, b_true, c_true, I_func, R_func)

    ell_obs = ell_true + np.random.normal(0, 0.0001, size=len(tau))

    print("Fitting linear model (Method B) to synthetic data...")
    result_b = fit_linear_method_b(tau, ell_obs, I_true, R_true)

    print(f"True parameters: a={a_true}, b={b_true}, c={c_true}")
    print(f"Fitted parameters: a={result_b['params']['a']:.2e}, b={result_b['params']['b']:.2e}, c={result_b['params']['c']:.2e}")
    print(f"Success: {result_b['success']}, Cost: {result_b['cost']:.2e}")

    aic, bic = compute_aic_bic(result_b['residuals'], n_params=3)
    print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")
