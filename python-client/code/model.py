#!/usr/bin/env python3
"""
model.py

Define the liquidity ODE models and integration routines.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple


def ode_linear(tau, R, a, b, c, I_func, D_func):
    """
    Linear spread ODE: dR/dτ = a·I(τ) - b·D(τ) + c·R(τ)

    NEW MODEL: We model spread dynamics instead of liquidity.

    Parameters
    ----------
    tau : float, current time-to-resolution
    R : float, current spread level
    a, b, c : model parameters
    I_func : callable, I_func(tau) returns trading intensity at tau
    D_func : callable, D_func(tau) returns market depth at tau

    Returns
    -------
    dR_dtau : float, derivative dR/dτ
    """
    I_val = I_func(tau)
    D_val = D_func(tau)
    return a * I_val - b * D_val + c * R


def ode_bounded(tau, ell, a, b, c, d, ell_max, ell_min, I_func, R_func):
    """
    Bounded liquidity ODE with logistic saturation and floor penalty:

    dliq/dtau = (a·I - b·R) + c·liq·(1 - liq/liq_max) - d·max{liq_min - liq, 0}

    Parameters
    ----------
    tau : float
    ell : float
    a, b, c, d : model parameters
    ell_max : upper bound (carrying capacity)
    ell_min : lower bound (floor)
    I_func, R_func : callables returning I(tau) and R(tau)

    Returns
    -------
    dell_dtau : float
    """
    I_val = I_func(tau)
    R_val = R_func(tau)

    # External forcing
    forcing = a * I_val - b * R_val

    # Logistic growth term
    logistic = c * ell * (1.0 - ell / ell_max) if ell_max > 0 else c * ell

    # Floor penalty
    floor_penalty = d * max(ell_min - ell, 0.0)

    return forcing + logistic - floor_penalty


def integrate_ode_linear(
    tau_grid: np.ndarray,
    R0: float,
    a: float,
    b: float,
    c: float,
    I_func: Callable,
    D_func: Callable,
) -> np.ndarray:
    """
    Integrate linear spread ODE over tau_grid with initial condition R0.

    dR/dτ = a·I(τ) - b·D(τ) + c·R(τ)

    Parameters
    ----------
    tau_grid : 1D array of tau values (must be sorted ascending or descending)
    R0 : initial spread R(tau_grid[0])
    a, b, c : model parameters
    I_func : callable, I_func(tau) -> float (trading intensity)
    D_func : callable, D_func(tau) -> float (market depth)

    Returns
    -------
    R_traj : 1D array of R(tau) values at each tau_grid point
    """
    def rhs(tau, R):
        val = np.array([ ode_linear(tau, R[0], a, b, c, I_func, D_func) ], dtype=float)
        # if not np.isfinite(val):
            # print("NONFINITE RHS:", tau, R[0], val,
            # I_func(tau), D_func(tau))
        return val

    # solve_ivp expects t_span and t_eval
    tau_span = (tau_grid[0], tau_grid[-1])

    sol = solve_ivp(
        fun=rhs,
        t_span=(tau_grid[0], tau_grid[-1]),
        y0=[R0],
        t_eval=tau_grid,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
        max_step=10.0
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return sol.y[0]


def integrate_ode_bounded(
    tau_grid: np.ndarray,
    R0: float,
    a: float,
    b: float,
    c: float,
    d: float,
    R_max: float,
    R_min: float,
    I_func: Callable,
    R_func: Callable,
) -> np.ndarray:
    """
    Integrate bounded ODE over tau_grid.

    Parameters
    ----------
    tau_grid : 1D array of tau values
    R0 : initial liquidity
    a, b, c, d : model parameters
    R_max, R_min : bounds
    I_func, R_func : callables

    Returns
    -------
    R_traj : 1D array of liq(tau)
    """
    def rhs(tau, R):
        return ode_bounded(tau, R[0], a, b, c, d, R_max, R_min, I_func, R_func)

    tau_span = (tau_grid[0], tau_grid[-1])

    sol = solve_ivp(
        rhs,
        tau_span,
        [R0],
        t_eval=tau_grid,
        method='RK45',
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return sol.y[0]


def analytic_solution_linear_constant_forcing(
    tau_grid: np.ndarray,
    ell0: float,
    a: float,
    b: float,
    c: float,
    I_const: float,
    R_const: float,
) -> np.ndarray:
    """
    Analytic solution for linear ODE with constant I and R.

    liq(tau) = exp(c·tau) · [liq₀ + (a·I - b·R)/c · (exp(-c·tau) - 1)]  if c ≠ 0
         = liq₀ + (a·I - b·R)·tau                                  if c = 0

    Parameters
    ----------
    tau_grid : 1D array
    ell0 : initial condition
    a, b, c : parameters
    I_const : constant I
    R_const : constant R

    Returns
    -------
    ell_traj : liq(tau) at each tau
    """
    tau = tau_grid
    forcing = a * I_const - b * R_const

    if abs(c) < 1e-12:
        # Linear growth
        return ell0 + forcing * tau
    else:
        # Exponential integrating factor
        exp_c_tau = np.exp(c * tau)
        integral_term = forcing / c * (1.0 - np.exp(-c * tau))
        return exp_c_tau * (ell0 + integral_term)


class LiquidityModel:
    """
    Wrapper class for liquidity models with convenient fit/predict interface.
    """

    def __init__(self, model_type: str = 'linear'):
        """
        Parameters
        ----------
        model_type : 'linear' or 'bounded'
        """
        self.model_type = model_type
        self.params = {}

    def set_params(self, **kwargs):
        """Set model parameters: a, b, c, [d, ell_max, ell_min]"""
        self.params.update(kwargs)

    def predict(
        self,
        tau_grid: np.ndarray,
        ell0: float,
        I_func: Callable,
        R_func: Callable,
    ) -> np.ndarray:
        """
        Predict liq(tau) trajectory given parameters and input functions.

        Parameters
        ----------
        tau_grid : time-to-resolution grid
        ell0 : initial condition
        I_func : callable I(tau)
        R_func : callable R(tau)

        Returns
        -------
        ell_traj : predicted liquidity trajectory
        """
        if self.model_type == 'linear':
            return integrate_ode_linear(
                tau_grid, ell0,
                self.params['a'], self.params['b'], self.params['c'],
                I_func, R_func
            )
        elif self.model_type == 'bounded':
            return integrate_ode_bounded(
                tau_grid, ell0,
                self.params['a'], self.params['b'], self.params['c'], self.params['d'],
                self.params['ell_max'], self.params['ell_min'],
                I_func, R_func
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


if __name__ == '__main__':
    # Example: test integration with dummy constant inputs
    tau = np.linspace(0, 100, 500)
    ell0 = 0.001
    a, b, c = 0.00001, 0.1, -0.01
    I_const, R_const = 50.0, 0.002

    I_func = lambda t: I_const
    R_func = lambda t: R_const

    ell_numeric = integrate_ode_linear(tau, ell0, a, b, c, I_func, R_func)
    ell_analytic = analytic_solution_linear_constant_forcing(tau, ell0, a, b, c, I_const, R_const)

    print("Numeric vs. Analytic solution (constant forcing):")
    print(f"  Max difference: {np.max(np.abs(ell_numeric - ell_analytic)):.2e}")
    print(f"  Relative error: {np.max(np.abs(ell_numeric - ell_analytic) / (np.abs(ell_analytic) + 1e-12)):.2e}")
    print("  Integration successful!" if np.allclose(ell_numeric, ell_analytic, rtol=1e-4) else "  WARNING: mismatch")
