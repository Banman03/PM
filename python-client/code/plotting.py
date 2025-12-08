
"""
plotting.py

Generate publication-quality plots for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from scipy import stats


sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


def plot_observables_timeseries(
    tau: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot time series of observables R(tau), liq(tau), I(tau).

    Parameters
    ----------
    tau : time-to-resolution (seconds or hours)
    R, D, I : observable arrays
    save_path : if provided, save figure to this path
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    
    axes[0].plot(tau, R, 'b-', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel(r'Spread $R(\tau)$', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    
    axes[1].plot(tau, D, 'g-', linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel(r'Liquidity $\D(\tau)$ ($/share)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    
    axes[2].plot(tau, I, 'r-', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel(r'Intensity $I(\tau)$ (shares/s)', fontsize=12)
    axes[2].set_xlabel(r'Time to resolution $\tau$ (seconds)', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved observables time series to {save_path}")

    return fig


def plot_model_vs_observed(
    tau: np.ndarray,
    R_obs: np.ndarray,
    R_model: np.ndarray,
    tau_test: Optional[np.ndarray] = None,
    R_obs_test: Optional[np.ndarray] = None,
    R_model_test: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """
    Plot observed vs. model-predicted liquidity.

    Parameters
    ----------
    tau : time-to-resolution (train)
    R_obs : observed liquidity (train)
    R_model : model prediction (train)
    tau_test, R_obs_test, R_model_test : test set data (optional)
    save_path : save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    
    ax.plot(tau, R_obs, 'o', color='blue', alpha=0.5, markersize=4, label='Observed (train)')
    ax.plot(tau, R_model, '-', color='red', linewidth=2, label='Model (train)')

    
    if tau_test is not None and R_obs_test is not None:
        ax.plot(tau_test, R_obs_test, 's', color='cyan', alpha=0.5, markersize=4, label='Observed (test)')
        if R_model_test is not None:
            ax.plot(tau_test, R_model_test, '--', color='orange', linewidth=2, label='Model (test)')

    ax.set_xlabel(r'Time to resolution $\tau$ (seconds)', fontsize=12)
    ax.set_ylabel(r'Bid-ask spread', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved model vs. observed plot to {save_path}")

    return fig


def plot_scatter_obs_vs_pred(
    R_obs: np.ndarray,
    R_pred: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Scatter plot of observed vs. predicted with 45° reference line.

    Parameters
    ----------
    R_obs : observed
    R_pred : predicted
    save_path : save path
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    
    valid = ~(np.isnan(R_obs) | np.isnan(R_pred))
    R_obs = R_obs[valid]
    R_pred = R_pred[valid]

    ax.scatter(R_obs, R_pred, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')

    ax.set_xlabel(r'Observed Spread ($)', fontsize=12)
    ax.set_ylabel(r'Predicted Spread ($)', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved scatter plot to {save_path}")

    return fig


def plot_residuals(
    tau: np.ndarray,
    residuals: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot residuals vs. time.

    Parameters
    ----------
    tau : time-to-resolution
    residuals : residuals (observed - predicted)
    save_path : save path
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(tau, residuals, 'o', color='black', alpha=0.5, markersize=3)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    ax.set_xlabel(r'Time to resolution $\tau$ (seconds)', fontsize=12)
    ax.set_ylabel(r'Residuals ($R_{obs} - R_{model}$)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved residuals plot to {save_path}")

    return fig


def plot_residual_diagnostics(
    residuals: np.ndarray,
    fitted: np.ndarray,
    acf_lags: np.ndarray,
    acf_values: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Four-panel residual diagnostic plot: histogram, Q-Q, ACF, residuals vs. fitted.

    Parameters
    ----------
    residuals : residuals
    fitted : fitted values
    acf_lags, acf_values : autocorrelation function
    save_path : save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    
    valid = ~np.isnan(residuals)
    residuals_clean = residuals[valid]
    fitted_clean = fitted[valid] if fitted is not None else np.arange(len(residuals_clean))

    
    ax = axes[0, 0]
    ax.hist(residuals_clean, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    mu, sigma = residuals_clean.mean(), residuals_clean.std()
    x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    ax.set_xlabel('Residuals', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Histogram of Residuals', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    
    ax = axes[0, 1]
    stats.probplot(residuals_clean, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot', fontsize=12)
    ax.grid(True, alpha=0.3)

    
    ax = axes[1, 0]
    ax.stem(acf_lags, acf_values, basefmt=" ", linefmt='blue', markerfmt='bo')
    ax.axhline(0, color='black', linewidth=1)
    
    n = len(residuals_clean)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color='red', linestyle='--', linewidth=1, label='95% CI')
    ax.axhline(-ci, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag', fontsize=11)
    ax.set_ylabel('ACF', fontsize=11)
    ax.set_title('Autocorrelation Function', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    
    ax = axes[1, 1]
    ax.scatter(fitted_clean, residuals_clean, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted values', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals vs. Fitted', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved residual diagnostics to {save_path}")

    return fig


def plot_bootstrap_distributions(
    bootstrap_df: pd.DataFrame,
    true_params: Optional[dict] = None,
    save_path: Optional[str] = None,
):
    """
    Plot histograms of bootstrap parameter distributions.

    Parameters
    ----------
    bootstrap_df : DataFrame with bootstrap parameter samples
    true_params : dict of true parameter values (optional, for synthetic validation)
    save_path : save path
    """
    n_params = len(bootstrap_df.columns)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    for i, col in enumerate(bootstrap_df.columns):
        ax = axes[i]
        values = bootstrap_df[col].dropna()

        if len(values) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col, fontsize=12)
            continue

        ax.hist(values, bins=30, alpha=0.6, color='blue', edgecolor='black')

        
        mean = values.mean()
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)

        ax.axvline(mean, color='red', linewidth=2, label=f'Mean: {mean:.3e}')
        ax.axvline(ci_lower, color='orange', linestyle='--', linewidth=1.5)
        ax.axvline(ci_upper, color='orange', linestyle='--', linewidth=1.5, label=f'95% CI: [{ci_lower:.2e}, {ci_upper:.2e}]')

        
        if true_params and col in true_params:
            ax.axvline(true_params[col], color='green', linewidth=2, linestyle=':', label=f'True: {true_params[col]:.3e}')

        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Bootstrap: {col}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved bootstrap distributions to {save_path}")

    return fig


def plot_sensitivity_to_trade_size(
    sensitivity_df: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Plot how fitted parameters vary with trade size Δq.

    Parameters
    ----------
    sensitivity_df : DataFrame with columns 'trade_size', 'a', 'b', 'c', ...
    save_path : save path
    """
    param_cols = [col for col in sensitivity_df.columns if col not in ['trade_size', 'cost']]
    n_params = len(param_cols)

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))

    if n_params == 1:
        axes = [axes]

    for i, param in enumerate(param_cols):
        ax = axes[i]
        ax.plot(sensitivity_df['trade_size'], sensitivity_df[param], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(r'Trade size $\Delta q$ (shares)', fontsize=11)
        ax.set_ylabel(param, fontsize=11)
        ax.set_title(f'Sensitivity of {param}', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved sensitivity plot to {save_path}")

    return fig


if __name__ == '__main__':
    
    np.random.seed(42)

    tau = np.linspace(0, 100, 200)
    R = 0.002 + 0.0005 * np.sin(tau / 10)
    ell = 0.001 + 0.0002 * np.exp(-tau / 50)
    I = 50 + 10 * np.random.randn(200)

    ell_obs = ell + np.random.normal(0, 0.00005, size=200)
    ell_model = ell + np.random.normal(0, 0.00002, size=200)

    residuals = ell_obs - ell_model

    
    plot_observables_timeseries(tau, R, ell, I, save_path='test_observables.png')

    
    plot_model_vs_observed(tau, ell_obs, ell_model, save_path='test_model_vs_obs.png')

    
    plot_scatter_obs_vs_pred(ell_obs, ell_model, save_path='test_scatter.png')

    
    plot_residuals(tau, residuals, save_path='test_residuals.png')

    print("Test plots generated successfully!")
