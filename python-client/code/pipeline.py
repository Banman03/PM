
"""
pipeline.py

End-to-end pipeline: NDJSON → observables → fit → validate → plots.

Usage:
    python pipeline.py --csv data.csv --jsonl data.jsonl --resolution-time 1800000000000 --output-dir results/
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings

from observables import (
    compute_all_observables,
    convert_to_tau,
    resample_regular_grid,
    smooth_series,
)
from model import integrate_ode_linear, integrate_ode_bounded
from estimation import (
    fit_linear_method_a,
    fit_linear_method_b,
    fit_bounded_method_b,
    fit_with_multiple_restarts,
    compute_aic_bic,
)
from validation import (
    compute_goodness_of_fit,
    run_all_diagnostics,
    train_test_split_temporal,
    bootstrap_parameters,
    compute_parameter_confidence_intervals,
    sensitivity_analysis_trade_size,
)
from plotting import (
    plot_observables_timeseries,
    plot_model_vs_observed,
    plot_scatter_obs_vs_pred,
    plot_residuals,
    plot_residual_diagnostics,
    plot_bootstrap_distributions,
    plot_sensitivity_to_trade_size,
)


def main():
    parser = argparse.ArgumentParser(description='Spread dynamics model pipeline')
    parser.add_argument('--csv', required=True, help='Path to CSV file (from process_clob_data.py)')
    parser.add_argument('--jsonl', required=True, help='Path to JSONL file (original order book snapshots)')
    parser.add_argument('--resolution-time', type=int, required=True,
                        help='Market resolution timestamp (milliseconds since epoch)')
    parser.add_argument('--trade-size', type=float, default=1.0,
                        help='Simulated trade size for bid-ask estimation (default: 1.0 shares)')
    parser.add_argument('--output-dir', default='results/', help='Output directory for plots and results')
    parser.add_argument('--n-grid-points', type=int, default=500, help='Number of points in regular tau grid')
    parser.add_argument('--smooth-window', type=int, default=3, help='Smoothing window size')
    parser.add_argument('--model-type', choices=['linear', 'bounded'], default='linear',
                        help='Model type: linear or bounded')
    parser.add_argument('--n-bootstrap', type=int, default=100, help='Number of bootstrap samples')

    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("SPREAD DYNAMICS MODEL PIPELINE")
    print("=" * 80)




    print("\n[1/7] Computing observables from order book data...")
    print(f"  CSV: {args.csv}")
    print(f"  JSONL: {args.jsonl}")
    print(f"  Trade size: {args.trade_size}")

    df_obs = compute_all_observables(
        args.csv,
        args.jsonl,
        trade_size=args.trade_size,
        normalized_spread=False,
    )

    print(f"  Loaded {len(df_obs)} snapshots")


    df_obs = convert_to_tau(df_obs, args.resolution_time)


    obs_path = os.path.join(args.output_dir, 'observables.csv')
    df_obs.to_csv(obs_path, index=False)
    print(f"  Saved observables to {obs_path}")




    print("\n[2/7] Resampling to regular tau grid and smoothing...")

    df_obs = df_obs.dropna(subset=['R', 'D', 'I']).reset_index(drop=True)

    df_grid = resample_regular_grid(
        df_obs,
        tau_col='tau_sec',
        observables=['R', 'D', 'I'],
        n_points=args.n_grid_points,
    )


    df_grid['R_smooth'] = smooth_series(df_grid['R'], window=args.smooth_window)
    df_grid['D_smooth'] = smooth_series(df_grid['D'], window=args.smooth_window)
    df_grid['I_smooth'] = smooth_series(df_grid['I'], window=args.smooth_window)


    df_grid = df_grid.dropna(subset=['tau_sec', 'R_smooth', 'D_smooth', 'I_smooth']).reset_index(drop=True)

    print(f"  Resampled to {len(df_grid)} points")


    grid_path = os.path.join(args.output_dir, 'grid.csv')
    df_grid.to_csv(grid_path, index=False)
    print(f"  Saved grid to {grid_path}")




    print("\n[3/7] Splitting into train and test sets (70/30)...")

    df_train, df_test = train_test_split_temporal(df_grid, train_fraction=0.7, time_col='tau_sec')

    print(f"  Train: {len(df_train)} points")
    print(f"  Test: {len(df_test)} points")


    tau_train = df_train['tau_sec'].values
    R_train = df_train['R_smooth'].values 
    I_train = df_train['I_smooth'].values  
    D_train = df_train['D_smooth'].values  

    tau_test = df_test['tau_sec'].values
    R_test = df_test['R_smooth'].values   
    I_test = df_test['I_smooth'].values    
    D_test = df_test['D_smooth'].values    




    print(f"\n[4/7] Fitting {args.model_type} model using Method B (trajectory fit)...")

    if args.model_type == 'linear':
    
        initial_guess = (1e-7, 1e-8, -0.001) 
        bounds = ((0, 0, -10), (1e-4, 1e-6, 10)) 

        result = fit_with_multiple_restarts(
            fit_linear_method_b,
            n_restarts=5,
            tau=tau_train,
            R_obs=R_train, 
            I_obs=I_train, 
            D_obs=D_train, 
            initial_guess=initial_guess,
            bounds=bounds,
        )

    elif args.model_type == 'bounded':
    
        R_max_guess = np.nanmax(R_train) * 1.5
        R_min_guess = np.nanmin(R_train) * 0.5

        initial_guess = (1e-7, 1e-8, -0.001, 0.01, R_max_guess, R_min_guess)
        bounds = ((0, 0, -10, 0, R_max_guess * 0.5, 0), (1e-4, 1e-6, 10, 10, R_max_guess * 2, R_min_guess * 2))

        result = fit_with_multiple_restarts(
            fit_bounded_method_b,
            n_restarts=3,
            tau=tau_train,
            D_obs=D_train,
            I_obs=I_train,
            R_obs=R_train, 
            initial_guess=initial_guess,
            bounds=bounds,
        )

    print(f"  Optimization success: {result['success']}")
    print(f"  Cost: {result['cost']:.4e}")
    print("  Fitted parameters:")
    for k, v in result['params'].items():
        print(f"    {k} = {v:.6e}")


    n_params = len(result['params'])
    aic, bic = compute_aic_bic(result['residuals'], n_params)
    print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")


    params_path = os.path.join(args.output_dir, 'fitted_parameters.txt')
    with open(params_path, 'w') as f:
        f.write("Fitted Parameters\n")
        f.write("=" * 40 + "\n")
        for k, v in result['params'].items():
            f.write(f"{k} = {v:.6e}\n")
        f.write(f"\nAIC = {aic:.2f}\n")
        f.write(f"BIC = {bic:.2f}\n")
    print(f"  Saved parameters to {params_path}")




    print("\n[5/7] Generating model predictions...")


    I_train_func = interp1d(tau_train, I_train, kind='linear', fill_value='extrapolate')
    D_train_func = interp1d(tau_train, D_train, kind='linear', fill_value='extrapolate')

    I_test_func = interp1d(tau_test, I_test, kind='linear', fill_value='extrapolate')
    D_test_func = interp1d(tau_test, D_test, kind='linear', fill_value='extrapolate')

    R0 = R_train[0] 

    diffs = np.diff(tau_train)
    print("min |Δτ|:", np.min(np.abs(diffs)))
    print("any duplicates:", np.any(diffs == 0))
    print("monotone increasing:", np.all(diffs > 0))
    print("monotone decreasing:", np.all(diffs < 0))


    if args.model_type == 'linear':
        R_train_pred = integrate_ode_linear(
            tau_train, R0,
            result['params']['a'], result['params']['b'], result['params']['c'],
            I_train_func, D_train_func
        )
        R_test_pred = integrate_ode_linear(
            tau_test, R0,
            result['params']['a'], result['params']['b'], result['params']['c'],
            I_test_func, D_test_func
        )
    else:
        R_train_pred = integrate_ode_bounded(
            tau_train, R0,
            result['params']['a'], result['params']['b'], result['params']['c'], result['params']['d'],
            result['params']['R_max'], result['params']['R_min'],
            I_train_func, D_train_func
        )
        R_test_pred = integrate_ode_bounded(
            tau_test, R0,
            result['params']['a'], result['params']['b'], result['params']['c'], result['params']['d'],
            result['params']['R_max'], result['params']['R_min'],
            I_test_func, D_test_func
        )


    gof_train = compute_goodness_of_fit(R_train, R_train_pred)
    gof_test = compute_goodness_of_fit(R_test, R_test_pred)

    print("  Train set metrics:")
    for k, v in gof_train.items():
        print(f"    {k}: {v:.4f}")

    print("  Test set metrics:")
    for k, v in gof_test.items():
        print(f"    {k}: {v:.4f}")




    print("\n[6/7] Running diagnostics...")

    diagnostics = run_all_diagnostics(R_train, R_train_pred)

    print(f"  Ljung-Box test (autocorrelation): stat={diagnostics['ljung_box_statistic']:.2f}, p={diagnostics['ljung_box_pvalue']:.4f}")
    print(f"  Shapiro-Wilk test (normality): stat={diagnostics['shapiro_wilk_statistic']:.4f}, p={diagnostics['shapiro_wilk_pvalue']:.4f}")
    print(f"  Breusch-Pagan test (heteroskedasticity): stat={diagnostics['breusch_pagan_statistic']:.2f}, p={diagnostics['breusch_pagan_pvalue']:.4f}")
    print("\n[7/7] Generating plots...")


    plot_observables_timeseries(
        df_grid['tau_sec'].values,
        df_grid['R_smooth'].values,
        df_grid['D_smooth'].values,
        df_grid['I_smooth'].values,
        save_path=os.path.join(args.output_dir, 'fig_observables_timeseries.png')
    )


    plot_model_vs_observed(
        tau_train, R_train, R_train_pred,
        tau_test, R_test, R_test_pred,
        save_path=os.path.join(args.output_dir, 'fig_model_vs_observed.png')
    )


    plot_scatter_obs_vs_pred(
        np.concatenate([R_train, R_test]),
        np.concatenate([R_train_pred, R_test_pred]),
        save_path=os.path.join(args.output_dir, 'fig_scatter_obs_vs_pred.png')
    )


    residuals_train = R_train - R_train_pred
    plot_residuals(
        tau_train, residuals_train,
        save_path=os.path.join(args.output_dir, 'fig_residuals.png')
    )


    plot_residual_diagnostics(
        residuals_train, R_train_pred,
        diagnostics['acf_lags'], diagnostics['acf_values'],
        save_path=os.path.join(args.output_dir, 'fig_residual_diagnostics.png')
    )


    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print(f"All results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
