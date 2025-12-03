#!/usr/bin/env python3
"""
parameter_stats.py

Aggregate fitted parameters from multiple markets, compute statistics, and create visualizations.

Usage:
    python parameter_stats.py --results-dir results/ --output-dir parameter_analysis/
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_fitted_parameters_file(filepath):
    """
    Parse a fitted_parameters.txt file to extract a, b, c, AIC, BIC.

    Parameters
    ----------
    filepath : str or Path
        Path to fitted_parameters.txt file

    Returns
    -------
    params : dict
        Dictionary with keys 'a', 'b', 'c', 'AIC', 'BIC', 'market_name'
    """
    params = {}

    # Extract market name from parent directory
    market_name = Path(filepath).parent.name
    # Clean up the market name (remove the long hash suffix)
    market_name_clean = re.sub(r'_\d{70,}$', '', market_name).replace('_', ' ')
    params['market_name'] = market_name_clean

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Match "a = value" format
            match_a = re.match(r'a\s*=\s*([0-9.eE+-]+)', line)
            if match_a:
                params['a'] = float(match_a.group(1))

            match_b = re.match(r'b\s*=\s*([0-9.eE+-]+)', line)
            if match_b:
                params['b'] = float(match_b.group(1))

            match_c = re.match(r'c\s*=\s*([0-9.eE+-]+)', line)
            if match_c:
                params['c'] = float(match_c.group(1))

            match_aic = re.match(r'AIC\s*=\s*([0-9.eE+-]+)', line)
            if match_aic:
                params['AIC'] = float(match_aic.group(1))

            match_bic = re.match(r'BIC\s*=\s*([0-9.eE+-]+)', line)
            if match_bic:
                params['BIC'] = float(match_bic.group(1))

    return params


def collect_all_parameters(results_dir):
    """
    Recursively find and parse all fitted_parameters.txt files.

    Parameters
    ----------
    results_dir : str or Path
        Root directory containing market result subdirectories

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: market_name, a, b, c, AIC, BIC
    """
    results_dir = Path(results_dir)
    param_files = list(results_dir.glob('**/fitted_parameters.txt'))

    print(f"Found {len(param_files)} fitted parameter files")

    all_params = []
    for pf in param_files:
        try:
            params = parse_fitted_parameters_file(pf)
            all_params.append(params)
            print(f"  Loaded: {params['market_name']}")
        except Exception as e:
            print(f"  Error parsing {pf}: {e}")

    df = pd.DataFrame(all_params)
    return df


def compute_statistics(df):
    """
    Compute summary statistics for parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns a, b, c

    Returns
    -------
    stats_df : pd.DataFrame
        DataFrame with statistics
    """
    stats = {}

    for param in ['a', 'b', 'c']:
        if param in df.columns:
            values = df[param].dropna()
            stats[param] = {
                'mean': values.mean(),
                'std': values.std(),
                'variance': values.var(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'n_markets': len(values),
            }

    stats_df = pd.DataFrame(stats).T
    return stats_df


def plot_parameter_distributions(df, output_dir):
    """
    Create histogram plots for each parameter.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a, b, c columns
    output_dir : Path
        Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    params = ['a', 'b', 'c']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for ax, param, color in zip(axes, params, colors):
        if param in df.columns:
            values = df[param].dropna()

            # Histogram
            ax.hist(values, bins=15, color=color, alpha=0.7, edgecolor='black')

            # Add vertical line for mean
            mean_val = values.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2e}')

            # Labels
            ax.set_xlabel(f'Parameter {param}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {param}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'parameter_distributions.png'}")
    plt.close()


def plot_parameter_boxplots(df, output_dir):
    """
    Create box plots for parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a, b, c columns
    output_dir : Path
        Directory to save plots
    """
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    params = ['a', 'b', 'c']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for ax, param, color in zip(axes, params, colors):
        if param in df.columns:
            values = df[param].dropna()

            bp = ax.boxplot([values], vert=True, patch_artist=True,
                           widths=0.6, labels=[param])

            # Color the box
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Add individual points
            y = values.values
            x = np.random.normal(1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.5, s=50, color='black', zorder=3)

            ax.set_ylabel(f'Parameter {param}', fontsize=12, fontweight='bold')
            ax.set_title(f'Box Plot: {param}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'parameter_boxplots.png'}")
    plt.close()


def plot_parameter_scatter(df, output_dir):
    """
    Create scatter plots showing relationships between parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a, b, c columns
    output_dir : Path
        Directory to save plots
    """
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # a vs b
    if 'a' in df.columns and 'b' in df.columns:
        axes[0].scatter(df['a'], df['b'], alpha=0.7, s=100, edgecolors='black', linewidth=1.5)
        axes[0].set_xlabel('Parameter a (trading intensity effect)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Parameter b (depth effect)', fontsize=11, fontweight='bold')
        axes[0].set_title('a vs b', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

    # a vs c
    if 'a' in df.columns and 'c' in df.columns:
        axes[1].scatter(df['a'], df['c'], alpha=0.7, s=100, color='#e74c3c',
                       edgecolors='black', linewidth=1.5)
        axes[1].set_xlabel('Parameter a (trading intensity effect)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Parameter c (autoregressive)', fontsize=11, fontweight='bold')
        axes[1].set_title('a vs c', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # b vs c
    if 'b' in df.columns and 'c' in df.columns:
        axes[2].scatter(df['b'], df['c'], alpha=0.7, s=100, color='#2ecc71',
                       edgecolors='black', linewidth=1.5)
        axes[2].set_xlabel('Parameter b (depth effect)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Parameter c (autoregressive)', fontsize=11, fontweight='bold')
        axes[2].set_title('b vs c', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'parameter_scatter.png'}")
    plt.close()


def plot_parameter_by_market(df, output_dir):
    """
    Create a plot showing parameter values for each market.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with market_name, a, b, c columns
    output_dir : Path
        Directory to save plots
    """
    output_dir = Path(output_dir)

    if 'market_name' not in df.columns:
        return

    # Sort by market name
    df_sorted = df.sort_values('market_name').reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    x = np.arange(len(df_sorted))

    params = ['a', 'b', 'c']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for ax, param, color in zip(axes, params, colors):
        if param in df_sorted.columns:
            values = df_sorted[param].values

            ax.bar(x, values, color=color, alpha=0.7, edgecolor='black')
            ax.axhline(df_sorted[param].mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {df_sorted[param].mean():.2e}')

            ax.set_ylabel(f'{param}', fontsize=12, fontweight='bold')
            ax.set_title(f'Parameter {param} by Market', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            if param == 'c':
                ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    # Set x-axis labels only on bottom plot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(df_sorted['market_name'], rotation=45, ha='right', fontsize=9)
    axes[-1].set_xlabel('Market', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'parameters_by_market.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'parameters_by_market.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze fitted parameters across multiple markets')
    parser.add_argument('--results-dir', default='results/',
                       help='Directory containing market result subdirectories')
    parser.add_argument('--output-dir', default='parameter_analysis/',
                       help='Directory to save analysis outputs')

    args = parser.parse_args()

    print("=" * 80)
    print("PARAMETER STATISTICS AND ANALYSIS")
    print("=" * 80)

    # Collect all parameters
    print("\n[1/6] Collecting parameters from all markets...")
    df = collect_all_parameters(args.results_dir)

    if df.empty:
        print("ERROR: No fitted parameters found!")
        return

    print(f"\nCollected parameters from {len(df)} markets")
    print("\nMarkets:")
    for market in df['market_name']:
        print(f"  - {market}")

    # Compute statistics
    print("\n[2/6] Computing statistics...")
    stats_df = compute_statistics(df)

    print("\nParameter Statistics:")
    print(stats_df.to_string())

    # Save statistics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / 'parameter_statistics.csv'
    stats_df.to_csv(stats_path)
    print(f"\nSaved statistics to: {stats_path}")

    # Save raw data
    raw_path = output_dir / 'all_parameters.csv'
    df.to_csv(raw_path, index=False)
    print(f"Saved raw parameter data to: {raw_path}")

    # Create visualizations
    print("\n[3/6] Creating distribution plots...")
    plot_parameter_distributions(df, output_dir)

    print("\n[4/6] Creating box plots...")
    plot_parameter_boxplots(df, output_dir)

    print("\n[5/6] Creating scatter plots...")
    plot_parameter_scatter(df, output_dir)

    print("\n[6/6] Creating per-market plots...")
    plot_parameter_by_market(df, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nNumber of markets analyzed: {len(df)}")
    print(f"\nParameter means:")
    for param in ['a', 'b', 'c']:
        if param in df.columns:
            mean_val = df[param].mean()
            std_val = df[param].std()
            print(f"  {param}: {mean_val:.6e} ± {std_val:.6e}")

    print(f"\nAll outputs saved to: {output_dir}/")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
