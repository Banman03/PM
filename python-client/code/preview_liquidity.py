#!/usr/bin/env python3
"""
preview_liquidity.py

Quick preview of computed liquidity vs time-to-resolution for all markets.
No model fitting - just compute and plot observables.

Usage:
    python preview_liquidity.py --jsonl-dir data/order-books/jsons --output preview_liquidity.png
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from observables import compute_all_observables, convert_to_tau
from extract_resolution_time import get_last_timestamp

def extract_market_name_from_filename(filename):
    """Extract market name from filename (before the token ID)."""
    base = filename.replace('.jsonl', '')
    parts = base.split('_')
    for i in range(len(parts)-1, -1, -1):
        if parts[i].isdigit():
            return '_'.join(parts[:i])
    return base

def main():
    parser = argparse.ArgumentParser(description='Preview liquidity for all markets')
    parser.add_argument('--jsonl-dir', required=True, help='Directory with JSONL files')
    parser.add_argument('--csv-dir', required=True, help="Directory with CSV files")
    parser.add_argument('--output', default='preview_liquidity.png', help='Output plot path')
    parser.add_argument('--max-markets', type=int, default=None, help='Max markets to process')

    args = parser.parse_args()

    print("=" * 80)
    print("LIQUIDITY PREVIEW")
    print("=" * 80)

    # Find all JSONL files
    jsonl_files = sorted([f for f in os.listdir(args.jsonl_dir) if f.endswith('.jsonl')])
    print(f"\nFound {len(jsonl_files)} JSONL files")

    if args.max_markets is not None and len(jsonl_files) > args.max_markets:
        print(f"Processing first {args.max_markets} markets (use --max-markets to change)")
        jsonl_files = jsonl_files[:args.max_markets]

    # Process each market
    market_data = []

    for i, jsonl_file in enumerate(jsonl_files, 1):
        jsonl_path = os.path.join(args.jsonl_dir, jsonl_file)
        market_name = extract_market_name_from_filename(jsonl_file)

        print(f"\n[{i}/{len(jsonl_files)}] {market_name}")

        # Get resolution time
        resolution_ms = get_last_timestamp(jsonl_path)
        if not resolution_ms:
            print("  SKIP: No resolution time")
            continue

        # Check for CSV
        csv_dir = os.path.dirname(args.csv_dir)
        csv_filename = 'PROCESSED' + jsonl_file.replace('.jsonl', '.csv')
        csv_path = os.path.join(csv_dir, csv_filename)

        if not os.path.exists(csv_path):
            print(f"  SKIP: No CSV found at {csv_path}")
            print(f"  Run: python process_clob_data.py --in {args.jsonl_dir} --out {csv_dir}")
            continue

        # Compute observables
        try:
            df = compute_all_observables(csv_path, jsonl_path, use_depth_liquidity=True)
            df = convert_to_tau(df, resolution_ms)

            # Filter valid data
            valid = df['ell_avg'].notna() & np.isfinite(df['ell_avg']) & (df['ell_avg'] > 0)
            df_valid = df[valid]

            if len(df_valid) < 10:
                print(f"  SKIP: Only {len(df_valid)} valid liquidity points")
                continue

            # Store
            market_data.append({
                'name': market_name,
                'tau': df_valid['tau_sec'].values,
                'ell': df_valid['ell_avg'].values,
                'R': df_valid['R'].values,
                'I': df_valid['I'].values,
            })

            # Print stats
            print(f"  Snapshots: {len(df_valid)}")
            print(f"  Liquidity range: [{df_valid['ell_avg'].min():.2e}, {df_valid['ell_avg'].max():.2e}]")
            print(f"  Tau range: [{df_valid['tau_sec'].min():.1f}, {df_valid['tau_sec'].max():.1f}] sec")
            print(f"  Mean liquidity: {df_valid['ell_avg'].mean():.2e}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not market_data:
        print("\nNo markets with valid data!")
        return

    # Create plot
    print(f"\nGenerating plot with {len(market_data)} markets...")

    n_markets = len(market_data)
    ncols = 2
    nrows = (n_markets + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    if n_markets == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, market in enumerate(market_data):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Plot liquidity vs tau
        ax.scatter(market['tau'], market['ell'], alpha=0.5, s=10)
        ax.set_xlabel('Time to Resolution (seconds)')
        ax.set_ylabel('Liquidity (spread/volume)')
        ax.set_title(f"{market['name']}\n(n={len(market['tau'])})", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Log scale if wide range
        ell_range = market['ell'].max() / (market['ell'].min() + 1e-10)
        if ell_range > 100:
            ax.set_yscale('log')

    # Remove empty subplots
    for idx in range(len(market_data), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {args.output}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for market in market_data:
        mean_ell = np.mean(market['ell'])
        std_ell = np.std(market['ell'])
        print(f"{market['name']:50s} mean={mean_ell:.2e}, std={std_ell:.2e}")

if __name__ == '__main__':
    main()
