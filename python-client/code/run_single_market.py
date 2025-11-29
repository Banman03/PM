#!/usr/bin/env python3
"""
run_single_market.py

Simplified script to run the analysis on a single market.
Automatically looks up the resolution time from market-data.json.

Usage:
    python run_single_market.py Georgia_vs._Georgia_Tech_20812556181257502299452162938829645659671675918267704479092636289804545785305.jsonl
"""

import argparse
import os
import json
from datetime import datetime
import subprocess
import sys

def parse_iso_to_ms(iso_string):
    """Convert ISO timestamp string to milliseconds since epoch."""
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    return int(dt.timestamp() * 1000)

def extract_token_id_from_filename(filename):
    """Extract token ID from filename."""
    base = filename.replace('.jsonl', '')
    parts = base.split('_')
    for i in range(len(parts)-1, -1, -1):
        if parts[i].isdigit():
            return parts[i]
    return None

def find_resolution_time(token_id, market_data_path='market-data.json'):
    """Find resolution time for a given token ID."""
    with open(market_data_path, 'r') as f:
        markets = json.load(f)

    for market in markets:
        clob_token_ids = market.get('clobTokenIds')
        if not clob_token_ids:
            continue

        try:
            token_ids = json.loads(clob_token_ids)
        except:
            continue

        if token_id in token_ids:
            end_date = market.get('endDate')
            if end_date:
                return parse_iso_to_ms(end_date), market.get('question', 'Unknown')

    return None, None

def main():
    parser = argparse.ArgumentParser(description='Run liquidity analysis on a single market')
    parser.add_argument('jsonl_file', help='Path to JSONL file')
    parser.add_argument('--market-data', default='market-data.json', help='Path to market-data.json')
    parser.add_argument('--output-dir', help='Output directory (auto-generated if not specified)')
    parser.add_argument('--trade-size', type=float, default=1.0)
    parser.add_argument('--model-type', choices=['linear', 'bounded'], default='linear')
    parser.add_argument('--skip-bootstrap', action='store_true')
    parser.add_argument('--skip-sensitivity', action='store_true')
    parser.add_argument('--force-csv-regen', action='store_true', help='Force regeneration of CSV file')

    args = parser.parse_args()

    # Extract token ID
    jsonl_filename = os.path.basename(args.jsonl_file)
    token_id = extract_token_id_from_filename(jsonl_filename)

    if not token_id:
        print(f"ERROR: Could not extract token ID from {jsonl_filename}")
        sys.exit(1)

    print(f"Token ID: {token_id}")

    # Find resolution time
    resolution_ms, market_name = find_resolution_time(token_id, args.market_data)

    if resolution_ms is None:
        print(f"ERROR: Could not find resolution time for token ID {token_id}")
        sys.exit(1)

    print(f"Market: {market_name}")
    print(f"Resolution time: {resolution_ms} ms")

    # Generate or find CSV
    jsonl_dir = os.path.dirname(args.jsonl_file) or '.'
    csv_dir = os.path.join(os.path.dirname(jsonl_dir), 'csvs')
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = 'PROCESSED' + jsonl_filename.replace('.jsonl', '.csv')
    csv_path = os.path.join(csv_dir, csv_filename)

    if not os.path.exists(csv_path) or args.force_csv_regen:
        print(f"\nGenerating CSV file...")
        result = subprocess.run(
            ['python3', 'process_clob_data.py', '--in', jsonl_dir, '--out', csv_dir],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"ERROR: Failed to generate CSV:")
            print(result.stderr)
            sys.exit(1)
        print(f"Created: {csv_path}")
    else:
        print(f"Using existing CSV: {csv_path}")

    # Set output directory
    if not args.output_dir:
        safe_name = market_name.replace(' ', '_').replace('/', '_').replace('.', '')
        args.output_dir = f"results/{safe_name}_{token_id}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Build command
    cmd = [
        'python3', 'pipeline.py',
        '--csv', csv_path,
        '--jsonl', args.jsonl_file,
        '--resolution-time', str(resolution_ms),
        '--output-dir', args.output_dir,
        '--trade-size', str(args.trade_size),
        '--model-type', args.model_type,
    ]

    if args.skip_bootstrap:
        cmd.append('--skip-bootstrap')
    if args.skip_sensitivity:
        cmd.append('--skip-sensitivity')

    print(f"\nRunning pipeline...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run pipeline
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\nSUCCESS! Results saved to {args.output_dir}")
    else:
        print(f"\nERROR: Pipeline failed with return code {result.returncode}")
        sys.exit(1)

if __name__ == '__main__':
    main()
