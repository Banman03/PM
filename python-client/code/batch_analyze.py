#!/usr/bin/env python3
"""
batch_analyze.py

Batch process all JSONL files using the liquidity dynamics pipeline.
Automatically extracts market resolution times from market-data.json and
processes each market individually.

Usage:
    python batch_analyze.py --jsonl-dir data/order-books/jsons --market-data market-data.json --output-base results/
"""

import argparse
import os
import json
from datetime import datetime, timezone
import subprocess
import sys

def parse_iso_to_ms(iso_string):
    """Convert ISO timestamp string to milliseconds since epoch."""
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    return int(dt.timestamp() * 1000)

def load_market_data(market_data_path):
    """
    Load market data JSON and create a mapping from token ID to resolution time.

    Returns:
        dict: {token_id: resolution_time_ms}
    """
    with open(market_data_path, 'r') as f:
        markets = json.load(f)

    token_to_resolution = {}

    for market in markets:
        # Extract endDate (resolution time)
        end_date = market.get('endDate')
        if not end_date:
            print(f"  Warning: No endDate for market {market.get('question', 'Unknown')}")
            continue

        resolution_ms = parse_iso_to_ms(end_date)

        # Extract token IDs
        clob_token_ids = market.get('clobTokenIds')
        if not clob_token_ids:
            continue

        # Parse token IDs (it's a JSON string)
        try:
            token_ids = json.loads(clob_token_ids)
        except:
            print(f"  Warning: Could not parse clobTokenIds for {market.get('question', 'Unknown')}")
            continue

        # Map each token ID to this resolution time
        for token_id in token_ids:
            token_to_resolution[token_id] = {
                'resolution_ms': resolution_ms,
                'market_name': market.get('question', 'Unknown'),
                'end_date': end_date,
            }

    return token_to_resolution

def extract_token_id_from_filename(filename):
    """
    Extract token ID from filename.

    Filename format: MarketName_TOKEN_ID.jsonl
    Example: Georgia_vs._Georgia_Tech_20812556181257502299452162938829645659671675918267704479092636289804545785305.jsonl
    """
    # Remove .jsonl extension
    base = filename.replace('.jsonl', '')

    # The token ID is the last part after the last underscore,
    # but we need to handle market names with underscores
    # Strategy: find the longest numeric suffix
    parts = base.split('_')
    for i in range(len(parts)-1, -1, -1):
        if parts[i].isdigit():
            return parts[i]

    return None

def main():
    parser = argparse.ArgumentParser(description='Batch process liquidity dynamics for all markets')
    parser.add_argument('--jsonl-dir', required=True, help='Directory containing JSONL files')
    parser.add_argument('--market-data', required=True, help='Path to market-data.json')
    parser.add_argument('--output-base', default='results/', help='Base output directory')
    parser.add_argument('--trade-size', type=float, default=1.0, help='Trade size for liquidity estimation')
    parser.add_argument('--model-type', choices=['linear', 'bounded'], default='linear', help='Model type')
    parser.add_argument('--skip-bootstrap', action='store_true', help='Skip bootstrap for faster processing')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Skip sensitivity analysis')
    parser.add_argument('--n-bootstrap', type=int, default=100, help='Number of bootstrap samples')

    args = parser.parse_args()

    # Load market data
    print("=" * 80)
    print("BATCH LIQUIDITY DYNAMICS ANALYSIS")
    print("=" * 80)
    print(f"\nLoading market data from {args.market_data}...")

    token_to_resolution = load_market_data(args.market_data)
    print(f"  Loaded {len(token_to_resolution)} token-to-resolution mappings")

    # Find all JSONL files
    print(f"\nScanning {args.jsonl_dir} for JSONL files...")
    jsonl_files = [f for f in os.listdir(args.jsonl_dir) if f.endswith('.jsonl')]
    print(f"  Found {len(jsonl_files)} JSONL files")

    # Process each file
    processed = 0
    skipped = 0
    failed = 0

    for jsonl_file in sorted(jsonl_files):
        jsonl_path = os.path.join(args.jsonl_dir, jsonl_file)

        # Extract token ID from filename
        token_id = extract_token_id_from_filename(jsonl_file)

        if token_id is None:
            print(f"\n[SKIP] {jsonl_file}")
            print(f"  Reason: Could not extract token ID from filename")
            skipped += 1
            continue

        if token_id not in token_to_resolution:
            print(f"\n[SKIP] {jsonl_file}")
            print(f"  Reason: Token ID {token_id} not found in market data")
            skipped += 1
            continue

        market_info = token_to_resolution[token_id]
        market_name = market_info['market_name']
        resolution_ms = market_info['resolution_ms']

        print(f"\n[PROCESS] {jsonl_file}")
        print(f"  Market: {market_name}")
        print(f"  Resolution time: {market_info['end_date']} ({resolution_ms} ms)")

        # First, check if CSV needs to be generated
        csv_dir = os.path.join(os.path.dirname(args.jsonl_dir), 'csvs')
        os.makedirs(csv_dir, exist_ok=True)

        csv_filename = jsonl_file.replace('.jsonl', '.csv')
        csv_path = os.path.join(csv_dir, 'PROCESSED' + csv_filename)

        if not os.path.exists(csv_path):
            print(f"  Generating CSV with process_clob_data.py...")
            try:
                # Create temporary directories for process_clob_data.py
                temp_jsonl_dir = os.path.join('/tmp', f'jsonl_{token_id}')
                os.makedirs(temp_jsonl_dir, exist_ok=True)

                # Copy JSONL to temp dir
                import shutil
                shutil.copy(jsonl_path, os.path.join(temp_jsonl_dir, jsonl_file))

                # Run process_clob_data.py
                result = subprocess.run(
                    ['python3', 'process_clob_data.py', '--in', temp_jsonl_dir, '--out', csv_dir],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    print(f"  ERROR: process_clob_data.py failed:")
                    print(result.stderr)
                    failed += 1
                    continue

                # Clean up temp dir
                shutil.rmtree(temp_jsonl_dir)

                print(f"  Created {csv_path}")
            except Exception as e:
                print(f"  ERROR: Failed to create CSV: {e}")
                failed += 1
                continue
        else:
            print(f"  CSV already exists: {csv_path}")

        # Create output directory for this market
        safe_market_name = market_name.replace(' ', '_').replace('/', '_').replace('.', '')
        output_dir = os.path.join(args.output_base, f"{safe_market_name}_{token_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Build pipeline command
        cmd = [
            'python3', 'pipeline.py',
            '--csv', csv_path,
            '--jsonl', jsonl_path,
            '--resolution-time', str(resolution_ms),
            '--output-dir', output_dir,
            '--trade-size', str(args.trade_size),
            '--model-type', args.model_type,
            '--n-bootstrap', str(args.n_bootstrap),
        ]

        if args.skip_bootstrap:
            cmd.append('--skip-bootstrap')
        if args.skip_sensitivity:
            cmd.append('--skip-sensitivity')

        # Run pipeline
        print(f"  Running pipeline...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ERROR: Pipeline failed:")
                print(result.stderr)
                failed += 1
            else:
                print(f"  SUCCESS! Results saved to {output_dir}")
                processed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total files: {len(jsonl_files)}")
    print("=" * 80)

if __name__ == '__main__':
    main()
