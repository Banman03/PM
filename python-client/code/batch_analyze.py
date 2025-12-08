#!/usr/bin/env python3
"""
batch_analyze.py

Batch process all JSONL files using the liquidity dynamics pipeline.
Automatically extracts market resolution times from the last timestamp in each JSONL file.

Usage:
    python batch_analyze.py --jsonl-dir data/order-books/jsons --output-base results/
"""

import argparse
import os
import json
from datetime import datetime, timezone
import subprocess
import sys

def get_last_timestamp(jsonl_path):
    """Get the timestamp from the last line of a JSONL file (resolution time)."""
    last_line = None

    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        return None

    data = json.loads(last_line)

    # Try top-level timestamp
    ts_ms = data.get('timestamp')
    if ts_ms and isinstance(ts_ms, (int, float)):
        return int(ts_ms)

    # Try nested data.timestamp
    if 'data' in data:
        ts_ms = data['data'].get('timestamp')
        if ts_ms:
            try:
                return int(ts_ms)
            except:
                try:
                    return int(float(ts_ms))
                except:
                    pass

    return None

def extract_token_id_from_filename(filename):
    """
    Extract token ID from filename.

    Filename format: MarketName_TOKEN_ID.jsonl
    """
    base = filename.replace('.jsonl', '')
    parts = base.split('_')
    for i in range(len(parts)-1, -1, -1):
        if parts[i].isdigit():
            return parts[i]
    return None

def extract_market_name_from_filename(filename):
    """Extract market name from filename (before the token ID)."""
    base = filename.replace('.jsonl', '')
    parts = base.split('_')
    for i in range(len(parts)-1, -1, -1):
        if parts[i].isdigit():
            return '_'.join(parts[:i])
    return base

def main():
    parser = argparse.ArgumentParser(description='Batch process liquidity dynamics for all markets')
    parser.add_argument('--jsonl-dir', required=True, help='Directory containing JSONL files')
    parser.add_argument('--output-base', default='results/', help='Base output directory')
    parser.add_argument('--trade-size', type=float, default=1.0, help='Trade size for liquidity estimation')
    parser.add_argument('--model-type', choices=['linear', 'bounded'], default='linear', help='Model type')
    parser.add_argument('--skip-bootstrap', action='store_true', help='Skip bootstrap for faster processing')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Skip sensitivity analysis')
    parser.add_argument('--n-bootstrap', type=int, default=100, help='Number of bootstrap samples')

    args = parser.parse_args()

    print("=" * 80)
    print("BATCH LIQUIDITY DYNAMICS ANALYSIS")
    print("=" * 80)

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

        # Extract info from filename
        token_id = extract_token_id_from_filename(jsonl_file)
        market_name = extract_market_name_from_filename(jsonl_file)

        if token_id is None:
            print(f"\n[SKIP] {jsonl_file}")
            print(f"  Reason: Could not extract token ID from filename")
            skipped += 1
            continue

        resolution_ms = get_last_timestamp(jsonl_path)

        if resolution_ms is None:
            print(f"\n[SKIP] {jsonl_file}")
            print(f"  Reason: Could not extract resolution time from JSONL")
            skipped += 1
            continue

        print(f"\n[PROCESS] {jsonl_file}")
        print(f"  Market: {market_name}")
        print(f"  Resolution time: {resolution_ms} ms")

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
