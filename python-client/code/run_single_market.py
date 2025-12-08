
"""
run_single_market.py

Simplified script to run the analysis on a single market.
Automatically extracts the resolution time from the last timestamp in the JSONL file.

Usage:
    python run_single_market.py Georgia_vs._Georgia_Tech_20812556181257502299452162938829645659671675918267704479092636289804545785305.jsonl
"""

import argparse
import os
import json
from datetime import datetime
import subprocess
import sys

def extract_token_id_from_filename(filename):
    """Extract token ID from filename."""
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

    
    ts_ms = data.get('timestamp')
    if ts_ms and isinstance(ts_ms, (int, float)):
        return int(ts_ms)

    
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

def main():
    parser = argparse.ArgumentParser(description='Run liquidity analysis on a single market')
    parser.add_argument('jsonl_file', help='Path to JSONL file')
    parser.add_argument('--output-dir', help='Output directory (auto-generated if not specified)')
    parser.add_argument('--trade-size', type=float, default=1.0)
    parser.add_argument('--model-type', choices=['linear', 'bounded'], default='linear')
    parser.add_argument('--force-csv-regen', action='store_true', help='Force regeneration of CSV file')

    args = parser.parse_args()

    
    jsonl_filename = os.path.basename(args.jsonl_file)
    token_id = extract_token_id_from_filename(jsonl_filename)
    market_name = extract_market_name_from_filename(jsonl_filename)

    if not token_id:
        print(f"ERROR: Could not extract token ID from {jsonl_filename}")
        sys.exit(1)

    print(f"Market: {market_name}")
    print(f"Token ID: {token_id}")

    
    print("Extracting resolution time from last JSONL timestamp...")
    resolution_ms = get_last_timestamp(args.jsonl_file)

    if resolution_ms is None:
        print(f"ERROR: Could not extract resolution time from {args.jsonl_file}")
        sys.exit(1)

    print(f"Resolution time: {resolution_ms} ms")

    
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(resolution_ms / 1000.0, tz=timezone.utc)
    print(f"Resolution time (readable): {dt.isoformat()}")

    
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

    
    if not args.output_dir:
        safe_name = market_name.replace(' ', '_').replace('/', '_').replace('.', '')
        args.output_dir = f"results/{safe_name}_{token_id}"

    os.makedirs(args.output_dir, exist_ok=True)

    
    cmd = [
        'python3', 'pipeline.py',
        '--csv', csv_path,
        '--jsonl', args.jsonl_file,
        '--resolution-time', str(resolution_ms),
        '--output-dir', args.output_dir,
        '--trade-size', str(args.trade_size),
        '--model-type', args.model_type,
    ]

    print(f"\nRunning pipeline...")
    print(f"Command: {' '.join(cmd)}\n")

    
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\nSUCCESS! Results saved to {args.output_dir}")
    else:
        print(f"\nERROR: Pipeline failed with return code {result.returncode}")
        sys.exit(1)

if __name__ == '__main__':
    main()
