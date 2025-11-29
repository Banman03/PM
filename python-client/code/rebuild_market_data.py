#!/usr/bin/env python3
"""
rebuild_market_data.py

Rebuild market-data.json by fetching market information from Polymarket API
for each JSONL file in a directory.

For each JSONL file:
1. Read the first line to extract the asset_id
2. Fetch market data from Polymarket API using the asset_id
3. Compile all market data into market-data.json

Usage:
    python rebuild_market_data.py --jsonl-dir data/order-books/jsons --output market-data.json
"""

import argparse
import os
import json
import requests
from typing import Optional, Dict, Any
import time

def extract_asset_id_from_jsonl(jsonl_path: str) -> Optional[str]:
    """
    Extract asset_id from the first line of a JSONL file.

    Parameters
    ----------
    jsonl_path : str
        Path to JSONL file

    Returns
    -------
    asset_id : str or None
        The asset_id from the first entry, or None if not found
    """
    try:
        with open(jsonl_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return None

            data = json.loads(first_line)

            # Try to get asset_id from top level
            market_id = data['data']["market"]
            if market_id:
                return market_id

            return None

    except Exception as e:
        print(f"  ERROR reading {jsonl_path}: {e}")
        return None


def fetch_market_by_token_id(token_id: str) -> Optional[Dict[Any, Any]]:
    """
    Fetch market data from Polymarket API using a token/asset ID.

    The Polymarket API has several endpoints we can try:
    1. /markets endpoint with filtering
    2. Direct market lookup by condition ID (if we can derive it)

    Parameters
    ----------
    token_id : str
        The CLOB token ID / asset ID

    Returns
    -------
    market_data : dict or None
        Market information from API, or None if not found
    """

    # Try the markets endpoint with token ID search
    # This searches through markets that contain this token ID
    base_url = "https://gamma-api.polymarket.com"

    try:
        # Method 1: Search markets endpoint
        # The gamma API allows searching by various fields
        search_url = f"{base_url}/markets"

        print(f"    Fetching from {search_url}...")
        params = {
            "condition_ids": token_id,
            "closed": True
        }
        
        response = requests.get(search_url, params=params, timeout=10)

        if response.status_code != 200:
            print(f"    API returned status {response.status_code}")
            return None

        market = response.json()
        # print("markets: ", json.dumps(market, indent=2))
        return market[0]

    except requests.exceptions.RequestException as e:
        print(f"    ERROR fetching from API: {e}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Rebuild market-data.json from JSONL files')
    parser.add_argument('--jsonl-dir', required=True, help='Directory containing JSONL files')
    parser.add_argument('--output', default='market-data.json', help='Output path for market-data.json')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API requests in seconds (default: 0.5)')

    args = parser.parse_args()

    print("=" * 80)
    print("REBUILDING MARKET-DATA.JSON")
    print("=" * 80)

    # Find all JSONL files
    print(f"\nScanning {args.jsonl_dir} for JSONL files...")

    if not os.path.exists(args.jsonl_dir):
        print(f"ERROR: Directory {args.jsonl_dir} does not exist")
        return

    jsonl_files = sorted([f for f in os.listdir(args.jsonl_dir) if f.endswith('.jsonl')])

    print(f"Found {len(jsonl_files)} JSONL files")

    # Process each file
    all_markets = []
    seen_token_ids = set()
    fetched = 0
    skipped = 0
    failed = 0

    for i, jsonl_file in enumerate(jsonl_files, 1):
        jsonl_path = os.path.join(args.jsonl_dir, jsonl_file)

        print(f"\n[{i}/{len(jsonl_files)}] Processing {jsonl_file}")

        # Extract asset ID
        asset_id = extract_asset_id_from_jsonl(jsonl_path)

        if not asset_id:
            print(f"  SKIP: Could not extract asset_id")
            skipped += 1
            continue

        print(f"  Asset ID: {asset_id}")

        # Check if we've already seen this token
        if asset_id in seen_token_ids:
            print(f"  SKIP: Already fetched market for this token ID")
            skipped += 1
            continue

        seen_token_ids.add(asset_id)

        # Fetch market data
        market_data = fetch_market_by_token_id(asset_id)

        if market_data:
            all_markets.append(market_data)
            fetched += 1

            # Show key info
            print(f"  ✓ Market: {market_data.get('question', 'Unknown')}")
            print(f"  ✓ End date: {market_data.get('endDate', 'Unknown')}")
        else:
            print(f"  ✗ Failed to fetch market data")
            failed += 1

        # Rate limiting: sleep between requests
        if i < len(jsonl_files):
            time.sleep(args.delay)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    if all_markets:
        with open(args.output, 'w') as f:
            json.dump(all_markets, f, indent=2)

        print(f"Saved {len(all_markets)} markets to {args.output}")
    else:
        print("WARNING: No markets were fetched. Not writing output file.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total JSONL files: {len(jsonl_files)}")
    print(f"  Successfully fetched: {fetched}")
    print(f"  Skipped (duplicates/errors): {skipped}")
    print(f"  Failed to fetch: {failed}")
    print("=" * 80)

    if failed > 0:
        print("\nNOTE: Some markets could not be fetched from the API.")
        print("This might be because:")
        print("  - The market is no longer active/archived")
        print("  - The API endpoint changed")
        print("  - Rate limiting or network issues")
        print("\nYou can manually add missing markets to market-data.json or")
        print("manually specify resolution times when running the pipeline.")


if __name__ == '__main__':
    main()
