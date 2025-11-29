#!/usr/bin/env python3
"""
extract_resolution_time.py

Extract resolution time from a JSONL file by reading the last timestamp.
The assumption is that the last snapshot is closest to market resolution.

Usage:
    python extract_resolution_time.py <jsonl_file>
"""

import sys
import json

def get_last_timestamp(jsonl_path):
    """
    Get the timestamp from the last line of a JSONL file.

    Returns:
        int: Timestamp in milliseconds, or None if not found
    """
    last_line = None

    try:
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

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_resolution_time.py <jsonl_file>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    resolution_ms = get_last_timestamp(jsonl_path)

    if resolution_ms:
        print(resolution_ms)
        sys.exit(0)
    else:
        print("ERROR: Could not extract resolution time", file=sys.stderr)
        sys.exit(1)
