#!/usr/bin/env python3
"""
process_clob.py

Given newline-delimited JSON CLOB snapshots (one JSON object per line),
extract top-of-book, depth statistics, and (if present) trade sizes.

Usage:
    # from a file of snapshots:
    python process_clob.py --in snapshots.ndjson --out clob_timeseries.csv

Assumes timestamps in the JSON are milliseconds since epoch (integers or strings).
"""

import json
import argparse
import csv
from datetime import datetime, timezone

def parse_snapshot(msg):
    """
    Parse one JSON snapshot (dict) and return a row dict.
    msg: parsed JSON as python dict (structure like your sample).
    """
    data = msg.get("data", msg)
    
    ts_ms = None
    if "timestamp" in msg and isinstance(msg["timestamp"], (int, float)):
        ts_ms = int(msg["timestamp"])
    elif "timestamp" in data:
        try:
            ts_ms = int(data["timestamp"])
        except Exception:
            # maybe it's a string number
            try:
                ts_ms = int(float(data["timestamp"]))
            except Exception:
                ts_ms = None

    # Parse bids & asks lists
    bids = data.get("bids", [])
    asks = data.get("asks", [])

    # Convert to numeric lists of (price, size)
    def to_pairs(side):
        pairs = []
        for e in side:
            try:
                p = float(e.get("price", e[0] if isinstance(e, (list,tuple)) else 0.0))
                s = float(e.get("size", e[1] if isinstance(e, (list,tuple)) else 0.0))
                pairs.append((p, s))
            except Exception:
                continue
        return pairs

    bids_pairs = to_pairs(bids)
    asks_pairs = to_pairs(asks)

    best_bid = None
    best_bid_size = 0.0
    if bids_pairs:
        # best bid is max price
        best_bid, best_bid_size = max(bids_pairs, key=lambda x: x[0])

    best_ask = None
    best_ask_size = 0.0
    if asks_pairs:
        # best ask is min price
        best_ask, best_ask_size = min(asks_pairs, key=lambda x: x[0])

    mid = None
    spread = None
    if best_bid is not None and best_ask is not None:
        mid = 0.5*(best_bid + best_ask)
        spread = best_ask - best_bid

    # total depth = sum of sizes across entire book (careful: can be huge)
    total_depth = sum(s for _, s in bids_pairs + asks_pairs)

    # top-N depth (e.g., top 5 on each side) as a more stable measure
    bids_sorted = sorted(bids_pairs, key=lambda x:-x[0])[:5]
    asks_sorted = sorted(asks_pairs, key=lambda x:x[0])[:5]
    top5_depth_sum = sum(s for _, s in bids_sorted + asks_sorted)

    # trades: some CLOB feeds include explicit trade messages. If present,
    # extract executed trade volume from a field like data['trades'] or msg['trade'].
    # For this snapshot, there wasn't a 'trades' array. We'll default to 0.
    trade_vol = 0.0
    # Example: if data contains trades list with 'size' keys:
    if "trades" in data and isinstance(data["trades"], list):
        for t in data["trades"]:
            try:
                trade_vol += float(t.get("size", 0.0))
            except Exception:
                pass
    # if top-level has 'trade' entry that is a single dict:
    if "trade" in data and isinstance(data["trade"], dict):
        try:
            trade_vol += float(data["trade"].get("size", 0.0))
        except Exception:
            pass

    # Compose row
    if ts_ms is None:
        # fallback: use current time
        ts_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    t_iso = datetime.fromtimestamp(ts_ms/1000.0, tz=timezone.utc).isoformat()

    row = {
        "timestamp_ms": int(ts_ms),
        "t_iso": t_iso,
        "best_bid": best_bid if best_bid is not None else "",
        "best_bid_size": best_bid_size,
        "best_ask": best_ask if best_ask is not None else "",
        "best_ask_size": best_ask_size,
        "mid": mid if mid is not None else "",
        "spread": spread if spread is not None else "",
        "total_depth": total_depth,
        "top5_depth_sum": top5_depth_sum,
        "trade_volume": trade_vol
    }
    return row

def process_ndjson(infile, outfile):
    with open(infile, "r") as fin, open(outfile, "w", newline="") as fout:
        fieldnames = ["timestamp_ms","t_iso","best_bid","best_bid_size","best_ask","best_ask_size",
                      "mid","spread","total_depth","top5_depth_sum","trade_volume"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                # try to ignore non-json lines
                continue
            row = parse_snapshot(msg)
            writer.writerow(row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="Input ndjson of snapshots")
    ap.add_argument("--out", dest="outfile", required=True, help="CSV output path")
    args = ap.parse_args()
    process_ndjson(args.infile, args.outfile)
