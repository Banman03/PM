#!/usr/bin/env python3
"""
observables.py

Compute observable quantities (R, ℓ, I) from order book snapshots.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def compute_spread(df: pd.DataFrame, normalized: bool = False) -> pd.Series:
    """
    Compute bid-ask spread R(t).

    Parameters
    ----------
    df : DataFrame with columns 'best_bid', 'best_ask', optionally 'mid'
    normalized : if True, compute normalized spread R̃ = (ask-bid)/mid

    Returns
    -------
    spread : Series indexed by df.index
    """
    spread = df['best_ask'] - df['best_bid']

    if normalized:
        mid = df['mid'] if 'mid' in df.columns else 0.5 * (df['best_ask'] + df['best_bid'])
        spread = spread / mid

    return spread


def compute_liquidity_from_book(
    bids: list,
    asks: list,
    trade_size: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Estimate liquidity ℓ via price impact simulation.

    Walk the order book to simulate executing a trade of size `trade_size`,
    compute volume-weighted average execution price, and return price impact per share.

    Parameters
    ----------
    bids : list of (price, size) tuples, sorted descending by price
    asks : list of (price, size) tuples, sorted ascending by price
    trade_size : simulated trade size in shares

    Returns
    -------
    ell_buy : price impact coefficient for a buy ($/share)
    ell_sell : price impact coefficient for a sell ($/share)
    ell_avg : symmetric average (ell_buy + ell_sell)/2

    Notes
    -----
    - If insufficient depth, returns np.nan
    - Interpretation: higher ℓ means less liquid (larger price impact per share)
    """
    if not bids or not asks:
        return np.nan, np.nan, np.nan

    # Midpoint
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    mid = 0.5 * (best_bid + best_ask)

    # --- Buy-side (walk asks) ---
    cumulative = 0.0
    cost = 0.0
    for price, size in sorted(asks, key=lambda x: x[0]):  # ascending price
        if cumulative >= trade_size:
            break
        take = min(size, trade_size - cumulative)
        cost += take * price
        cumulative += take

    if cumulative < trade_size * 0.99:  # insufficient depth
        ell_buy = np.nan
    else:
        vwap_buy = cost / cumulative
        delta_p_buy = vwap_buy - mid
        ell_buy = delta_p_buy / trade_size

    # --- Sell-side (walk bids) ---
    cumulative = 0.0
    revenue = 0.0
    for price, size in sorted(bids, key=lambda x: -x[0]):  # descending price
        if cumulative >= trade_size:
            break
        take = min(size, trade_size - cumulative)
        revenue += take * price
        cumulative += take

    if cumulative < trade_size * 0.99:  # insufficient depth
        ell_sell = np.nan
    else:
        vwap_sell = revenue / cumulative
        delta_p_sell = mid - vwap_sell
        ell_sell = delta_p_sell / trade_size

    # Symmetric average
    if np.isnan(ell_buy) and np.isnan(ell_sell):
        ell_avg = np.nan
    elif np.isnan(ell_buy):
        ell_avg = ell_sell
    elif np.isnan(ell_sell):
        ell_avg = ell_buy
    else:
        ell_avg = 0.5 * (ell_buy + ell_sell)

    return ell_buy, ell_sell, ell_avg


def parse_order_book_from_row(row: pd.Series, jsonl_path: str = None) -> Tuple[list, list]:
    """
    Helper to reconstruct full bid/ask ladders from original JSONL if needed.

    If your CSV only has top-of-book, you need to go back to the original JSONL
    to get full depth. This function is a placeholder.

    For now, we assume you can re-parse or store full depth in a separate structure.

    Returns
    -------
    bids : list of (price, size)
    asks : list of (price, size)
    """
    # Placeholder: in practice you'd load the corresponding JSONL line by timestamp
    # For this template, return empty
    return [], []


def compute_liquidity_series(
    df: pd.DataFrame,
    jsonl_path: str,
    trade_size: float = 1.0,
) -> pd.DataFrame:
    """
    Compute liquidity time series from order book snapshots.

    Parameters
    ----------
    df : DataFrame with parsed snapshots (from process_clob_data.py output CSV)
    jsonl_path : path to original JSONL file to extract full depth
    trade_size : simulated trade size

    Returns
    -------
    df_out : DataFrame with columns ['ell_buy', 'ell_sell', 'ell_avg']
    """
    import json

    # Load JSONL into a dictionary keyed by timestamp for fast lookup
    snapshots = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                data = msg.get('data', msg)

                # Extract timestamp
                ts_ms = None
                if 'timestamp' in msg and isinstance(msg['timestamp'], (int, float)):
                    ts_ms = int(msg['timestamp'])
                elif 'timestamp' in data:
                    try:
                        ts_ms = int(data['timestamp'])
                    except:
                        try:
                            ts_ms = int(float(data['timestamp']))
                        except:
                            pass

                if ts_ms is None:
                    continue

                # Parse bids & asks
                bids_raw = data.get('bids', [])
                asks_raw = data.get('asks', [])

                bids = []
                for e in bids_raw:
                    try:
                        p = float(e.get('price', e[0] if isinstance(e, (list, tuple)) else 0.0))
                        s = float(e.get('size', e[1] if isinstance(e, (list, tuple)) else 0.0))
                        bids.append((p, s))
                    except:
                        continue

                asks = []
                for e in asks_raw:
                    try:
                        p = float(e.get('price', e[0] if isinstance(e, (list, tuple)) else 0.0))
                        s = float(e.get('size', e[1] if isinstance(e, (list, tuple)) else 0.0))
                        asks.append((p, s))
                    except:
                        continue

                snapshots[ts_ms] = {'bids': bids, 'asks': asks}
            except:
                continue

    # Now compute liquidity for each row in df
    results = []
    for idx, row in df.iterrows():
        ts_ms = row['timestamp_ms']
        if ts_ms in snapshots:
            snap = snapshots[ts_ms]
            ell_buy, ell_sell, ell_avg = compute_liquidity_from_book(
                snap['bids'], snap['asks'], trade_size=trade_size
            )
        else:
            ell_buy = ell_sell = ell_avg = np.nan

        results.append({
            'ell_buy': ell_buy,
            'ell_sell': ell_sell,
            'ell_avg': ell_avg,
        })

    df_out = pd.DataFrame(results, index=df.index)
    return df_out


def compute_trading_intensity_depth_change(df: pd.DataFrame, depth_col: str = 'top5_depth_sum') -> pd.Series:
    """
    Compute trading intensity proxy from depth changes.

    I_depth(t) = |ΔD(t)| / Δt

    Parameters
    ----------
    df : DataFrame with 'timestamp_ms' and depth column
    depth_col : name of column containing depth measure

    Returns
    -------
    I_depth : Series of intensity (shares/second)
    """
    depth = df[depth_col].values
    ts_ms = df['timestamp_ms'].values

    # Compute differences
    delta_depth = np.abs(np.diff(depth))
    delta_t_sec = np.diff(ts_ms) / 1000.0  # convert ms to seconds

    # Avoid division by zero
    delta_t_sec = np.maximum(delta_t_sec, 1e-6)

    I_depth = delta_depth / delta_t_sec

    # Prepend NaN for first point (no previous to diff from)
    I_depth = np.concatenate([[np.nan], I_depth])

    return pd.Series(I_depth, index=df.index, name='I_depth')


def compute_trading_intensity_signed(df: pd.DataFrame) -> pd.Series:
    """
    Compute trading intensity proxy from top-level size removals.

    I_signed(t) = [max{0, -Δbest_bid_size} + max{0, -Δbest_ask_size}] / Δt

    Parameters
    ----------
    df : DataFrame with 'timestamp_ms', 'best_bid_size', 'best_ask_size'

    Returns
    -------
    I_signed : Series of intensity (shares/second)
    """
    bid_size = df['best_bid_size'].values
    ask_size = df['best_ask_size'].values
    ts_ms = df['timestamp_ms'].values

    delta_bid = -np.diff(bid_size)  # negative change = removal
    delta_ask = -np.diff(ask_size)

    removals = np.maximum(delta_bid, 0) + np.maximum(delta_ask, 0)

    delta_t_sec = np.diff(ts_ms) / 1000.0
    delta_t_sec = np.maximum(delta_t_sec, 1e-6)

    I_signed = removals / delta_t_sec
    I_signed = np.concatenate([[np.nan], I_signed])

    return pd.Series(I_signed, index=df.index, name='I_signed')


def compute_all_observables(
    csv_path: str,
    jsonl_path: str,
    trade_size: float = 1.0,
    normalized_spread: bool = False,
) -> pd.DataFrame:
    """
    End-to-end: load CSV, compute all observables, return unified DataFrame.

    Parameters
    ----------
    csv_path : path to CSV from process_clob_data.py
    jsonl_path : path to original JSONL for full depth
    trade_size : simulated trade size for liquidity estimation
    normalized_spread : if True, compute R̃ instead of R

    Returns
    -------
    df : DataFrame with columns:
        - timestamp_ms, t_iso, best_bid, best_ask, mid, spread, ...
        - R (bid-ask spread)
        - ell_buy, ell_sell, ell_avg (liquidity)
        - I_depth, I_signed (trading intensity proxies)
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Compute spread
    df['R'] = compute_spread(df, normalized=normalized_spread)

    # Compute liquidity
    ell_df = compute_liquidity_series(df, jsonl_path, trade_size=trade_size)
    df = pd.concat([df, ell_df], axis=1)

    # Compute intensity proxies
    df['I_depth'] = compute_trading_intensity_depth_change(df)
    df['I_signed'] = compute_trading_intensity_signed(df)

    # Default combined intensity (just use I_depth for now)
    df['I'] = df['I_depth']

    return df


def convert_to_tau(df: pd.DataFrame, resolution_time_ms: int) -> pd.DataFrame:
    """
    Convert calendar time t to time-to-resolution τ = T - t.

    Parameters
    ----------
    df : DataFrame with 'timestamp_ms'
    resolution_time_ms : market resolution timestamp in milliseconds

    Returns
    -------
    df : DataFrame with new column 'tau_sec' (time to resolution in seconds)
    """
    df = df.copy()
    df['tau_sec'] = (resolution_time_ms - df['timestamp_ms']) / 1000.0
    return df


def resample_regular_grid(
    df: pd.DataFrame,
    tau_col: str = 'tau_sec',
    observables: list = ['R', 'ell_avg', 'I'],
    n_points: int = 500,
) -> pd.DataFrame:
    """
    Resample observables onto a regular grid in τ.

    Parameters
    ----------
    df : DataFrame with tau_col and observable columns
    tau_col : name of time-to-resolution column
    observables : list of column names to resample
    n_points : number of points in regular grid

    Returns
    -------
    df_grid : DataFrame with regular τ grid and interpolated observables
    """
    df = df.sort_values(tau_col).reset_index(drop=True)

    tau_min = df[tau_col].min()
    tau_max = df[tau_col].max()
    tau_grid = np.linspace(tau_min, tau_max, n_points)

    # Interpolate each observable
    data = {tau_col: tau_grid}
    for obs in observables:
        if obs in df.columns:
            # Linear interpolation
            data[obs] = np.interp(tau_grid, df[tau_col].values, df[obs].values, left=np.nan, right=np.nan)
        else:
            data[obs] = np.full(n_points, np.nan)

    df_grid = pd.DataFrame(data)
    return df_grid


def smooth_series(series: pd.Series, window: int = 3, method: str = 'moving_average') -> pd.Series:
    """
    Apply light smoothing to reduce noise.

    Parameters
    ----------
    series : input time series
    window : smoothing window size
    method : 'moving_average' or 'savgol'

    Returns
    -------
    smoothed : smoothed series
    """
    if method == 'moving_average':
        return series.rolling(window=window, center=True, min_periods=1).mean()
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        # Need odd window
        if window % 2 == 0:
            window += 1
        return pd.Series(
            savgol_filter(series.fillna(method='ffill').fillna(method='bfill'), window_length=window, polyorder=2),
            index=series.index
        )
    else:
        return series


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python observables.py <csv_path> <jsonl_path> [trade_size]")
        sys.exit(1)

    csv_path = sys.argv[1]
    jsonl_path = sys.argv[2]
    trade_size = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    print(f"Computing observables from {csv_path} and {jsonl_path} with trade_size={trade_size}...")

    df_obs = compute_all_observables(csv_path, jsonl_path, trade_size=trade_size)

    # Save to CSV
    out_path = csv_path.replace('.csv', '_observables.csv')
    df_obs.to_csv(out_path, index=False)
    print(f"Saved observables to {out_path}")

    # Print summary stats
    print("\nSummary statistics:")
    print(df_obs[['R', 'ell_avg', 'I_depth', 'I_signed']].describe())
