#!/usr/bin/env python3
"""
model_from_csv.py

Reads CSV produced by process_clob.py and builds I(t) and R(t) time series,
then computes analytic and numerically-solved liquidity trajectories.

Usage:
    python model_from_csv.py --in clob_timeseries.csv --resolution_ts 1764344000000
"""

import numpy as np
import pandas as pd
import argparse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ---------- analytic and RK4 solvers (reusable) ----------

def cumulative_trapz(y, x):
    x = np.asarray(x); y = np.asarray(y)
    dx = x[1:] - x[:-1]
    trap = 0.5*(y[:-1] + y[1:]) * dx
    return np.concatenate(([0.0], np.cumsum(trap)))

def analytic_liquidity(tau_grid, a, b, c, Ifun, Rfun, ell0):
    s = tau_grid
    integrand = np.exp(-c * s) * (a * Ifun(s) - b * Rfun(s))
    integral = cumulative_trapz(integrand, s)
    ell = np.exp(c * s) * (ell0 + integral)
    return ell

def rk4_rhs(tau, ell, a, b, c, d, ell_min, ell_max, Ifun, Rfun):
    I = Ifun(np.array([tau]))[0]
    R = Rfun(np.array([tau]))[0]
    soft_lower = d * max(ell_min - ell, 0.0)
    logistic = c * ell * (1.0 - ell/ell_max)
    return (a * I - b * R) + logistic - soft_lower

def rk4_integrate(tau_grid, ell0, a, b, c, d, ell_min, ell_max, Ifun, Rfun):
    ell = np.zeros_like(tau_grid)
    ell[0] = ell0
    for i in range(1, len(tau_grid)):
        dt = tau_grid[i] - tau_grid[i-1]
        t0 = tau_grid[i-1]
        y0 = ell[i-1]
        k1 = rk4_rhs(t0, y0, a, b, c, d, ell_min, ell_max, Ifun, Rfun)
        k2 = rk4_rhs(t0 + dt/2, y0 + dt*k1/2, a, b, c, d, ell_min, ell_max, Ifun, Rfun)
        k3 = rk4_rhs(t0 + dt/2, y0 + dt*k2/2, a, b, c, d, ell_min, ell_max, Ifun, Rfun)
        k4 = rk4_rhs(t0 + dt,     y0 + dt*k3,     a, b, c, d, ell_min, ell_max, Ifun, Rfun)
        ynew = y0 + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
        # numerical clamp to avoid tiny overshoots
        if np.isnan(ynew) or np.isinf(ynew):
            ynew = max(ell_min, min(ell_max, y0))
        ell[i] = max(ell_min, min(ell_max, ynew))
    return ell

# ---------- helpers to compute I(t) and R(t) from CSV ----------

def compute_I_from_trades(df, dt_seconds=1.0):
    """
    Preferred: If df has a 'trade_volume' column that equals executed volume observed during
    each snapshot or the volume observed in between timestamps, compute cumulative V and differentiate.
    This function assumes trade_volume is volume observed in that snapshot interval (not cumulative).
    Returns I(t) in same units as 'trade_volume' per second on the df timestamps.
    """
    ts = df['timestamp_ms'].values.astype(float) / 1000.0
    # ensure trade_volume column exists
    if 'trade_volume' not in df.columns:
        raise ValueError("No trade_volume column in df.")
    vol = df['trade_volume'].astype(float).values
    # if trade_volume is per-snapshot, convert to rate by dividing by dt between rows
    # compute dt seconds between consecutive timestamps
    dt = np.diff(ts)
    # avoid zero dt
    dt[dt <= 0] = np.nanmedian(dt[dt>0]) if np.any(dt>0) else dt_seconds
    # append last dt to keep same length
    dt_full = np.concatenate(([dt[0]], dt))
    I_rate = vol / dt_full  # volume per second
    return ts, I_rate

def compute_I_proxy_from_orderbook(df, window_seconds=1.0):
    """
    Fallback proxy: use absolute change in top5_depth_sum (or total_depth) as a proxy for trading intensity.
    This will capture both trades and orderbook updates — treat as a noisy proxy.
    Returns I_proxy(t) as absolute depth change per second.
    """
    ts = df['timestamp_ms'].values.astype(float) / 1000.0
    depth = df['top5_depth_sum'].astype(float).values
    dt = np.diff(ts)
    dt[dt <= 0] = np.nanmedian(dt[dt>0]) if np.any(dt>0) else window_seconds
    delta = np.abs(np.concatenate(([0.0], np.diff(depth))))
    dt_full = np.concatenate(([dt[0] if len(dt)>0 else window_seconds], dt))
    I_proxy = delta / dt_full
    return ts, I_proxy

def compute_R_from_book(df, normalized=False):
    """
    Compute R(t) as best_ask - best_bid or normalized by mid-price.
    Missing values are filled forward/backward as needed.
    """
    # convert to numeric, handle missing
    df2 = df.copy()
    df2['best_bid'] = pd.to_numeric(df2['best_bid'], errors='coerce')
    df2['best_ask'] = pd.to_numeric(df2['best_ask'], errors='coerce')
    df2['best_bid'].ffill(inplace=True); df2['best_bid'].bfill(inplace=True)
    df2['best_ask'].ffill(inplace=True); df2['best_ask'].bfill(inplace=True)

    bid = df2['best_bid'].values.astype(float)
    ask = df2['best_ask'].values.astype(float)
    spread = ask - bid
    if normalized:
        mid = 0.5*(ask + bid)
        # avoid divide by zero
        mid[mid==0] = np.nan
        spread = spread / mid
    ts = df2['timestamp_ms'].values.astype(float) / 1000.0
    return ts, spread

# ---------- main pipeline ----------

def main(in_csv, resolution_ts_ms, ell0=0.2, a=0.1, b=1.0, c=0.0, d=0.0,
         ell_min=0.01, ell_max=1.0, use_trades=True):
    # read CSV
    df = pd.read_csv(in_csv)
    # If 'trade_volume' present and use_trades True -> use it to compute I
    try:
        if use_trades and 'trade_volume' in df.columns and df['trade_volume'].sum() > 0:
            ts, I_series = compute_I_from_trades(df)
        else:
            ts, I_series = compute_I_proxy_from_orderbook(df)
    except Exception as e:
        print("Warning: could not compute trade-based I(t). Falling back to proxy.", e)
        ts, I_series = compute_I_proxy_from_orderbook(df)

    ts_R, R_series = compute_R_from_book(df, normalized=False)

    # Convert to τ = T - t (seconds). resolution_ts_ms is ms since epoch.
    T_s = resolution_ts_ms / 1000.0
    tau_from_t = T_s - ts  # tau in seconds; you may prefer minutes/hours
    tau_R = T_s - ts_R

    # Sort by tau increasing (we want tau from 0 to T - earliest)
    # In many formulations tau = time-to-resolution, so tau small near resolution.
    # We'll build tau_grid that spans the common tau interval present in both series.
    # Use seconds; if you want hours, divide by 3600.
    # Convert arrays to numpy and sort by tau
    idx = np.argsort(tau_from_t)
    tau_I = tau_from_t[idx]
    I_sorted = I_series[idx]

    idxR = np.argsort(tau_R)
    tau_R_sorted = tau_R[idxR]
    R_sorted = R_series[idxR]

    # Build a common tau grid (e.g., linspace from min to max)
    tau_min = max(0.0, min(tau_I.min(), tau_R_sorted.min()))
    tau_max = max(tau_I.max(), tau_R_sorted.max())
    ngrid = 1000
    tau_grid = np.linspace(tau_min, tau_max, ngrid)

    # Interpolate I and R onto tau_grid (extrapolate using nearest)
    Ifun_interp = interp1d(tau_I, I_sorted, kind='linear', bounds_error=False,
                           fill_value=(I_sorted[0], I_sorted[-1]))
    Rfun_interp = interp1d(tau_R_sorted, R_sorted, kind='linear', bounds_error=False,
                           fill_value=(R_sorted[0], R_sorted[-1]))

    # vectorized call wrappers
    def Ifun_vec(tau_arr):
        return Ifun_interp(tau_arr)

    def Rfun_vec(tau_arr):
        return Rfun_interp(tau_arr)

    # Compute analytic solution (convert a,b,c per second units)
    ell_analytic = analytic_liquidity(tau_grid, a, b, c, Ifun_vec, Rfun_vec, ell0)

    # Compute bounded numeric solution
    ell_bounded = rk4_integrate(tau_grid, ell0, a, b, c, d, ell_min, ell_max, Ifun_vec, Rfun_vec)

    # Save or plot
    plt.figure(figsize=(8,4))
    plt.plot(tau_grid, ell_analytic, label='analytic')
    plt.plot(tau_grid, ell_bounded, label='bounded RK4', linestyle='--')
    plt.xlabel("τ (seconds until resolution)")
    plt.ylabel("ℓ(τ)")
    plt.legend()
    plt.title("Liquidity model: analytic vs bounded numerical")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("liquidity_model_comparison.png", dpi=200)
    print("Saved liquidity_model_comparison.png")

    # Also export tau_grid, ell_bounded, ell_analytic to CSV for later
    out_df = pd.DataFrame({
        "tau_s": tau_grid,
        "ell_analytic": ell_analytic,
        "ell_bounded": ell_bounded,
        "I": Ifun_vec(tau_grid),
        "R": Rfun_vec(tau_grid)
    })
    out_df.to_csv("liquidity_model_timeseries.csv", index=False)
    print("Saved liquidity_model_timeseries.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV from process_clob.py")
    ap.add_argument("--resolution_ts", dest="resolution_ts", required=True,
                    help="Market resolution timestamp in milliseconds since epoch (integer)")
    ap.add_argument("--ell0", dest="ell0", type=float, default=0.2)
    ap.add_argument("--a", type=float, default=0.1)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=0.0)
    ap.add_argument("--d", type=float, default=0.0)
    ap.add_argument("--ell_min", type=float, default=0.01)
    ap.add_argument("--ell_max", type=float, default=1.0)
    ap.add_argument("--no_trades", dest="use_trades", action="store_false",
                    help="Disable use of trade_volume even if present; use orderbook proxy instead")
    args = ap.parse_args()
    main(args.in_csv, int(args.resolution_ts), ell0=args.ell0, a=args.a, b=args.b, c=args.c, d=args.d,
         ell_min=args.ell_min, ell_max=args.ell_max, use_trades=args.use_trades)
