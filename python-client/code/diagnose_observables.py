#!/usr/bin/env python3
"""
diagnose_observables.py

Quick diagnostic to check if observable computation is working correctly.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python diagnose_observables.py <path_to_observables.csv>")
    sys.exit(1)

csv_path = sys.argv[1]

print("=" * 80)
print("OBSERVABLE DIAGNOSTICS")
print("=" * 80)

df = pd.read_csv(csv_path)

print(f"\nLoaded {len(df)} snapshots")
print(f"Columns: {list(df.columns)}")

# Check for key columns
required = ['tau_sec', 'R', 'ell_avg', 'I']
for col in required:
    if col not in df.columns:
        print(f"ERROR: Missing column '{col}'")
        sys.exit(1)

print("\n" + "=" * 80)
print("LIQUIDITY (ell_avg)")
print("=" * 80)
print(df['ell_avg'].describe())
print(f"Non-NaN: {df['ell_avg'].notna().sum()}")
print(f"Zeros: {(df['ell_avg'] == 0).sum()}")
print(f"Negative: {(df['ell_avg'] < 0).sum()}")
print(f"Infinite: {np.isinf(df['ell_avg']).sum()}")

print("\n" + "=" * 80)
print("SPREAD (R)")
print("=" * 80)
print(df['R'].describe())
print(f"Non-NaN: {df['R'].notna().sum()}")
print(f"Zeros: {(df['R'] == 0).sum()}")

print("\n" + "=" * 80)
print("INTENSITY (I)")
print("=" * 80)
print(df['I'].describe())
print(f"Non-NaN: {df['I'].notna().sum()}")
print(f"Zeros: {(df['I'] == 0).sum()}")
print(f"Infinite: {np.isinf(df['I']).sum()}")

print("\n" + "=" * 80)
print("TIME-TO-RESOLUTION (tau_sec)")
print("=" * 80)
print(df['tau_sec'].describe())
print(f"Non-NaN: {df['tau_sec'].notna().sum()}")
print(f"Negative (BUG!): {(df['tau_sec'] < 0).sum()}")

print("\n" + "=" * 80)
print("FIRST 20 ROWS")
print("=" * 80)
print(df[['tau_sec', 'R', 'ell_avg', 'I']].head(20).to_string())

print("\n" + "=" * 80)
print("LAST 20 ROWS")
print("=" * 80)
print(df[['tau_sec', 'R', 'ell_avg', 'I']].tail(20).to_string())

# Plot
print("\nGenerating diagnostic plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Liquidity over tau
ax = axes[0, 0]
valid = df['ell_avg'].notna() & np.isfinite(df['ell_avg'])
ax.scatter(df.loc[valid, 'tau_sec'], df.loc[valid, 'ell_avg'], alpha=0.5, s=10)
ax.set_xlabel('tau (seconds to resolution)')
ax.set_ylabel('Liquidity (ell_avg)')
ax.set_title('Liquidity vs Time-to-Resolution')
ax.grid(True, alpha=0.3)

# Plot 2: Spread over tau
ax = axes[0, 1]
valid = df['R'].notna() & np.isfinite(df['R'])
ax.scatter(df.loc[valid, 'tau_sec'], df.loc[valid, 'R'], alpha=0.5, s=10)
ax.set_xlabel('tau (seconds to resolution)')
ax.set_ylabel('Spread (R)')
ax.set_title('Spread vs Time-to-Resolution')
ax.grid(True, alpha=0.3)

# Plot 3: Intensity over tau
ax = axes[1, 0]
valid = df['I'].notna() & np.isfinite(df['I'])
ax.scatter(df.loc[valid, 'tau_sec'], df.loc[valid, 'I'], alpha=0.5, s=10)
ax.set_xlabel('tau (seconds to resolution)')
ax.set_ylabel('Intensity (I)')
ax.set_title('Intensity vs Time-to-Resolution')
ax.grid(True, alpha=0.3)

# Plot 4: Histogram of liquidity
ax = axes[1, 1]
valid = df['ell_avg'].notna() & np.isfinite(df['ell_avg'])
ax.hist(df.loc[valid, 'ell_avg'], bins=50, alpha=0.7, edgecolor='black')
ax.set_xlabel('Liquidity (ell_avg)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Liquidity Values')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = csv_path.replace('.csv', '_diagnostics.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved diagnostic plot to: {output_path}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
