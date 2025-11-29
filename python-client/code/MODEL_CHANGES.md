# Model Redesign: Spread Dynamics Instead of Liquidity

## Summary

We've switched from modeling **liquidity** to modeling **spread** because:
1. Liquidity was circular (ℓ = R/V, but R was also in the model)
2. Spread is directly observable and non-circular
3. Spread is a standard market quality measure

## New Model

### **Mathematical Formulation**

**Old Model:**
```
dℓ/dτ = a·I(τ) - b·R(τ) + c·ℓ(τ)
```

**New Model:**
```
dR/dτ = a·I(τ) - b·D(τ) + c·R(τ)
```

Where:
- **R(τ)**: Bid-ask spread (DEPENDENT VARIABLE - what we're modeling)
- **I(τ)**: Trading intensity (depth changes per second)
- **D(τ)**: Market depth (top-5 bid+ask depth sum)
- **a, b, c**: Parameters to estimate

### **Interpretation**

- `a·I(τ)`: High trading activity **widens** spread (depletes liquidity)
  - More trading → more inventory risk → wider spreads
  - **Expected:** a > 0

- `-b·D(τ)`: High depth **narrows** spread (improves liquidity)
  - More depth → easier to trade → narrower spreads
  - **Expected:** b > 0

- `c·R(τ)`: Exponential growth/decay
  - c < 0: Spreads mean-revert
  - c > 0: Spreads have momentum
  - **Expected:** c ≈ 0 or c < 0

## Code Changes

### Already Updated

1. **`model.py`**:
   - `ode_linear()`: Changed from (ell, I_func, R_func) → (R, I_func, D_func)
   - `integrate_ode_linear()`: Changed from (ell0, R_func) → (R0, D_func)

2. **`observables.py`**:
   - `compute_all_observables()`: Now computes R, D, I instead of R, ℓ, I
   - R = bid-ask spread (dependent variable)
   - D = top5_depth_sum (independent variable)
   - I = trading intensity (independent variable)

### Still Need to Update

3. **`estimation.py`**:
   - `fit_linear_method_a()`: Change ell → R, R_obs → D_obs
   - `fit_linear_method_b()`: Change ell → R, R_obs → D_obs, R_func → D_func

4. **`pipeline.py`**:
   - Extract R, D, I instead of R, ℓ, I
   - Change variable names throughout
   - Update plots to show "Spread" instead of "Liquidity"

5. **`validation.py`**: Should work as-is (just operates on residuals)

6. **`plotting.py`**: Update axis labels from "Liquidity" → "Spread"

## Alternative: Just fix estimation.py and pipeline.py variable names

The quickest fix is to:
1. In `estimation.py` and `pipeline.py`, rename all `ell` → `R_pred`
2. Rename `R` → `D`
3. Everything else stays the same

This way we're just treating the spread as the "thing we're modeling" and depth as an input.
