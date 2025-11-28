# Mathematical Framework: Liquidity Dynamics from Order Book Data

## 1. Introduction and Motivation

In prediction markets, liquidity—the ease with which participants can trade without moving prices—is a critical determinant of market quality. This paper develops a dynamical model for liquidity evolution using only observable order book data, without requiring direct access to trade executions.

## 2. Model Definition

### 2.1 Core Dynamical System

We model liquidity ℓ(τ) as a function of time-to-resolution τ = T - t, where T is the market resolution time and t is calendar time. The fundamental dynamical law is:

```
dℓ/dτ = a·I(τ) - b·R(τ) + c·ℓ(τ)     (1)
```

where:
- **ℓ(τ)**: Liquidity level (units: price/share) - measures price impact per unit volume
- **I(τ)**: Trading intensity (units: shares/time) - rate of market activity
- **R(τ)**: Bid-ask spread (units: price) - transaction cost
- **a, b, c**: Model parameters to be estimated from data

**Interpretation of terms:**
1. `a·I(τ)`: Trading intensity *increases* liquidity by revealing information and attracting market makers (a > 0)
2. `-b·R(τ)`: Wide spreads *reduce* liquidity growth, representing adverse selection costs (b > 0)
3. `c·ℓ(τ)`: Exponential growth/decay term (c can be positive or negative)

### 2.2 Bounded Variant

To prevent unrealistic unbounded growth, we introduce upper and lower bounds:

```
dℓ/dτ = (a·I(τ) - b·R(τ)) + c·ℓ(τ)·(1 - ℓ(τ)/ℓ_max) - d·max{ℓ_min - ℓ(τ), 0}     (2)
```

where:
- **ℓ_max**: Upper bound on liquidity (carrying capacity)
- **ℓ_min**: Lower bound on liquidity (floor)
- **d**: Restoring force parameter when ℓ drops below ℓ_min

The logistic term `c·ℓ·(1 - ℓ/ℓ_max)` provides soft upper saturation, while the max term provides a soft lower floor.

### 2.3 Analytic Solution (Linear Case)

For the linear model (1) with time-varying forcing I(τ) and R(τ), the solution is:

```
ℓ(τ) = exp(c·τ) · [ℓ_0 + ∫₀^τ exp(-c·s)·(a·I(s) - b·R(s)) ds]     (3)
```

where ℓ_0 = ℓ(0) is the initial condition.

**Derivation:**
This is a first-order linear ODE of the form dℓ/dτ - c·ℓ = a·I(τ) - b·R(τ).

Using integrating factor μ(τ) = exp(-c·τ):
```
d/dτ[μ(τ)·ℓ(τ)] = μ(τ)·(a·I(τ) - b·R(τ))
```

Integrating from 0 to τ:
```
μ(τ)·ℓ(τ) - μ(0)·ℓ_0 = ∫₀^τ μ(s)·(a·I(s) - b·R(s)) ds
```

Substituting μ(τ) = exp(-c·τ) and solving for ℓ(τ) yields equation (3).

## 3. Observable Quantities from Order Book Data

The critical challenge is that we only observe limit order book (CLOB) snapshots, not individual trade executions. We therefore construct empirical proxies for each theoretical quantity.

### 3.1 Bid-Ask Spread R(τ)

**Definition:** The spread is directly observable from the top of the book:
```
R(t) = p_ask(t) - p_bid(t)
```

**Normalized version (optional):**
```
R̃(t) = (p_ask - p_bid) / p_mid,   where p_mid = (p_ask + p_bid)/2
```

**Data requirements:** Best bid price, best ask price from each snapshot.

**Units:** Price (dollars, or unitless if normalized)

### 3.2 Liquidity ℓ(τ) via Price Impact Simulation

**Motivation:** In market microstructure theory, liquidity is inversely related to price impact. For a small trade of size Δq, the price impact is Δp ≈ λ·Δq where λ is the Kyle lambda (price impact coefficient). We identify ℓ with this impact coefficient.

**Algorithm:**
1. For each order book snapshot at time t:
   - Let p_mid = (p_ask + p_bid)/2
   - Choose a fixed simulated trade size Δq (e.g., 1 share or 0.01 × median top-5 depth)

2. **Buy-side liquidity:**
   - Walk the ask ladder, accumulating sizes until cumulative ≥ Δq
   - Compute volume-weighted average execution price p_exec_buy
   - Compute Δp_buy = p_exec_buy - p_mid
   - Liquidity measure: ℓ_buy = Δp_buy / Δq

3. **Sell-side liquidity:**
   - Walk the bid ladder, accumulating sizes until cumulative ≥ Δq
   - Compute volume-weighted average execution price p_exec_sell
   - Compute Δp_sell = p_mid - p_exec_sell
   - Liquidity measure: ℓ_sell = Δp_sell / Δq

4. **Symmetric estimate:**
   ```
   ℓ_obs(t) = (ℓ_buy + ℓ_sell) / 2
   ```

**Data requirements:** Full order book (all bid/ask price-size pairs)

**Units:** Price per share ($/share)

**Sensitivity:** Choice of Δq affects measurement. We will conduct sensitivity analysis over range [0.1, 1, 10] shares.

### 3.3 Trading Intensity I(τ)

**Challenge:** Without observing individual trades, we must infer activity from changes in the order book state.

**Proxy 1: Depth-change intensity**
```
I_depth(t) = |ΔD(t)| / Δt
```
where D(t) is aggregate depth (e.g., sum of top-5 bid + ask sizes) and Δt is time between snapshots.

**Rationale:** Large removals of depth likely indicate executions or aggressive order arrivals.

**Proxy 2: Top-level size removals**
```
I_signed(t) = [max{0, -Δbest_bid_size} + max{0, -Δbest_ask_size}] / Δt
```

**Rationale:** Negative changes at best prices indicate likely executions.

**Proxy 3: Quote update rate**
```
I_rate(t) = (# of price level changes) / Δt
```

**Rationale:** More frequent updates correlate with higher trading activity.

**Combined proxy:**
```
I(t) = γ₁·I_depth(t) + γ₂·I_signed(t) + γ₃·I_rate(t)
```

For simplicity, we begin with I_depth as the primary proxy (γ₁=1, γ₂=γ₃=0).

**Data requirements:** Time-series of order book snapshots with full depth.

**Units:** Shares per second (or shares per unit Δt)

## 4. Parameter Estimation

We estimate parameters θ = (a, b, c) for the linear model or θ = (a, b, c, d, ℓ_max, ℓ_min) for the bounded model.

### 4.1 Method A: Derivative-Residual Fit (Fast, Local)

**Approach:** Approximate the left-hand side dℓ/dτ by finite differences, then minimize squared residuals.

**Algorithm:**
1. Smooth ℓ_obs(t) with moving average or low-pass filter
2. Compute numerical derivative: ℓ̇_obs(tᵢ) ≈ (ℓ_obs(tᵢ) - ℓ_obs(tᵢ₋₁)) / Δt
3. For linear model, solve:
   ```
   min_(a,b,c) Σᵢ [ℓ̇_obs(tᵢ) - (a·I(tᵢ) - b·R(tᵢ) + c·ℓ_obs(tᵢ))]²
   ```
4. Use scipy.optimize.least_squares with bounds (a≥0, b≥0) and robust loss function (Huber or soft_l1) to reduce sensitivity to outliers.

**Advantages:** Fast, linear regression structure (for linear model)

**Disadvantages:** Numerical derivatives amplify noise; sensitive to smoothing choice

### 4.2 Method B: Trajectory Fit (Integrate-then-Fit) (Robust, Preferred)

**Approach:** For candidate parameters θ, numerically integrate the ODE from τ=0, then compare simulated trajectory to observations.

**Algorithm:**
1. Define objective function:
   ```
   L(θ) = Σᵢ [ℓ_obs(tᵢ) - ℓ_model(tᵢ; θ)]²
   ```
2. For each evaluation of L(θ):
   - Given initial condition ℓ₀ = ℓ_obs(t₀)
   - Integrate dℓ/dτ numerically (RK4 or solve_ivp) over observed time grid
   - Use observed I(t) and R(t) as time-varying inputs (interpolate between snapshots)
   - Compute ℓ_model(tᵢ; θ) at each observation time
3. Minimize L(θ) using scipy.optimize.least_squares or scipy.optimize.minimize
4. Use bounds: a≥0, b≥0, -10≤c≤10, d≥0 (if bounded model)
5. Multiple random restarts to check for local minima

**Advantages:** No derivative computation; more robust to noise; handles nonlinear bounded model

**Disadvantages:** Computationally more expensive (requires ODE solve per evaluation)

**Preferred method for paper:** Method B (trajectory fit)

## 5. Model Validation and Diagnostics

### 5.1 Goodness of Fit

**Metrics:**
- **Residual sum of squares (RSS):** Σᵢ (ℓ_obs(tᵢ) - ℓ_model(tᵢ))²
- **Coefficient of determination (R²):** 1 - RSS/TSS, where TSS = Σᵢ (ℓ_obs(tᵢ) - ℓ̄_obs)²
- **Root mean squared error (RMSE):** sqrt(RSS / N)
- **Mean absolute percentage error (MAPE):** (1/N) Σᵢ |ℓ_obs(tᵢ) - ℓ_model(tᵢ)| / ℓ_obs(tᵢ)

**Plots:**
- Observed vs. predicted ℓ(t) time series (train and test sets)
- Scatter plot: ℓ_model vs. ℓ_obs with 45° reference line

### 5.2 Residual Analysis

**Tests:**
1. **Plot residuals vs. time:** Check for temporal patterns or heteroskedasticity
2. **Plot residuals vs. I(t) and R(t):** Check for missed nonlinear effects
3. **Autocorrelation function (ACF):** Check for serial correlation (Ljung-Box test)
4. **Normality:** Q-Q plot and Shapiro-Wilk test (for inference validity)
5. **Heteroskedasticity:** Plot |residuals| vs. fitted values; Breusch-Pagan test

### 5.3 Parameter Uncertainty

**Bootstrap procedure:**
1. Block-bootstrap the time series (block size = 10-20 snapshots to preserve temporal dependence)
2. For each bootstrap sample (B=1000 replicates):
   - Re-estimate parameters θ̂ₖ
3. Compute 95% confidence intervals from bootstrap distribution (percentile or BCa method)
4. Report parameter estimates as: θ̂ ± 1.96·SE(θ̂)

### 5.4 Sensitivity Analysis

**Test sensitivity to:**
1. **Simulated trade size Δq:** Vary over [0.1, 1, 10] shares; plot ℓ_obs(Δq) and fitted parameters
2. **Choice of I(t) proxy:** Compare models using I_depth, I_signed, I_rate, and combined
3. **Smoothing window:** Vary moving average window from 1 to 20 snapshots
4. **Market conditions:** Stratify by time-to-resolution (early vs. late market life)

### 5.5 Out-of-Sample Validation

**Procedure:**
1. Split data: 70% train, 30% test (chronological split to respect time structure)
2. Fit parameters on training set
3. Integrate model forward on test set using test-set I(t) and R(t)
4. Compute test-set RMSE and R²
5. Compare in-sample vs. out-of-sample performance

**Alternative:** Use multiple markets and cross-validate across markets

### 5.6 Model Comparison

**Compare:**
1. Linear model (1) vs. bounded model (2)
2. Different I(t) proxy choices
3. Null model: dℓ/dτ = 0 (constant liquidity baseline)

**Criteria:**
- **Akaike Information Criterion (AIC):** 2k - 2ln(L), where k = # parameters, L = likelihood
- **Bayesian Information Criterion (BIC):** k·ln(N) - 2ln(L)
- **Predictive MSE** on test set

Prefer model with lowest AIC/BIC and best out-of-sample performance.

## 6. Limitations and Caveats

1. **Proxy validity:** I(t) is not directly observed; our proxies confound trade executions with order cancellations and replacements.

2. **Causality:** The model describes correlation and dynamics but does not establish causal mechanisms. Market maker behavior, information arrival, and strategic trading are unmodeled.

3. **Microstructure noise:** Order book snapshots contain noise from fleeting quotes and spoofing, which can bias ℓ_obs.

4. **Parameter identifiability:** If I(t) and R(t) are highly correlated (collinear), parameters a and b may be poorly identified. Check variance inflation factors (VIF).

5. **Market heterogeneity:** Different markets may have different dynamics; pooling requires care.

6. **Time resolution:** Snapshot frequency (Δt ~ 5-10s) may miss high-frequency dynamics.

7. **ℓ definition:** Our ℓ_obs depends on choice of Δq and measures *local* price impact, not global market depth.

## 7. Data Pipeline Summary

**Input:** NDJSON file of order book snapshots (timestamp, bids, asks)

**Step 1:** Parse snapshots → CSV with (timestamp_ms, best_bid, best_bid_size, best_ask, best_ask_size, top5_depth_sum, ...)

**Step 2:** Compute observables for each snapshot:
- R(t) = best_ask - best_bid
- ℓ_obs(t) via price impact simulation (walk-the-book algorithm)
- I(t) via depth-change proxy

**Step 3:** Convert to τ = T - t (time-to-resolution) and resample to regular grid

**Step 4:** Smooth I(t) and R(t) lightly (3-point moving average)

**Step 5:** Fit parameters using Method B (trajectory fit with scipy.optimize)

**Step 6:** Validate: compute diagnostics, plot figures, bootstrap confidence intervals

**Output:**
- Fitted parameters with confidence intervals
- Goodness-of-fit statistics (R², RMSE, AIC)
- Plots for paper (time series, residuals, sensitivity, out-of-sample)

## 8. Expected Results

**Hypotheses to test:**
1. **H1:** a > 0 (trading intensity increases liquidity)
2. **H2:** b > 0 (wider spreads reduce liquidity growth)
3. **H3:** c ≈ 0 or c < 0 (liquidity decays or remains stable absent external forcing)
4. **H4:** Bounded model (2) fits better than linear model (1) for long time horizons

**Interpretation:**
- Large |a| suggests markets respond strongly to trading activity
- Large b suggests adverse selection is important
- Negative c suggests liquidity mean-reverts; positive c suggests momentum

## 9. Paper Structure Outline

**Title:** "Dynamics of Liquidity in Prediction Markets: A Differential Equation Approach Using Order Book Data"

**Abstract:** (200 words summarizing motivation, model, data, and key findings)

**1. Introduction**
- Motivation: importance of liquidity in prediction markets
- Research question: can we model liquidity dynamics using only order book data?
- Contributions: novel proxy construction, empirical validation

**2. Related Work**
- Market microstructure literature (Kyle 1985, Glosten-Milgrom, etc.)
- Liquidity measurement (bid-ask spreads, price impact, depth)
- Prediction market microstructure (if prior work exists)

**3. Theoretical Framework**
- Section 3.1: Model derivation (equations 1-3)
- Section 3.2: Interpretation of parameters
- Section 3.3: Analytic solution

**4. Data and Observable Construction**
- Section 4.1: Data source (Polymarket CLOB)
- Section 4.2: Order book structure
- Section 4.3: Proxy construction for R, ℓ, I (detailed algorithms)
- Section 4.4: Descriptive statistics

**5. Estimation Methodology**
- Section 5.1: Trajectory-fit approach (Method B)
- Section 5.2: Optimization procedure
- Section 5.3: Parameter bounds and regularization

**6. Results**
- Section 6.1: Parameter estimates with confidence intervals
- Section 6.2: Goodness of fit (R², RMSE)
- Section 6.3: Model comparison (linear vs. bounded)
- Section 6.4: Sensitivity analysis

**7. Model Validation**
- Section 7.1: Residual diagnostics
- Section 7.2: Out-of-sample performance
- Section 7.3: Bootstrap uncertainty quantification

**8. Discussion**
- Economic interpretation of findings
- Comparison to prior theory
- Limitations (Section 6 of this document)

**9. Conclusion**
- Summary of contributions
- Future work: extend to multi-asset, incorporate news events, test on other markets

**References**

**Appendix A:** Derivation details

**Appendix B:** Robustness checks (alternative proxies, smoothing, markets)

**Appendix C:** Code availability statement

---

## 10. Software Implementation Overview

All analysis will be implemented in Python 3.10+ using:
- **NumPy/Pandas:** Data manipulation
- **SciPy:** ODE integration (solve_ivp), optimization (least_squares, minimize)
- **Matplotlib/Seaborn:** Plotting
- **Statsmodels:** Diagnostic tests (ACF, Ljung-Box, heteroskedasticity tests)
- **scikit-learn:** Block bootstrap (via TimeSeriesSplit or custom)

**Key modules:**
1. `observables.py`: Compute R(t), ℓ(t), I(t) from order book snapshots
2. `model.py`: Define ODE right-hand side; integrate trajectories
3. `estimation.py`: Implement Method A and Method B parameter fitting
4. `validation.py`: Goodness-of-fit, residual tests, bootstrap, cross-validation
5. `plotting.py`: Generate all figures for paper
6. `pipeline.py`: End-to-end script from NDJSON → fitted model → plots

---

**End of Mathematical Framework Document**
