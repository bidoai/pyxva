# Design Notes

Key architectural decisions behind `risk_analytics` — the *why* behind the choices,
not just the *what*.

---

## 1. Sparse time grid with cashflow merging

**Choice:** The standard grid uses daily steps for the first two weeks, weekly steps to
one year, and monthly steps thereafter.  All known cashflow dates are unconditionally
merged as hard nodes before simulation.

**Why:** Monte Carlo exposure calculations require time resolution near the valuation date
(short MPOR windows, daily margin calls) but the absolute number of paths per step
dominates runtime and memory.  The sparse schedule gives ~170 nodes for a 30-year deal
vs ~360 on a uniform monthly grid.

Cashflow merging matters because instrument MTM has a discontinuity at each payment date
— the floating leg resets, coupons settle.  If a payment date falls between two sparse
nodes, interpolation across that gap produces systematically wrong exposures (the MTM
drop at payment looks like a smooth drift).  Hard nodes prevent this at zero extra cost
during simulation.

---

## 2. Per-factor interpolation space ("log" vs "linear")

**Choice:** `StochasticModel.interpolation_space` declares whether each factor should be
interpolated in log-space (`"log"`) or linear-space (`"linear"`).  `SimulationResult.at(t)`
uses this to interpolate between sparse grid nodes on demand.

**Why:** Equity spot prices and FX rates are log-normally distributed: interpolating
linearly between two nodes on a GBM path introduces a downward bias (Jensen's inequality)
because the log-normal distribution has a positively skewed shape.  Interpolating in
log-space recovers the geometric mean, which is the correct conditional expectation.
Short rates (Hull-White) are Gaussian so linear interpolation is exact for the first two
moments.

---

## 3. Hull-White affine discount factor P(t,T) = A·exp(−B·r)

**Choice:** Rate pricers (swap, ZCB, fixed-rate bond) use the closed-form Hull-White
discount factor rather than the naive `exp(-r·(T-t))` flat-curve approximation.

**Why:** Under Hull-White the conditional distribution of `r(T)` given `r(t)` is Gaussian
with mean depending on mean reversion.  The simple `exp(-r·τ)` formula ignores mean
reversion and the convexity correction: it systematically over-discounts when rates are
high and under-discounts when they are low, producing biased EE profiles.  The affine
formula `A(t,T)·exp(-B(t,T)·r(t))` is exact for the conditional expectation
`E[exp(-∫_t^T r(s)ds) | r(t)]` under the Hull-White model.

When the model has been calibrated to an initial yield curve, `A(t,T)` uses
`P(0,T)/P(0,t)` and the instantaneous forward `f(0,t)` to fit the initial term structure
exactly.  Without a curve, the formula falls back to the Vasicek analytical form using
`r0` as the long-run mean.

---

## 4. PathState / StatefulPricer separation

**Choice:** State is a separate `PathState` dataclass; the pricer receives it, returns an
updated copy, and never mutates in-place.

**Why:** Immutable (functional) state makes the step loop easy to reason about, test,
and parallelise.  There is no hidden mutation between steps, which matters for:

- **Testing:** you can call `step()` directly with a constructed state and compare
  against a known expected result without running a full simulation.
- **Price-at replay:** `price_at(result, t_idx)` replays from t=0 by calling `step()`
  in sequence.  If state were stored on the pricer, re-entrancy would require cloning
  the entire pricer; with separate state the pricer itself is stateless and reusable.
- **Parallel repricing:** multiple workers can call the same pricer instance
  simultaneously because there is no shared mutable state on it.

---

## 5. price_at() with efficient overrides

**Choice:** `Pricer.price_at(result, t_idx)` has a default that calls
`price(result)[:, t_idx]`.  Vanilla rate pricers override it to compute only the
requested slice.

**Why:** The default is correct and requires no boilerplate from instrument authors.  But
for the most common instruments (IRS, bonds) we can compute the `(n_paths,)` MTM at a
single step using only the `r_t` factor slice and the HW discount factors — without
ever allocating the `(n_paths, T)` matrix.  `StreamingExposureEngine` calls `price_at()`
at each step, so the override reduces per-step allocation from O(n_paths × T) to
O(n_paths × k) where k is the number of future payment dates.

`StatefulPricer` cannot safely return just the last step without replaying from t=0
(the barrier state accumulated along the path is load-bearing), so its `price_at()`
replays the full step loop — this is the documented contract.

---

## 6. StreamingExposureEngine vs batch ISDAExposureCalculator

**Choice:** Two distinct exposure engines co-exist.

| | `ISDAExposureCalculator` | `StreamingExposureEngine` |
|---|---|---|
| Input | full `(n_paths, T)` MTM | time-step loop |
| Memory | O(n_paths × T) | O(n_paths) |
| StatefulPricers | no (needs full matrix upfront) | yes |
| Collateral model | full REGVMEngine | REGVMStepper |
| Use case | pipeline engine, batch runs | large portfolios, exotics |

**Why two:** The batch engine is simpler to reason about and already integrated into the
3-phase pipeline.  The streaming engine is the right tool when either (a) memory is the
binding constraint (large bank scale: thousands of agreements, millions of paths) or (b)
path-dependent instruments require state that cannot be decomposed into a static MTM
matrix.  Providing both preserves backward compatibility and lets users choose based on
their actual constraints.

---

## 7. REGVMStepper: per-path CSB state

**Choice:** `REGVMStepper` maintains a `(n_paths,)` Credit Support Balance array that
advances one step at a time.

**Why:** Variation margin under an ISDA CSA is inherently path-dependent: whether a call
is made depends on the current CSB, which depends on all past margin calls.  A batch
engine that computes the full MTM matrix first and then applies margining in a second
pass can do the same thing; the stepper version is the natural fit for the streaming
engine where you never have the full matrix.

The CSB state is kept as a plain numpy array rather than a `PathState` subclass because
`REGVMStepper` is not a pricer — it is a margining filter applied after pricing.  Mixing
the two concerns into a single `PathState` would make the instrument model responsible
for the CSA terms, which would break the separation of concerns.

---

## 8. Agreement: VM on aggregate MTM, no per-netting-set floor

**Choice:** `Agreement.aggregate_mtm()` returns the raw sum of netting set MTMs with no
`max(..., 0)` floor applied.  VM is computed on the aggregate.

**Why:** This is the correct ISDA mechanics.  A single CSA (Credit Support Annex) governs
the combined net obligation across all netting sets under one ISDA Master Agreement.  The
counterparty delivers (or returns) variation margin on the *net* position, not on each
netting set independently.  Applying a floor before aggregation would overstate the
margin requirement (it would treat each netting set as if it had its own independent
CSA).

Pre-collateral EE/PFE are still computed per netting set (using `max(V, 0)`) because
those metrics represent close-out exposure in a netting set — the bilateral agreement
allows offsetting within a netting set but not across them.

---

## 9. SimulationSharedMemory: zero-copy via multiprocessing.shared_memory

**Choice:** Simulation paths are placed in named `SharedMemory` blocks; worker processes
receive lightweight descriptor dicts (picklable, ~100 bytes each) and attach numpy views
into the shared buffer.

**Why:** At large-bank scale a single simulation run might produce 100,000 paths × 200
time steps × 5 models × 8 bytes ≈ 800 MB.  `ProcessPoolExecutor` would normally pickle
this entire array into each worker's process address space, multiplying memory usage by
the number of workers.  `SharedMemory` maps the same physical pages into each worker,
keeping total memory roughly constant regardless of parallelism level.

The context manager (`__enter__` / `__exit__`) ensures all blocks are `unlink()`ed even
if a worker crashes — leaked shared memory blocks would otherwise persist until system
reboot.  Workers call `attach()` / `detach()` so the owning process remains the
canonical owner and unlinker.

---

## 10. MarketData immutability: bump() / scenario() return new copies

**Choice:** `MarketData.bump()` and `MarketData.scenario()` never modify the receiver;
they return a new `MarketData` instance with the bumped values.

**Why:** Immutability makes stress-test workflows safe to compose.  If bumps mutated the
base object, running two scenarios concurrently (e.g. in a parallel stress-test loop)
would race.  With new copies, the base `MarketData` is shared read-only, and each
scenario worker holds its own independent copy.  It also makes it trivial to compare
base vs stressed results: you hold both objects simultaneously without any bookkeeping.

---

## 11. TradeFactory.register(): class-level registry

**Choice:** `@TradeFactory.register("TypeName")` stores a `(params) -> Pricer` builder
function in a `dict` on the class.  It is checked before the built-in if/elif chain.

**Why:** The built-in chain handles the four canonical vanilla types.  A registry
decorator lets users add any instrument without modifying library code or subclassing
`TradeFactory`.  The custom type is then usable in YAML configs the same way as built-in
types — no special syntax.

The registry is a class-level `dict` rather than a module-level variable so that it
travels with `TradeFactory` when the class is imported across module boundaries, and so
that tests can `clear()` it in `setup_method` without affecting other tests.

---

## 12. RiskEngine: serial below 2 agreements, parallel above

**Choice:** `RiskEngine._run_exposure_phase()` uses serial execution for ≤ 2 agreements
and `ProcessPoolExecutor` for more.

**Why:** `ProcessPoolExecutor` has a fixed startup overhead (spawning worker processes,
pickling arguments).  For a single agreement or a quick smoke-test the overhead dominates
and the parallel path is slower.  The threshold of 2 is a practical heuristic: a
single-counterparty run (common during development) stays fast; a production run with
dozens of counterparties gets full parallelism.

The split between phases 1 (simulate), 2 (price), and 3 (aggregate) reflects where the
data dependencies lie.  Simulation is serial because all models share the same correlated
Cholesky draw.  Pricing is independent per agreement (each agreement only reads the
simulation output, never writes it).  Aggregation is serial because it is O(n_agreements)
arithmetic.

---

## 13. _ModelWrapper: user-defined model names for SimulationResult keying

**Choice:** `_ModelWrapper` wraps a `StochasticModel` and overrides its `.name` property
with the user-supplied name from the config (e.g. `"rates_usd"` instead of
`"HullWhite1F"`).

**Why:** Multiple agreements in the same run may reference the same model type on
different currencies (e.g. `"rates_usd"` and `"rates_eur"`, both `HullWhite1F`).
`MonteCarloEngine` keys `SimulationResult`s by `model.name`; if the name is the class
name, only the last model of each type survives.  Using the user-defined config name as
the key allows arbitrarily many models of the same type.  Trade configs reference models
by this same user-defined name (`model: rates_usd`), closing the loop.
