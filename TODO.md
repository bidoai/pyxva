# TODO / Improvements

Tracked improvements and known gaps in the `pyxva` library.
Items are grouped by area; priority labels: `[P1]` urgent / `[P2]` important / `[P3]` nice-to-have.

---

## Pipeline

- [x] **[P1] Wire `SimulationSharedMemory` into `RiskEngine` parallel path.**
  `_run_parallel()` currently pickles the full `simulation_results` dict into each
  worker process.  Replace with `SimulationSharedMemory` to avoid the O(workers × data)
  memory blow-up described in DESIGN.md §9.

- [x] **[P1] Remove dead code: `_make_aggregate_ns()` in `pipeline/engine.py`.**
  The function raises `NotImplementedError` and is no longer referenced.

- [x] **[P2] `pipeline/persistence.py` — lazy RunResult loading.**
  Implemented in v1.1. `RunResult.from_parquet(path, use_duckdb=False)` loads scalars
  eagerly and profile arrays via pandas/DuckDB. Optional DuckDB dependency for queries.

- [x] **[P2] Support `StatefulPricer` instruments inside `RiskEngine`.**
  `NettingSet._price_trade()` calls `pricer.price(result)` which is correct for both
  plain and stateful pricers, but it tries all models in order until one succeeds.  For
  stateful pricers this is unnecessarily expensive and can produce wrong results if an
  incompatible model "succeeds" by accident.  `Trade.model_name` should be used to look
  up the right `SimulationResult` directly (see netting set improvement below).

- [x] **[P3] `TradeFactory`: auto-register `BarrierOption` and other built-in exotics.**
  Users currently have to call `@TradeFactory.register("BarrierOption")` manually.
  The built-in exotic pricers should be registered by default on import.

---

## Exposure / Netting

- [x] **[P1] `NettingSet._price_trade()` model lookup is fragile.**
  It iterates all models until one doesn't raise, so the first model whose factor names
  happen to match wins — even if it is the wrong model.  Fix: `NettingSet` should
  store trades as `Trade` objects (not `(str, Pricer)` tuples) so that `trade.model_name`
  can be used to retrieve the correct `SimulationResult` directly.

- [x] **[P2] `StreamingExposureEngine` MPOR calculation is a proxy.**
  Fixed in v1.2: replaced proxy with proper CSB ring buffer.
  `CE_mpor(t) = max(V(t) - CSB(t - mpor), 0)` where `CSB(t - mpor)` is
  the settled collateral from `mpor_steps` time steps ago.

- [x] **[P2] `REGVMStepper` rounding convention.**
  The current implementation rounds to the nearest multiple.  ISDA 2016 VM CSA specifies
  that delivery amounts round *up* and return amounts round *down*.  Fix the rounding
  branches accordingly.

- [x] **[P3] `StreamingExposureEngine`: support multi-model netting sets.**
  Implemented in v1.2: `run()` now accepts `SimulationResult | dict[str, SimulationResult]`.
  `Trade` objects with `model_name` route to the correct result; legacy `(id, pricer)` tuples
  remain supported for backwards compatibility.

---

## XVA

- [x] **[P1] Asymmetric FVA (borrow vs lend spread).**
  Implemented in v1.2: `fva_approx(borrow_spread, lend_spread=None)`.
  Backward-compatible via legacy `funding=` keyword. Propagated through
  `bilateral_summary()`, `ISDAExposureCalculator.run()`, `AgreementConfig`, and
  `xva_attribution()`. `lend_spread` field added to `AgreementConfig` and YAML parsing.

- [x] **[P1] Path-dependent KVA EAD profile.**
  Implemented in v1.2: `SACCRCalculator.ead_profile(time_grid, mean_mtm_by_trade)`
  computes EAD at each time step with declining residual maturity.
  `kva_approx(ead, ...)` now accepts `float | np.ndarray`; array form uses
  trapezoidal integration `CoC × ∫ EAD(t) dt`.

- [ ] **[P2] SIMM-based `im_time_profile`.**
  Current `REGIMEngine.im_time_profile()` uses Schedule IM. For Phase 5/6 firms
  under UMR, SIMM sensitivities produce a different declining profile. Requires
  sensitivity input interface design before implementation.

- [x] **[P2] FVA (Funding Valuation Adjustment).**
  Implemented in v1.1. `BilateralExposureCalculator.fva_approx()` accepts
  `float | HazardCurve` funding spread. `AgreementResult.fva` and `RunResult.total_fva` added.

- [x] **[P2] MVA (Margin Valuation Adjustment).**
  Implemented in v1.1. `BilateralExposureCalculator.mva_approx()` + `REGIMEngine.im_time_profile()`
  produce the declining IM profile. `AgreementResult.mva` and `RunResult.total_mva` added.

- [x] **[P3] KVA (Capital Valuation Adjustment).**
  Implemented in v1.1 using flat t=0 SA-CCR EAD profile (documented approximation).
  `BilateralExposureCalculator.kva_approx()`. `AgreementResult.kva` and `RunResult.total_kva` added.

---

## Pricing

- [x] **[P2] `EuropeanOption.price_at()` override.**
  `EuropeanOption` currently falls back to the default `price(result)[:, t_idx]`.
  An efficient override computing only the `(n_paths,)` Black-Scholes MTM from the spot
  slice would be consistent with the IRS/bond overrides.

- [x] **[P2] `BarrierOption` pre-expiry MTM.**
  Currently `step()` returns 0 for `t < expiry`.  A more useful implementation would
  return the Black-Scholes barrier option price at `t` (analytical formula exists),
  making the MTM profile smooth and usable for EE/PFE pre-expiry.

- [x] **[P3] Asian option pricer.**
  Good second example of `StatefulPricer` (state = running arithmetic average).
  Payoff = `max(avg(S) - K, 0)` at expiry; interim MTM via Monte Carlo sub-simulation
  or control-variate approximation.

---

## Models

- [x] **[P2] Verify `HestonModel.interpolation_space` returns `["log", "linear"]`.**
  The variance factor `v` should be interpolated in linear space (it is mean-reverting
  and can approach zero), while the spot `S` should be log-space.  Confirm this is
  correctly declared and tested.

- [x] **[P3] Multi-curve Hull-White (HullWhite2F).**
  Second mean-reverting factor captures a richer term structure of interest rate
  volatility.  Useful for swaption calibration and longer-dated exposure.

- [ ] **[P3] Local volatility / SABR for FX/equity.**
  Current GBM and GK models use constant volatility.  A local vol surface would
  improve calibration accuracy for vanilla option portfolios.

---

## Core

- [x] **[P2] `SimulationResult.at(t)` boundary handling.**
  Fixed in v1.2: `at()` now raises `ValueError` with a descriptive message when
  `t < time_grid[0]` or `t > time_grid[-1]` (within 1e-9 tolerance).

- [x] **[P3] `SparseTimeGrid.dt()` method.**
  Already implemented as a static method returning `np.diff(grid)`. No further action needed.

---

## Testing

- [x] **[P2] Integration test: `RiskEngine` end-to-end with `StatefulPricer`.**
  Add a `BarrierOption` trade to a YAML config and run the full pipeline to confirm
  `NettingSet` + `Agreement` + `RiskEngine` handle stateful pricers correctly.

- [x] **[P2] Property-based tests for `REGVMStepper`.**
  CSB should be non-negative when `threshold=0`, `mta=0`; post-margin CE should equal 0
  when `V(t) > 0`; CSB should never decrease below `ia_counterparty - ia_party`.

- [x] **[P2] Regression tests for `_discount_factors()` against known Hull-White prices.**
  ZCB prices computed via the affine formula should match the closed-form
  `P(0,T) * exp(B*f(0,t) - ...)` expression to machine precision.

- [x] **[P3] Benchmark: streaming vs batch EE profiles match to <1bp.**
  `StreamingExposureEngine` and `ISDAExposureCalculator` should produce identical (up to
  float precision) EE profiles for plain vanilla swaps with zero threshold/MTA.

- [x] **[P2] `calibrate_to` missing curve should raise, not warn.**
  When a model config's `calibrate_to` key references a curve not present in
  `MarketData`, `pipeline/engine.py` logs a warning and continues with an uncalibrated
  model. An uncalibrated Hull-White model produces systematically wrong CVA/EE profiles.
  Fix: raise `ValueError` with a clear message identifying the missing curve name.
  Add a test that asserts the error is raised (vs. silently proceeding).

---

## Documentation / DX

- [x] **[P3] Example YAML configs in `examples/`.**
  Ship a `single_swap.yaml`, `multi_asset.yaml`, and `stress_test.yaml` that users can
  run immediately with `uv run pyxva-demo`.

- [x] **[P3] Changelog (`CHANGELOG.md`).**
  Track breaking changes and notable additions per version so library consumers know
  what to expect on upgrade.
