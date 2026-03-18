# Changelog

All notable changes to `pyxva` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-03-18

### Added

- **HazardCurve** (`src/pyxva/exposure/hazard_curve.py`): term-structure of
  default or funding spreads with piecewise-constant hazard rates.
  - `from_flat_spread(spread, lgd)` — one-bucket flat curve
  - `from_tenors(tenors, hazard_rates)` — explicit bucketed curve
  - `calibrate(cds_tenors, cds_spreads, recovery)` — closed-form bootstrap
    from CDS par spreads (no external solver required)
  - `survival_probability(t)` and `marginal_default_prob(t_prev, t)` queries

- **FVA** (`BilateralExposureCalculator.fva_approx`): funding cost on
  uncollateralised exposure. Accepts `float | HazardCurve` funding spread.
  `AgreementResult.fva` and `RunResult.total_fva` fields.

- **MVA** (`BilateralExposureCalculator.mva_approx`): cost of funding posted
  initial margin. Uses `REGIMEngine.im_time_profile()` to produce the declining
  IM profile (residual maturity drops to zero as trades approach maturity).
  `AgreementResult.mva` and `RunResult.total_mva` fields.

- **KVA** (`BilateralExposureCalculator.kva_approx`): regulatory capital cost
  using flat t=0 SA-CCR EAD profile. `AgreementResult.kva` and
  `RunResult.total_kva` fields. Documented approximation; path-dependent EAD
  profile deferred to v1.2.

- **xVA attribution waterfall** (`BilateralExposureCalculator.xva_attribution`):
  per-time-bucket breakdown of CVA, DVA, FVA, and MVA contributions. Returns
  `{'time', 'cva', 'dva', 'fva', 'mva', 'total'}` — each a `(T-1,)` array.

- **`_integral_xva()` shared helper**: replaces `_integral_cva()` in
  `bilateral.py`. All four xVA integrals (CVA, DVA, FVA, MVA) share one
  `lgd × Σ profile(t_i) × [Q(t_{i-1}) − Q(t_i)]` implementation. Accepts
  `float | HazardCurve`.

- **`REGIMEngine.im_time_profile(trades, time_grid)`**: time-varying Schedule IM
  profile where each trade's residual maturity declines as `max(0, mat − t)`.
  Naturally drops to zero at maturity. Used as E[IM(t)] in the MVA integral.

- **`AgreementResult.xva_summary()`**: returns a labelled dict of all xVA
  scalars including `total_xva = CVA − DVA + FVA + MVA + KVA`.

- **`RunResult.from_parquet(path, use_duckdb=False)`**: load a `RunResult`
  persisted by `to_parquet()`. Scalars loaded eagerly; profile arrays loaded
  via pandas (or DuckDB when `use_duckdb=True`). DuckDB is an optional
  dependency (`pip install duckdb`).

- **`AgreementConfig.funding_spread`** and **`AgreementConfig.cost_of_capital`**
  fields (both optional; `cost_of_capital` defaults to 10%). Parsed from YAML.
  Wired through `ISDAExposureCalculator.run()` → `bilateral_summary()`.

### Changed

- `BilateralExposureCalculator.cva_approx()` and `dva_approx()` now accept
  `float | HazardCurve` for the hazard rate argument (backwards-compatible:
  existing scalar callers unchanged).

- `ISDAExposureCalculator.run()` gains `funding_spread` and `cost_of_capital`
  parameters (both default to `None`/`0.10` — no behaviour change for existing
  callers).

- `AgreementResult.to_dict()` now includes `fva`, `mva`, `kva` keys.

- `RunResult.to_dict()` now includes `total_fva`, `total_mva`, `total_kva` keys.

### Notes

- KVA uses a flat t=0 EAD profile as a documented approximation. Full
  path-dependent EAD (SA-CCR at each Monte Carlo step) is planned for v1.2.
- Asymmetric FVA (different borrow vs lend rates) is deferred to v1.2.
- SIMM-based `im_time_profile` (sensitivity scaling approximation) is deferred to v1.2.

---

## [1.0.0] — 2026-03-18

### Added

- **HullWhite2F** two-factor Hull-White (G2++) short rate model
  (`src/pyxva/models/rates/hull_white2f.py`). Compatible with
  existing rate pricers via the combined short-rate factor "r" = r(t)+u(t).

- **AsianOption** arithmetic average `StatefulPricer`
  (`src/pyxva/pricing/exotic/asian_option.py`). Payoff at expiry:
  `max(avg(S) - K, 0)` where avg is the arithmetic mean over all simulation
  steps up to expiry.

- **SA-CCR calculator** (`src/pyxva/exposure/saccr.py`). Formula-
  based Basel III Standardised Approach for measuring counterparty credit risk
  EAD. Supports IR, equity, and FX asset classes.

- **TradeFactory auto-registration**: `BarrierOption` and `AsianOption` are
  now automatically registered on import — no manual `@TradeFactory.register`
  call required.

- **BarrierOption pre-expiry analytical MTM**: `step()` now returns the
  Black-Scholes barrier option price (down-and-out / up-and-out, call / put)
  for `t < expiry`, making EE/PFE profiles smooth and informative pre-expiry.

- **SharedMemory in parallel execution**: `_run_parallel()` in the pipeline
  engine now uses `SimulationSharedMemory` to share simulation paths across
  worker processes via OS shared memory, avoiding O(workers × data) memory
  duplication from pickling.

- **Example YAML configs** in `examples/`:
  - `single_swap.yaml` — single IRS with HullWhite1F
  - `multi_asset.yaml` — IRS + European equity option with correlation
  - `stress_test.yaml` — three agreements triggering the parallel execution path

- **Streaming vs batch EE parity test**: `tests/test_streaming_exposure.py`
  now includes `test_streaming_batch_ee_parity` which asserts that
  `StreamingExposureEngine` and `ISDAExposureCalculator` produce matching EE
  profiles within 1bp for a plain vanilla swap with zero threshold/MTA.

### Changed

- `pyproject.toml` version bumped to `1.0.0`.
- `pyproject.toml` description updated from placeholder to meaningful text.

### Exports

- `HullWhite2F` added to `pyxva.models`.
- `AsianOption` added to `pyxva.pricing`.
- `SACCRCalculator` added to `pyxva.exposure`.

---

## [0.1.0] — initial release

Initial implementation including:
- `MonteCarloEngine` with antithetic and quasi-random variance reduction
- `HullWhite1F`, `GeometricBrownianMotion`, `HestonModel`, `Schwartz1F/2F`, `GarmanKohlhagen`
- `InterestRateSwap`, `ZeroCouponBond`, `FixedRateBond`, `EuropeanOption`
- `BarrierOption` (StatefulPricer) with knock-out monitoring
- `NettingSet`, `Agreement`, `ISDAExposureCalculator`
- `REGVMEngine`, `REGIMEngine` (Schedule + SIMM)
- `StreamingExposureEngine` for memory-efficient path-by-step exposure
- `SparseTimeGrid` with cashflow-date merging
- `SimulationSharedMemory` for zero-copy cross-process sharing
- Full `RiskEngine` pipeline with YAML config support
- CVA, DVA, bCVA, EE, ENE, PFE, EPE, EEPE metrics
