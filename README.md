# risk_analytics

A Python library for Monte Carlo counterparty credit risk analytics. Simulates correlated
stochastic paths for interest rates, equities, FX, and commodities; prices instruments on
those paths; and computes bilateral exposure metrics (EE, PFE, EPE, EEPE, CVA/DVA) with
full ISDA CSA support including VM, IM (Schedule and SIMM), and collateral.

---

## Installation

```bash
uv sync          # installs all dependencies
uv run pytest    # 165 tests
uv run python demo.py
```

**Requirements:** Python 3.12+, `numpy`, `scipy`, `pandas`

---

## Architecture

```
risk_analytics/
├── core/
│   ├── base.py          # StochasticModel + Pricer ABCs; save/load serialisation
│   ├── engine.py        # MonteCarloEngine (Cholesky correlation, antithetic, Sobol)
│   ├── grid.py          # TimeGrid utility
│   ├── paths.py         # SimulationResult dataclass
│   ├── conventions.py   # DayCountConvention, BusinessDayConvention, Calendar hierarchy
│   └── schedule.py      # Schedule (payment dates, day-count fractions)
├── models/
│   ├── rates/hull_white.py        # HullWhite1F — exact Gaussian discretisation
│   ├── equity/gbm.py              # GeometricBrownianMotion — exact log-normal
│   ├── equity/heston.py           # HestonModel — Euler + full truncation
│   ├── commodity/schwartz1f.py    # Schwartz1F — exact OU
│   ├── commodity/schwartz2f.py    # Schwartz2F — exact OU + BM cumsum
│   └── fx/garman_kohlhagen.py     # GarmanKohlhagen — exact log-normal
├── pricing/
│   ├── rates/swap.py              # InterestRateSwap (uniform or Schedule)
│   ├── rates/bond.py              # ZeroCouponBond, FixedRateBond
│   └── equity/vanilla_option.py   # EuropeanOption — vectorised Black-Scholes
└── exposure/
    ├── metrics.py                 # ExposureCalculator: EE, PFE, PSE, EPE
    ├── netting.py                 # NettingSet
    ├── bilateral.py               # BilateralExposureCalculator, ISDAExposureCalculator
    ├── margin/vm.py               # REGVMEngine — path-dependent MTA-gated CSB
    ├── margin/im.py               # REGIMEngine — Schedule IM
    ├── margin/simm.py             # SimmCalculator — ISDA SIMM IR/Equity delta
    ├── collateral.py              # CollateralAccount + HaircutSchedule
    └── csa.py                     # CSATerms (threshold, MTA, MPOR, IM model)
```

---

## Stochastic Models

All models implement `StochasticModel` with `simulate()`, `calibrate()`, `get_params()`,
`set_params()`, `save(path)`, and `load(path)`.

| Model | Asset class | SDE |
|---|---|---|
| `HullWhite1F` | Interest rates | `dr = (θ(t) − a·r) dt + σ dW` — exact Gaussian |
| `GeometricBrownianMotion` | Equity | `dS = μS dt + σS dW` — exact log-normal |
| `HestonModel` | Equity (stoch vol) | `dS`, `dv` joint — Euler + full truncation |
| `Schwartz1F` | Commodity | `dX = κ(μ − X) dt + σ dW` — exact OU |
| `Schwartz2F` | Commodity | log-spot + convenience yield — exact OU + BM |
| `GarmanKohlhagen` | FX | `dS = (r_d − r_f)S dt + σS dW` — exact log-normal |

### Serialisation

Every model's calibrated state can be round-tripped to/from JSON:

```python
hw = HullWhite1F(a=0.15, sigma=0.01, r0=0.04)
hw.calibrate(market_data)
hw.save("hw.json")

hw2 = HullWhite1F().load("hw.json")   # chaining pattern
```

The JSON file contains `{"model": "HullWhite1F", "params": {...}}`. NumPy arrays
(e.g. `theta`) are stored as JSON lists. Loading validates the model name before
calling `set_params()`.

---

## Monte Carlo Engine

`MonteCarloEngine` drives joint simulation of multiple models under a shared correlation
structure. It Cholesky-decomposes the global factor correlation matrix once, then slices
correlated draws to each model.

```python
engine = MonteCarloEngine(
    n_paths=10_000,
    seed=42,
    quasi_random=False,   # True → Sobol sequences via scipy.stats.qmc
    antithetic=False,     # True → antithetic variates (n_paths must be even)
)
results = engine.run(
    models=[hw, gbm, sch],
    time_grid=grid,
    correlation_matrix=corr,   # None → independence
)
# results: dict[str, SimulationResult]
```

`SimulationResult` exposes `.factor(name)` → `np.ndarray` of shape `(n_paths, T)`.

---

## Pricing

Pricers accept a `SimulationResult` and return an MTM array of shape `(n_paths, T)`.

```python
swap = InterestRateSwap(fixed_rate=0.045, maturity=5.0, notional=1_000_000, payer=True)
mtm  = swap.price(results["HullWhite1F"])   # (n_paths, T)

call = EuropeanOption(strike=105.0, expiry=2.0, sigma=0.22, risk_free_rate=0.04, option_type="call")
call_mtm = call.price(results["GBM"])
bs_price = EuropeanOption.black_scholes_price(S=100, K=105, T=2.0, r=0.04, sigma=0.22, option_type="call")
```

Both `InterestRateSwap` and `FixedRateBond` accept an optional `Schedule` for
realistic day-count and business-day-adjusted coupon dates.

---

## Day-Count Conventions and Schedules

```python
from risk_analytics.core import (
    DayCountConvention, BusinessDayConvention,
    NullCalendar, TARGET, USCalendar,
    Frequency, Schedule,
)
from datetime import date

yf = DayCountConvention.ACT_ACT_ISDA.year_fraction(date(2024, 7, 1), date(2025, 7, 1))

sched = Schedule.from_dates(
    date(2024, 1, 1), date(2029, 1, 1),
    Frequency.SEMI_ANNUAL,
    calendar=TARGET(),
    day_count=DayCountConvention.ACT_360,
    bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
)
# sched.payment_dates, sched.payment_times, sched.day_count_fractions
```

Supported conventions: `ACT_360`, `ACT_365`, `ACT_ACT_ISDA`, `THIRTY_360`, `THIRTY_E_360`.
Calendars: `NullCalendar` (weekends only), `TARGET` (ECB), `USCalendar` (Federal).

---

## Exposure Metrics

### Basic (uncollateralised)

```python
from risk_analytics import ExposureCalculator, NettingSet

calc = ExposureCalculator()
summary = calc.exposure_summary(mtm, time_grid, confidence=0.95)
# keys: ee_profile, pfe_profile, pse, epe

ns = NettingSet("Counterparty_A")
ns.add_trade("payer_5y", payer_swap)
ns.add_trade("receiver_3y", recv_swap)
net_mtm = ns.net_mtm(results)   # sum of MTMs before max(·,0)
```

### Bilateral (ISDA/regulatory)

```python
from risk_analytics import CSATerms, ISDAExposureCalculator

csa = CSATerms.regvm_standard("Counterparty_A", mta=10_000)
isda = ISDAExposureCalculator(ns, csa)
out = isda.run(
    results, time_grid,
    confidence=0.95,
    cp_hazard_rate=0.008,    # counterparty λ (for CVA)
    own_hazard_rate=0.004,   # own λ (for DVA)
)
# out keys: ee, ene, pfe, ee_coll, ee_mpor, pse, epe, eepe,
#           cva, dva, bcva, net_mtm, csb, lagged_csb, im, collateral
```

`REGVMEngine` computes the Credit Support Balance path-dependently: at each margin call
date, a transfer is made only if `|target − CSB_prev| ≥ MTA`. The MPOR-lagged version
is used for collateralised exposure (gap-risk approximation).

---

## Regulatory Initial Margin

```python
from risk_analytics import REGIMEngine, CSATerms, IMModel, SimmSensitivities
from risk_analytics.exposure import SimmCalculator

# Schedule IM
csa = CSATerms(im_model=IMModel.SCHEDULE)
im_engine = REGIMEngine(csa)
schedule_im = im_engine.schedule_im(trades=[
    {"asset_class": "IR", "gross_notional": 1_000_000, "maturity": 5.0, "net_replacement_cost": 8_000},
])

# SIMM
sens = SimmSensitivities(ir={"USD": {"1y": 200.0, "5y": 800.0}}, equity={})
simm_im = SimmCalculator().total_im(sens)
```

---

## Logging

The library uses Python's standard `logging` module throughout. To see INFO-level
pipeline events (calibration results, simulation milestones, exposure summaries):

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s — %(levelname)s — %(message)s")
```

Use `level=logging.DEBUG` for per-step internals (Cholesky application, file I/O,
per-model draw slicing). No output is produced by default (library-friendly behaviour).

---

## End-to-End Demo

```python
import numpy as np
from risk_analytics import (
    MonteCarloEngine, TimeGrid,
    HullWhite1F, GeometricBrownianMotion, Schwartz1F,
    InterestRateSwap, EuropeanOption,
    ExposureCalculator, NettingSet,
    CSATerms, ISDAExposureCalculator,
)

# --- 1. Models ---
hw  = HullWhite1F(a=0.15, sigma=0.010, r0=0.04)
gbm = GeometricBrownianMotion(S0=100.0, mu=0.06, sigma=0.22)
sch = Schwartz1F(S0=80.0, kappa=1.2, mu=np.log(80), sigma=0.35)

tenors     = np.array([0.5, 1, 2, 3, 5, 7, 10])
zero_rates = np.array([0.038, 0.040, 0.042, 0.044, 0.047, 0.050, 0.053])
grid = TimeGrid.uniform(5.0, 60)   # 5-year horizon, monthly steps

hw.calibrate({"tenors": tenors, "zero_rates": zero_rates, "time_grid": grid})
gbm.calibrate({"S0": 100.0, "atm_vol": 0.22, "mu": 0.06})

# --- 2. Correlated simulation ---
corr = np.array([
    [1.00, 0.10, 0.15],
    [0.10, 1.00, 0.25],
    [0.15, 0.25, 1.00],
])
engine  = MonteCarloEngine(n_paths=10_000, seed=42)
results = engine.run(models=[hw, gbm, sch], time_grid=grid, correlation_matrix=corr)

# --- 3. Pricing ---
payer_swap = InterestRateSwap(fixed_rate=0.045, maturity=5.0, notional=1_000_000, payer=True)
recv_swap  = InterestRateSwap(fixed_rate=0.035, maturity=3.0, notional=500_000, payer=False)
call       = EuropeanOption(strike=105.0, expiry=2.0, sigma=0.22, risk_free_rate=0.04, option_type="call")

payer_mtm = payer_swap.price(results["HullWhite1F"])
recv_mtm  = recv_swap.price(results["HullWhite1F"])
call_mtm  = call.price(results["GBM"])

# --- 4. Basic exposure ---
calc    = ExposureCalculator()
summary = calc.exposure_summary(payer_mtm, grid, confidence=0.95)
print(f"PSE: {summary['pse']:,.0f}   EPE: {summary['epe']:,.0f}")

# --- 5. Netting set ---
ns = NettingSet("Counterparty_A")
ns.add_trade("payer_5y",   payer_swap)
ns.add_trade("receiver_3y", recv_swap)

# --- 6. ISDA bilateral exposure with VM ---
csa  = CSATerms.regvm_standard("Counterparty_A", mta=10_000)
isda = ISDAExposureCalculator(ns, csa)
out  = isda.run(results, grid, confidence=0.95, cp_hazard_rate=0.008, own_hazard_rate=0.004)

print(f"EE (uncoll):    {out['ee'].mean():>12,.0f}")
print(f"EE (coll+MPOR): {out['ee_coll'].mean():>12,.0f}")
print(f"EEPE (reg cap): {out['eepe']:>12,.0f}")
print(f"CVA:            {out['cva']:>12,.0f}")
print(f"DVA:            {out['dva']:>12,.0f}")
print(f"BCVA:           {out['bcva']:>12,.0f}")
```

Run the full demo (with summary tables and legacy CSA / IM comparisons):

```bash
uv run python demo.py
```
