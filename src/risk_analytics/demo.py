"""
risk_analytics — End-to-End Demo
=================================
Demonstrates the complete pipeline:
  1.  Market data + sparse time grid
  2.  Model setup and calibration
  3.  Monte Carlo simulation (correlated multi-asset)
  4.  Pricing on simulated paths
  5.  Basic exposure metrics (PSE / EPE / PFE)
  6.  Netting set
  7.  ISDA bilateral exposure with CSA, VM, IM, collateral
  8.  CVA / DVA / BCVA
  9.  Backtesting (PFE exceedance, Kupiec test, EE bias t-test)
  10. Pipeline engine — YAML-style dict config (end-to-end)
  11. Stress testing via RunResult.stress_test()

Run with:  uv run risk-analytics-demo
       or: uv run python -m risk_analytics.demo
"""

import numpy as np

from risk_analytics import (
    # Backtest
    BacktestEngine,
    # Core
    MonteCarloEngine,
    TimeGrid,
    SparseTimeGrid,
    # Market data
    MarketData,
    BumpType,
    ScenarioBump,
    # Models
    HullWhite1F,
    GeometricBrownianMotion,
    Schwartz1F,
    # Pricing
    InterestRateSwap,
    EuropeanOption,
    # Portfolio
    Trade,
    Agreement,
    # Exposure — basic
    ExposureCalculator,
    NettingSet,
    # Exposure — ISDA/bilateral
    CSATerms,
    IMModel,
    REGIMEngine,
    SimmSensitivities,
    BilateralExposureCalculator,
    ISDAExposureCalculator,
    # Pipeline
    RiskEngine,
    EngineConfig,
)

np.set_printoptions(precision=4, suppress=True)
SEP = "─" * 64


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main() -> None:
    # ===========================================================================
    # 1. MARKET DATA + SPARSE TIME GRID
    # ===========================================================================
    section("1. MarketData + SparseTimeGrid")

    md = MarketData.from_dict({
        "curves": {
            "USD_OIS": {
                "tenors": [0.5, 1, 2, 3, 5, 7, 10],
                "rates":  [0.038, 0.040, 0.042, 0.044, 0.047, 0.050, 0.053],
                "interpolation": "LOG_LINEAR",
            },
        },
        "spots": {"CRUDE_WTI": 80.0},
        "vols":  {"SPX": 0.22},
    })

    print(f"  USD 5y zero rate:  {md.zero_rate('USD_OIS', 5.0):.4f}")
    print(f"  USD P(0,5):        {md.discount_factor('USD_OIS', 5.0):.6f}")
    print(f"  USD 1y→2y fwd:     {md.forward_rate('USD_OIS', 1.0, 2.0):.4f}")
    print(f"  WTI spot:          {md.spot('CRUDE_WTI'):.1f}")

    # Bump: +10bps parallel shift
    md_bumped = md.bump("USD_OIS", 0.001)
    print(f"  USD 5y (bumped +10bps):  {md_bumped.zero_rate('USD_OIS', 5.0):.4f}")

    # Scenario: slope steepener
    md_steep = md.scenario([
        ScenarioBump("USD_OIS", -0.001, BumpType.SLOPE),
    ])
    print(f"  USD 0.5y (steepener): {md_steep.zero_rate('USD_OIS', 0.5):.4f}  "
          f"(base: {md.zero_rate('USD_OIS', 0.5):.4f})")
    print(f"  USD 10y  (steepener): {md_steep.zero_rate('USD_OIS', 10.0):.4f}  "
          f"(base: {md.zero_rate('USD_OIS', 10.0):.4f})")

    # Sparse grid: daily → weekly → monthly
    sparse_grid = SparseTimeGrid.standard(5.0)
    print(f"\n  Sparse grid (5yr): {len(sparse_grid)} nodes  "
          f"(vs {61} nodes on uniform monthly)")
    print(f"  First 5 nodes: {sparse_grid[:5].round(4)}")
    print(f"  Last  3 nodes: {sparse_grid[-3:].round(4)}")

    # ===========================================================================
    # 2. SIMULATION SETUP
    # ===========================================================================
    section("2. Model Setup & Calibration")

    # Hull-White 1F: calibrate theta to the market curve
    hw = HullWhite1F(a=0.15, sigma=0.010, r0=0.04)
    tenors     = np.array([0.5, 1, 2, 3, 5, 7, 10])
    zero_rates = np.array([0.038, 0.040, 0.042, 0.044, 0.047, 0.050, 0.053])
    grid = TimeGrid.uniform(5.0, 60)   # uniform grid used in library-API sections
    hw.calibrate({"tenors": tenors, "zero_rates": zero_rates, "time_grid": grid})
    print(f"  Hull-White 1F params: {hw.get_params()}")

    gbm = GeometricBrownianMotion(S0=100.0, mu=0.06, sigma=0.22)
    gbm.calibrate({"S0": 100.0, "atm_vol": 0.22, "mu": 0.06})
    print(f"  GBM params:           {gbm.get_params()}")

    sch = Schwartz1F(S0=80.0, kappa=1.2, mu=np.log(80), sigma=0.35)
    sch.calibrate({
        "S0": 80.0,
        "hist_vol": 0.35,
        "forward_prices": np.array([81, 83, 86, 88]),
        "forward_tenors": np.array([0.5, 1.0, 2.0, 3.0]),
    })
    print(f"  Schwartz 1F params:   {sch.get_params()}")

    # interpolation_space: log for GBM/GK (spots), linear for rates
    print(f"\n  HullWhite1F interpolation_space:  {hw.interpolation_space}")
    print(f"  GBM         interpolation_space:  {gbm.interpolation_space}")
    print(f"  Schwartz1F  interpolation_space:  {sch.interpolation_space}")

    # ===========================================================================
    # 3. CORRELATED MONTE CARLO SIMULATION
    # ===========================================================================
    section("3. Correlated Multi-Asset Monte Carlo")

    corr = np.array([
        [1.00, 0.10, 0.15],
        [0.10, 1.00, 0.25],
        [0.15, 0.25, 1.00],
    ])

    engine  = MonteCarloEngine(n_paths=10_000, seed=42)
    results = engine.run(models=[hw, gbm, sch], time_grid=grid, correlation_matrix=corr)

    for name, res in results.items():
        print(f"  {name:15s}  paths={res.n_paths}  steps={res.n_steps}  "
              f"factors={res.factor_names}  interp={res.interpolation_space}")

    # On-demand sparse interpolation
    hw_result = results["HullWhite1F"]
    r_at_1y   = hw_result.at(1.0)          # (n_paths, n_factors) at t=1.0
    r_at_times = hw_result.at_times(np.array([0.5, 1.0, 2.5]))  # (n_paths, 3, n_factors)
    print(f"\n  hw.at(1.0) shape:            {r_at_1y.shape}")
    print(f"  hw.at_times([.5,1,2.5]) shape: {r_at_times.shape}")
    print(f"  Mean short rate at t=1y:     {r_at_1y[:, 0].mean():.4f}")

    # ===========================================================================
    # 4. PRICING ON SIMULATED PATHS
    # ===========================================================================
    section("4. Instrument Pricing on Simulated Paths")

    payer_swap = InterestRateSwap(fixed_rate=0.045, maturity=5.0, notional=1_000_000, payer=True)
    recv_swap  = InterestRateSwap(fixed_rate=0.035, maturity=3.0, notional=500_000, payer=False)
    call       = EuropeanOption(strike=105.0, expiry=2.0, sigma=0.22, risk_free_rate=0.04, option_type="call")

    payer_mtm = payer_swap.price(results["HullWhite1F"])
    recv_mtm  = recv_swap.price(results["HullWhite1F"])
    call_mtm  = call.price(results["GBM"])

    print(f"  Payer IRS    t=0 mean MTM: {payer_mtm[:,0].mean():>12,.0f}")
    print(f"  Receiver IRS t=0 mean MTM: {recv_mtm[:,0].mean():>11,.0f}")
    bs_ref = EuropeanOption.black_scholes_price(100, 105, 2.0, 0.04, 0.22, "call")
    print(f"  Call option  t=0 MC price: {call_mtm[:,0].mean():>12.4f}  (BS={bs_ref:.4f})")

    # cashflow_times() — used by pipeline to augment the sparse grid
    print(f"\n  payer_swap.cashflow_times()[:4]: {payer_swap.cashflow_times()[:4]}")
    print(f"  call.cashflow_times():           {call.cashflow_times()}")

    # ===========================================================================
    # 5. BASIC EXPOSURE METRICS (UNCOLLATERALISED)
    # ===========================================================================
    section("5. Basic Uncollateralised Exposure Metrics")

    calc    = ExposureCalculator()
    summary = calc.exposure_summary(payer_mtm, grid, confidence=0.95)

    print(f"  PSE (Peak Simulated Exposure):   {summary['pse']:>12,.0f}")
    print(f"  EPE (Expected Positive Exposure):{summary['epe']:>12,.0f}")
    print(f"  PFE 95th pct at 2yr:             "
          f"{summary['pfe_profile'][np.searchsorted(grid, 2.0)]:>12,.0f}")
    print(f"  EE  profile shape:               {summary['ee_profile'].shape}")

    # ===========================================================================
    # 6. NETTING SET
    # ===========================================================================
    section("6. Netting Set (IRS Payer + IRS Receiver)")

    ns = NettingSet("NS_001")
    ns.add_trade("payer_5y",    payer_swap)
    ns.add_trade("receiver_3y", recv_swap)

    net_mtm    = ns.net_mtm(results)
    ns_summary = ns.exposure(results, grid, confidence=0.95)

    gross_ee       = calc.expected_exposure(payer_mtm) + calc.expected_exposure(recv_mtm)
    net_ee         = ns_summary["ee_profile"]
    netting_benefit = 1.0 - net_ee.mean() / gross_ee.mean()

    print(f"  Gross EE (sum of trades):  {gross_ee.mean():>10,.0f}")
    print(f"  Net EE (netting set):      {net_ee.mean():>10,.0f}")
    print(f"  Netting benefit:           {netting_benefit:>10.1%}")
    print(f"  Net PSE:                   {ns_summary['pse']:>10,.0f}")
    print(f"  Net EPE:                   {ns_summary['epe']:>10,.0f}")

    # ===========================================================================
    # 7. ISDA BILATERAL EXPOSURE WITH CSA
    # ===========================================================================
    section("7. ISDA Bilateral Exposure — REGVM Zero-Threshold CSA")

    csa_regvm  = CSATerms.regvm_standard("Counterparty_A", mta=10_000)
    isda_calc  = ISDAExposureCalculator(ns, csa_regvm)

    out_regvm = isda_calc.run(
        results, grid,
        confidence=0.95,
        cp_hazard_rate=0.008,
        own_hazard_rate=0.004,
    )

    print(f"  EE  (uncoll):    {out_regvm['ee'].mean():>12,.0f}")
    print(f"  EE  (coll+MPOR): {out_regvm['ee_coll'].mean():>12,.0f}")
    print(f"  ENE:             {out_regvm['ene'].mean():>12,.0f}")
    print(f"  EEPE (reg cap):  {out_regvm['eepe']:>12,.0f}")
    print(f"  PSE:             {out_regvm['pse']:>12,.0f}")
    print(f"  EPE:             {out_regvm['epe']:>12,.0f}")
    print(f"  CVA:             {out_regvm['cva']:>12,.0f}")
    print(f"  DVA:             {out_regvm['dva']:>12,.0f}")
    print(f"  BCVA (CVA-DVA):  {out_regvm['bcva']:>12,.0f}")

    section("7b. Legacy CSA — Symmetric Threshold (50K per side)")

    csa_legacy = CSATerms.legacy_bilateral("Counterparty_A", threshold=50_000, mta=10_000)
    out_legacy = ISDAExposureCalculator(ns, csa_legacy).run(results, grid, confidence=0.95)

    print(f"  EE  (coll+MPOR, REGVM):  {out_regvm['ee_coll'].mean():>12,.0f}")
    print(f"  EE  (coll+MPOR, Legacy): {out_legacy['ee_coll'].mean():>12,.0f}")
    print(f"  → Legacy threshold adds EE: "
          f"{out_legacy['ee_coll'].mean() - out_regvm['ee_coll'].mean():>10,.0f}")

    # ===========================================================================
    # 8. REGULATORY IM — SCHEDULE AND SIMM
    # ===========================================================================
    section("8. Initial Margin — Schedule vs SIMM")

    csa_im    = CSATerms(im_model=IMModel.SCHEDULE)
    im_engine = REGIMEngine(csa_im)
    trades_descriptor = [
        {
            "asset_class": "IR",
            "gross_notional": 1_000_000,
            "maturity": 5.0,
            "net_replacement_cost": float(payer_mtm[:, 0].mean()),
        },
        {
            "asset_class": "IR",
            "gross_notional": 500_000,
            "maturity": 3.0,
            "net_replacement_cost": float(recv_mtm[:, 0].mean()),
        },
    ]
    schedule_im = im_engine.schedule_im(trades_descriptor)
    print(f"  Schedule IM (netting set): {float(schedule_im):>12,.0f}")

    from risk_analytics.exposure import SimmCalculator
    simm_calc = SimmCalculator()
    sens = SimmSensitivities(
        ir={"USD": {"1y": 200.0, "2y": 350.0, "5y": 800.0}},
        equity={},
    )
    simm_im = float(simm_calc.total_im(sens))
    print(f"  SIMM IM (IR delta only):   {simm_im:>12,.2f}")

    csa_with_im       = CSATerms.regvm_standard("CP", mta=10_000)
    csa_with_im.im_model = IMModel.SCHEDULE
    im_engine2        = REGIMEngine(csa_with_im)
    out_with_im       = ISDAExposureCalculator(ns, csa_with_im, im_engine=im_engine2).run(
        results, grid, im_trades=trades_descriptor, confidence=0.95
    )
    print(f"\n  EE_coll (VM only):          {out_regvm['ee_coll'].mean():>12,.0f}")
    print(f"  EE_coll (VM + Schedule IM): {out_with_im['ee_coll'].mean():>12,.0f}")
    print(f"  IM benefit to EE:           "
          f"{out_regvm['ee_coll'].mean() - out_with_im['ee_coll'].mean():>12,.0f}")

    # ===========================================================================
    # 9. BACKTESTING — PFE EXCEEDANCE AND EE ACCURACY
    # ===========================================================================
    section("9. Backtesting (Synthetic Realised Path)")

    bt_engine   = MonteCarloEngine(n_paths=2001, seed=99)
    bt_results  = bt_engine.run([hw], time_grid=grid)
    all_r_paths = bt_results["HullWhite1F"].factor("r")

    from risk_analytics import InterestRateSwap as IRS
    from risk_analytics.core.paths import SimulationResult
    bt_payer = IRS(fixed_rate=0.045, maturity=5.0, notional=1_000_000, payer=True)
    all_mtm  = bt_payer.price(
        SimulationResult(
            time_grid=grid,
            paths=all_r_paths[:, :, np.newaxis],
            model_name="HullWhite1F",
            factor_names=["r"],
        )
    )

    realized_mtm = all_mtm[0]
    forecast_mtm = all_mtm[1:]

    bt        = BacktestEngine(confidence=0.95)
    bt_result = bt.run(forecast_mtm, realized_mtm, grid)
    s         = bt_result.summary()

    print(f"  Exceptions:     {s['n_exceptions']}/{s['n_observations']} "
          f"({s['exception_rate']:.1%} vs {s['expected_exception_rate']:.1%} expected)")
    print(f"  Basel zone:     {s['basel_zone']}")
    print(f"  Kupiec p-value: {s['kupiec_pvalue']:.3f}   "
          f"({'fail to reject' if s['kupiec_pvalue'] > 0.05 else 'REJECT'} H₀ at 5%)")
    print(f"  EE bias:        {s['ee_bias']:>12,.0f}   "
          f"({'over-predicts' if s['ee_bias'] > 0 else 'under-predicts'})")
    print(f"  Bias p-value:   {s['bias_pvalue']:.3f}   "
          f"({'fail to reject' if s['bias_pvalue'] > 0.05 else 'REJECT'} H₀ at 5%)")
    print(f"  EE RMSE:        {s['ee_rmse']:>12,.0f}")

    # ===========================================================================
    # 10. PIPELINE ENGINE — YAML-STYLE CONFIG
    # ===========================================================================
    section("10. Pipeline Engine (dict / YAML config)")

    pipeline_config = {
        "simulation": {
            "n_paths": 2_000,       # small for demo speed
            "seed": 42,
            "antithetic": False,
            "time_grid": {"type": "standard"},   # daily→weekly→monthly
        },
        "market_data": {
            "curves": {
                "USD_OIS": {
                    "tenors": [0.5, 1, 2, 3, 5, 7, 10],
                    "rates":  [0.038, 0.040, 0.042, 0.044, 0.047, 0.050, 0.053],
                },
            },
            "spots": {"SPX": 100.0},
            "vols":  {"SPX": 0.22},
        },
        "models": [
            {
                "name": "rates_usd",
                "type": "HullWhite1F",
                "params": {"a": 0.15, "sigma": 0.01, "r0": 0.04},
                "calibrate_to": "USD_OIS",
            },
            {
                "name": "equity_spx",
                "type": "GBM",
                "params": {"S0": 100.0, "mu": 0.06, "sigma": 0.22},
            },
        ],
        "correlation": [
            ["rates_usd", "equity_spx", 0.10],
        ],
        "agreements": [
            {
                "id": "AGR_GOLDMAN",
                "counterparty": "Goldman_Sachs",
                "cp_hazard_rate": 0.010,
                "own_hazard_rate": 0.005,
                "csa": {"mta": 10_000, "threshold": 0, "margin_regime": "REGVM"},
                "netting_sets": [
                    {
                        "id": "NS_IR",
                        "trades": [
                            {
                                "id": "trade_payer_5y",
                                "type": "InterestRateSwap",
                                "model": "rates_usd",
                                "params": {
                                    "fixed_rate": 0.045,
                                    "maturity": 5.0,
                                    "notional": 1_000_000,
                                    "payer": True,
                                },
                            },
                            {
                                "id": "trade_recv_3y",
                                "type": "InterestRateSwap",
                                "model": "rates_usd",
                                "params": {
                                    "fixed_rate": 0.035,
                                    "maturity": 3.0,
                                    "notional": 500_000,
                                    "payer": False,
                                },
                            },
                        ],
                    },
                    {
                        "id": "NS_EQ",
                        "trades": [
                            {
                                "id": "trade_call_2y",
                                "type": "EuropeanOption",
                                "model": "equity_spx",
                                "params": {
                                    "strike": 105.0,
                                    "expiry": 2.0,
                                    "sigma": 0.22,
                                    "risk_free_rate": 0.04,
                                    "option_type": "call",
                                },
                            },
                        ],
                    },
                ],
            },
        ],
        "outputs": {
            "metrics": ["EE", "PFE", "CVA"],
            "confidence": 0.95,
            "write_raw_paths": False,
        },
    }

    risk_engine = RiskEngine(pipeline_config)
    run_result  = risk_engine.run()

    agr = run_result.agreement_results["AGR_GOLDMAN"]
    print(f"  Agreement:        {agr.id}  ({agr.counterparty_id})")
    print(f"  Time grid nodes:  {len(run_result.time_grid)}  "
          f"(sparse, includes cashflow dates)")
    print(f"  EE profile shape: {agr.ee_profile.shape}")
    print(f"  CVA:              {agr.cva:>12,.0f}")
    print(f"  DVA:              {agr.dva:>12,.0f}")
    print(f"  BCVA:             {agr.bcva:>12,.0f}")
    print(f"  PSE:              {agr.pse:>12,.0f}")
    print(f"  EPE:              {agr.epe:>12,.0f}")
    print(f"  EEPE:             {agr.eepe:>12,.0f}")

    print(f"\n  Per netting-set pre-collateral EPE:")
    for ns_id, ns_sum in agr.netting_set_summaries.items():
        print(f"    {ns_id:10s}  EPE={ns_sum.epe:>10,.0f}  PSE={ns_sum.pse:>10,.0f}")

    print(f"\n  Portfolio totals:")
    print(f"    Total CVA: {run_result.total_cva:>12,.0f}")
    print(f"    Total DVA: {run_result.total_dva:>12,.0f}")

    df = run_result.summary_df()
    print(f"\n  summary_df():\n{df.to_string()}")

    # ===========================================================================
    # 11. STRESS TESTING
    # ===========================================================================
    section("11. Stress Testing (reprice on existing paths)")

    base_md = MarketData.from_dict(pipeline_config["market_data"])

    # +25bps parallel shift on USD rates
    stressed_result = run_result.stress_test(
        bumps=[ScenarioBump("USD_OIS", 0.0025, BumpType.PARALLEL)],
        market_data=base_md,
    )

    agr_stressed = stressed_result.agreement_results["AGR_GOLDMAN"]
    cva_delta    = agr_stressed.cva - agr.cva

    print(f"  Scenario: USD_OIS +25bps (parallel, reprice only — no re-simulation)")
    print(f"  Base CVA:     {agr.cva:>12,.0f}")
    print(f"  Stressed CVA: {agr_stressed.cva:>12,.0f}")
    print(f"  CVA delta:    {cva_delta:>+12,.0f}  "
          f"({'increase' if cva_delta > 0 else 'decrease'})")
    print(f"  Base EPE:     {agr.epe:>12,.0f}")
    print(f"  Stressed EPE: {agr_stressed.epe:>12,.0f}")

    # Summary table: exposure profile at selected tenors
    section("Summary: Exposure Profile at Selected Tenors (pipeline run)")

    t_grid    = run_result.time_grid
    t_targets = [0.5, 1.0, 2.0, 3.0, 5.0]
    indices   = [np.searchsorted(t_grid, t) for t in t_targets]

    print(f"\n  {'Tenor':>6}  {'EE (base)':>12}  {'EE (stressed)':>14}  {'PFE 95%':>10}")
    print("  " + "─" * 52)
    for t, i in zip(t_targets, indices):
        ee_b = agr.ee_profile[i]
        ee_s = agr_stressed.ee_profile[i]
        pfe  = agr.pfe_profile[i]
        print(f"  {t:>6.1f}  {ee_b:>12,.0f}  {ee_s:>14,.0f}  {pfe:>10,.0f}")

    print(f"\n{SEP}")
    print("  Demo complete.")
    print(SEP)


if __name__ == "__main__":
    main()
