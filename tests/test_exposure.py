"""Tests for exposure metrics and netting."""
import numpy as np
import pytest

from risk_analytics.core import MonteCarloEngine, TimeGrid
from risk_analytics.models import GeometricBrownianMotion, HullWhite1F
from risk_analytics.pricing import EuropeanOption, InterestRateSwap
from risk_analytics.exposure import ExposureCalculator, NettingSet
from risk_analytics.portfolio.trade import Trade

N_PATHS = 2000
SEED = 42


def make_equity_results(n_paths=N_PATHS):
    model = GeometricBrownianMotion(S0=100.0, mu=0.05, sigma=0.20)
    grid = TimeGrid.uniform(1.0, 52)
    engine = MonteCarloEngine(n_paths, seed=SEED)
    results = engine.run([model], grid)
    return results, grid


def make_rates_results(n_paths=N_PATHS):
    model = HullWhite1F(a=0.1, sigma=0.01, r0=0.05)
    grid = TimeGrid.uniform(5.0, 60)
    engine = MonteCarloEngine(n_paths, seed=SEED)
    results = engine.run([model], grid)
    return results, grid


class TestExposureCalculator:
    def setup_method(self):
        results, self.grid = make_equity_results()
        option = EuropeanOption(strike=100, expiry=1.0, sigma=0.20, risk_free_rate=0.03)
        self.mtm = option.price(results["GBM"])
        self.calc = ExposureCalculator()

    def test_exposure_profile_non_negative(self):
        ep = self.calc.exposure_profile(self.mtm)
        assert np.all(ep >= 0)

    def test_expected_exposure_shape(self):
        ee = self.calc.expected_exposure(self.mtm)
        assert ee.shape == (len(self.grid),)

    def test_pse_scalar(self):
        pse = self.calc.pse(self.mtm)
        assert isinstance(pse, float)
        assert pse >= 0

    def test_epe_scalar(self):
        epe = self.calc.epe(self.mtm, self.grid)
        assert isinstance(epe, float)
        assert epe >= 0

    def test_pfe_shape_and_monotone(self):
        pfe_90 = self.calc.pfe(self.mtm, 0.90)
        pfe_95 = self.calc.pfe(self.mtm, 0.95)
        assert pfe_90.shape == (len(self.grid),)
        # 95th percentile should be >= 90th
        assert np.all(pfe_95 >= pfe_90 - 1e-8)

    def test_pse_geq_epe(self):
        """PSE (peak) should be >= EPE (average)."""
        assert self.calc.pse(self.mtm) >= self.calc.epe(self.mtm, self.grid)

    def test_summary_keys(self):
        summary = self.calc.exposure_summary(self.mtm, self.grid)
        for key in ("ee_profile", "pfe_profile", "pse", "epe", "confidence"):
            assert key in summary

    def test_negative_mtm_gives_zero_exposure(self):
        """All-negative MTM should produce zero exposure."""
        mtm_neg = -np.abs(self.mtm)
        ep = self.calc.exposure_profile(mtm_neg)
        assert np.all(ep == 0.0)


class TestNettingSet:
    def setup_method(self):
        self.results, self.grid = make_rates_results()

    def test_single_trade_netting(self):
        """Single trade: netted MTM should equal trade MTM."""
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6, payer=True)
        ns = NettingSet("test")
        ns.add_trade(Trade(id="swap1", pricer=swap, model_name="HullWhite1F"))
        net = ns.net_mtm(self.results)
        direct = swap.price(self.results["HullWhite1F"])
        assert np.allclose(net, direct)

    def test_netting_benefit(self):
        """Sum of individual positive exposures >= netted positive exposure."""
        payer = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6, payer=True)
        receiver = InterestRateSwap(fixed_rate=0.03, maturity=5.0, notional=1e6, payer=False)

        ns = NettingSet("test")
        ns.add_trade(Trade(id="payer", pricer=payer, model_name="HullWhite1F"))
        ns.add_trade(Trade(id="receiver", pricer=receiver, model_name="HullWhite1F"))

        calc = ExposureCalculator()
        r = self.results["HullWhite1F"]

        mtm_payer = payer.price(r)
        mtm_receiver = receiver.price(r)

        ee_sum = (
            calc.expected_exposure(mtm_payer) + calc.expected_exposure(mtm_receiver)
        )
        net_mtm = ns.net_mtm(self.results)
        ee_net = calc.expected_exposure(net_mtm)

        # Netting benefit: gross >= net at every time step
        assert np.all(ee_sum >= ee_net - 1e-6)

    def test_trade_ids(self):
        ns = NettingSet("ns1")
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6)
        ns.add_trade(Trade(id="trade_A", pricer=swap, model_name="HullWhite1F"))
        assert ns.trade_ids == ["trade_A"]

    def test_empty_netting_set_raises(self):
        ns = NettingSet("empty")
        with pytest.raises(ValueError):
            ns.net_mtm(self.results)

    def test_exposure_summary_keys(self):
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6)
        ns = NettingSet("test")
        ns.add_trade(Trade(id="swap1", pricer=swap, model_name="HullWhite1F"))
        summary = ns.exposure(self.results, self.grid)
        for key in ("ee_profile", "pfe_profile", "pse", "epe", "net_mtm"):
            assert key in summary

    def test_add_trade_rejects_non_trade(self):
        """Passing anything other than a Trade object must raise TypeError."""
        ns = NettingSet("test")
        with pytest.raises(TypeError, match="Trade object"):
            ns.add_trade("swap1")  # type: ignore[arg-type]

    def test_wrong_model_name_raises(self):
        """A trade whose model_name is not in simulation_results raises KeyError."""
        swap = InterestRateSwap(fixed_rate=0.05, maturity=5.0, notional=1e6)
        ns = NettingSet("test")
        ns.add_trade(Trade(id="swap1", pricer=swap, model_name="NONEXISTENT_MODEL"))
        with pytest.raises(KeyError, match="NONEXISTENT_MODEL"):
            ns.net_mtm(self.results)

    def test_correct_model_used_when_names_overlap(self):
        """Each trade must price against its declared model, not another model with
        compatible factor names."""
        # Two GBM models with identical factor 'S' but very different spot prices.
        model_lo = GeometricBrownianMotion(S0=50.0, mu=0.0, sigma=0.01)
        model_hi = GeometricBrownianMotion(S0=200.0, mu=0.0, sigma=0.01)
        grid = TimeGrid.uniform(1.0, 10)
        engine = MonteCarloEngine(200, seed=0)
        results = engine.run([model_lo, model_hi], grid)
        # Both keyed by their model names: 'GBM' — but run() uses model.name as key,
        # which for both is 'GBM'. We rename one via a wrapper to give distinct keys.
        # Use the pipeline's _ModelWrapper pattern directly.
        from risk_analytics.pipeline.engine import _ModelWrapper
        model_lo_w = _ModelWrapper(model_lo, "eq_lo")
        model_hi_w = _ModelWrapper(model_hi, "eq_hi")
        results = engine.run([model_lo_w, model_hi_w], grid)

        option_lo = EuropeanOption(strike=50.0, expiry=1.0, sigma=0.01, risk_free_rate=0.0)
        option_hi = EuropeanOption(strike=200.0, expiry=1.0, sigma=0.01, risk_free_rate=0.0)

        ns = NettingSet("test")
        ns.add_trade(Trade(id="opt_lo", pricer=option_lo, model_name="eq_lo"))
        ns.add_trade(Trade(id="opt_hi", pricer=option_hi, model_name="eq_hi"))

        net = ns.net_mtm(results)

        # Price each trade directly on its declared model to verify NettingSet matches.
        expected = (
            option_lo.price(results["eq_lo"]) + option_hi.price(results["eq_hi"])
        )
        np.testing.assert_allclose(net, expected)
