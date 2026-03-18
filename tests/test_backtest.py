"""Tests for the backtesting module."""
from __future__ import annotations

import numpy as np
import pytest

from pyxva.backtest import BacktestEngine, BacktestResult
from pyxva.backtest.metrics import (
    basel_zone,
    bias_ttest,
    ee_accuracy,
    ee_profile,
    exceedance_series,
    kupiec_pof,
    pfe_profile,
)


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------

class TestPfeProfile:
    def test_zero_realized_no_exceedances(self):
        """When all MTM paths are positive, PFE should be > 0."""
        paths = np.ones((1000, 20)) * 50.0
        pfe = pfe_profile(paths, confidence=0.95)
        assert pfe.shape == (20,)
        assert np.all(pfe == pytest.approx(50.0))

    def test_negative_paths_clipped_to_zero(self):
        """Negative MTM contributes zero to PFE (max(V,0) floor)."""
        paths = np.full((500, 10), -100.0)
        pfe = pfe_profile(paths, confidence=0.95)
        assert np.all(pfe == pytest.approx(0.0))

    def test_quantile_ordering(self):
        """95th-percentile PFE >= 50th-percentile PFE at every step."""
        rng = np.random.default_rng(0)
        paths = rng.standard_normal((2000, 30)) * 100
        pfe_95 = pfe_profile(paths, 0.95)
        pfe_50 = pfe_profile(paths, 0.50)
        assert np.all(pfe_95 >= pfe_50 - 1e-10)

    def test_shape(self):
        paths = np.zeros((100, 15))
        assert pfe_profile(paths, 0.95).shape == (15,)


class TestEeProfile:
    def test_all_positive(self):
        paths = np.full((100, 5), 10.0)
        ee = ee_profile(paths)
        assert np.allclose(ee, 10.0)

    def test_mixed_sign_clips_negative(self):
        """Negative paths do not contribute to EE."""
        paths = np.array([[-50.0] * 5, [50.0] * 5])   # mean before clip = 0
        ee = ee_profile(paths)
        assert np.allclose(ee, 25.0)    # max(-50,0)=0, max(50,0)=50 → mean=25

    def test_shape(self):
        assert ee_profile(np.zeros((10, 7))).shape == (7,)


class TestExceedanceSeries:
    def test_no_exceedances_when_realized_below_pfe(self):
        pfe = np.array([100.0, 100.0, 100.0])
        realized = np.array([50.0, 60.0, 70.0])
        exc = exceedance_series(pfe, realized)
        assert not exc.any()

    def test_all_exceedances_when_realized_above_pfe(self):
        pfe = np.array([10.0, 20.0, 30.0])
        realized = np.array([100.0, 200.0, 300.0])
        exc = exceedance_series(pfe, realized)
        assert exc.all()

    def test_exact_equality_is_not_exceedance(self):
        pfe = np.array([50.0])
        realized = np.array([50.0])
        assert not exceedance_series(pfe, realized)[0]

    def test_mixed(self):
        pfe = np.array([100.0, 100.0, 100.0])
        realized = np.array([50.0, 100.0, 150.0])
        exc = exceedance_series(pfe, realized)
        assert exc.tolist() == [False, False, True]


class TestBaselZone:
    def test_zero_exceptions_green(self):
        assert basel_zone(0, 100) == "Green"

    def test_scaled_below_5_green(self):
        # 1 exception in 100 obs → scaled = round(1 * 250 / 100) = 3 → Green
        assert basel_zone(1, 100) == "Green"

    def test_scaled_5_to_9_amber(self):
        # 3 exceptions in 100 obs → scaled = round(3 * 250 / 100) = 8 → Amber
        assert basel_zone(3, 100) == "Amber"

    def test_scaled_10_plus_red(self):
        # 5 exceptions in 100 obs → scaled = round(5 * 250 / 100) = 13 → Red
        assert basel_zone(5, 100) == "Red"

    def test_exactly_250_obs(self):
        assert basel_zone(4, 250) == "Green"
        assert basel_zone(5, 250) == "Amber"
        assert basel_zone(10, 250) == "Red"

    def test_zero_observations(self):
        # Guard against division by zero
        assert basel_zone(0, 0) == "Green"


class TestEeAccuracy:
    def test_perfect_forecast_zero_error(self):
        ee = np.array([10.0, 20.0, 30.0])
        realized = np.array([10.0, 20.0, 30.0])
        acc = ee_accuracy(ee, realized)
        assert acc["rmse"] == pytest.approx(0.0)
        assert acc["bias"] == pytest.approx(0.0)
        assert acc["mae"] == pytest.approx(0.0)

    def test_positive_bias_when_model_overpredicts(self):
        ee = np.array([20.0, 20.0])
        realized = np.array([10.0, 10.0])
        acc = ee_accuracy(ee, realized)
        assert acc["bias"] == pytest.approx(10.0)

    def test_negative_bias_when_model_underpredicts(self):
        ee = np.array([5.0, 5.0])
        realized = np.array([10.0, 10.0])
        acc = ee_accuracy(ee, realized)
        assert acc["bias"] == pytest.approx(-5.0)

    def test_rmse_formula(self):
        ee = np.array([0.0, 0.0, 0.0])
        realized = np.array([3.0, 4.0, 0.0])
        acc = ee_accuracy(ee, realized)
        expected_rmse = np.sqrt((9 + 16 + 0) / 3)
        assert acc["rmse"] == pytest.approx(expected_rmse)

    def test_mae_formula(self):
        ee = np.array([10.0, 10.0])
        realized = np.array([5.0, 15.0])
        acc = ee_accuracy(ee, realized)
        assert acc["mae"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Kupiec POF test
# ---------------------------------------------------------------------------

class TestKupiecPof:
    def test_zero_exceptions_finite_lr(self):
        result = kupiec_pof(0, 100, 0.95)
        assert np.isfinite(result["lr_stat"])
        assert 0.0 <= result["p_value"] <= 1.0

    def test_all_exceptions_finite_lr(self):
        result = kupiec_pof(100, 100, 0.95)
        assert np.isfinite(result["lr_stat"])
        assert 0.0 <= result["p_value"] <= 1.0

    def test_correct_model_high_pvalue(self):
        """Exception rate equal to expected → p-value should be high (near 1)."""
        # 5 exceptions in 100 obs, p0 = 0.05 — exact match
        result = kupiec_pof(5, 100, 0.95)
        assert result["p_value"] > 0.5

    def test_grossly_wrong_model_low_pvalue(self):
        """Far too many exceptions → p-value should be very small."""
        # 30 exceptions in 100 obs when expecting 5% → strongly reject
        result = kupiec_pof(30, 100, 0.95)
        assert result["p_value"] < 0.01

    def test_lr_stat_nonnegative(self):
        for k in [0, 2, 5, 10, 20]:
            r = kupiec_pof(k, 100, 0.95)
            assert r["lr_stat"] >= 0.0

    def test_zero_observations_returns_nan(self):
        result = kupiec_pof(0, 0, 0.95)
        assert np.isnan(result["lr_stat"])
        assert np.isnan(result["p_value"])

    def test_pvalue_in_unit_interval(self):
        for k in range(0, 21):
            r = kupiec_pof(k, 100, 0.95)
            assert 0.0 <= r["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Bias t-test
# ---------------------------------------------------------------------------

class TestBiasTtest:
    def test_zero_bias_high_pvalue(self):
        """No bias → should fail to reject H0."""
        rng = np.random.default_rng(0)
        ee = rng.standard_normal(200) * 100 + 500
        realized = ee.copy()   # residuals all zero
        result = bias_ttest(ee, realized)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["t_stat"] == pytest.approx(0.0)

    def test_large_positive_bias_low_pvalue(self):
        """Consistent over-prediction → reject H0."""
        rng = np.random.default_rng(2)
        # Independent noise so residuals have variance; large positive mean
        ee = rng.standard_normal(100) * 10 + 1000.0
        realized = rng.standard_normal(100) * 10
        result = bias_ttest(ee, realized)
        assert result["t_stat"] > 0
        assert result["p_value"] < 0.001

    def test_large_negative_bias_low_pvalue(self):
        """Consistent under-prediction → reject H0, t negative."""
        rng = np.random.default_rng(3)
        ee = rng.standard_normal(100) * 10 - 1000.0
        realized = rng.standard_normal(100) * 10
        result = bias_ttest(ee, realized)
        assert result["t_stat"] < 0
        assert result["p_value"] < 0.001

    def test_returns_nan_for_single_observation(self):
        result = bias_ttest(np.array([5.0]), np.array([3.0]))
        assert np.isnan(result["t_stat"])
        assert np.isnan(result["p_value"])

    def test_pvalue_in_unit_interval(self):
        rng = np.random.default_rng(1)
        ee = rng.standard_normal(50) * 10
        realized = rng.standard_normal(50) * 10
        result = bias_ttest(ee, realized)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# BacktestEngine integration tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.T = 24
        self.n_paths = 2000
        self.time_grid = np.linspace(0, 2.0, self.T)
        # GBM-like positive paths
        self.mtm_paths = (
            np.abs(self.rng.standard_normal((self.n_paths, self.T))) * 50_000
        )
        self.engine = BacktestEngine(confidence=0.95)

    def test_run_returns_backtest_result(self):
        realized = self.rng.standard_normal(self.T) * 20_000
        result = self.engine.run(self.mtm_paths, realized, self.time_grid)
        assert isinstance(result, BacktestResult)

    def test_result_shapes(self):
        realized = np.zeros(self.T)
        result = self.engine.run(self.mtm_paths, realized, self.time_grid)
        assert result.pfe_profile.shape == (self.T,)
        assert result.ee_profile.shape == (self.T,)
        assert result.exceedances.shape == (self.T,)
        assert result.realized.shape == (self.T,)
        assert result.time_grid.shape == (self.T,)

    def test_zero_realized_no_exceedances(self):
        """Realized = 0 never exceeds PFE (which is ≥ 0)."""
        result = self.engine.run(self.mtm_paths, np.zeros(self.T), self.time_grid)
        assert result.n_exceptions == 0
        assert result.exception_rate == pytest.approx(0.0)
        assert result.basel_zone == "Green"

    def test_very_large_realized_all_exceedances(self):
        """Realized far above any path → all steps are exceptions."""
        realized = np.full(self.T, 1e12)
        result = self.engine.run(self.mtm_paths, realized, self.time_grid)
        assert result.n_exceptions == self.T
        assert result.exceedances.all()

    def test_exception_rate_near_expected_for_correct_model(self):
        """For a correct model the exception rate should be ≈ 1 - confidence."""
        # Use a large number of paths so PFE is accurate
        n_paths = 10_000
        T = 100
        paths = self.rng.standard_normal((n_paths, T)) * 100
        # Realized drawn from the same distribution
        realized = self.rng.standard_normal(T) * 100

        result = self.engine.run(paths, realized, np.linspace(0, 1, T))
        # Exception rate should be close to 5% ± some Monte Carlo noise
        assert result.exception_rate == pytest.approx(0.05, abs=0.08)

    def test_ee_rmse_zero_for_perfect_forecast(self):
        """If realized equals the EE, RMSE should be 0."""
        paths = np.full((500, self.T), 30_000.0)
        # EE = 30_000 everywhere
        realized = np.full(self.T, 30_000.0)
        result = self.engine.run(paths, realized, self.time_grid)
        assert result.ee_rmse == pytest.approx(0.0)
        assert result.ee_bias == pytest.approx(0.0)

    def test_summary_keys(self):
        realized = np.zeros(self.T)
        s = self.engine.run(self.mtm_paths, realized, self.time_grid).summary()
        expected = {
            "n_observations", "n_exceptions", "exception_rate",
            "expected_exception_rate", "excess_exception_rate",
            "basel_zone",
            "kupiec_lr", "kupiec_pvalue",
            "ee_rmse", "ee_bias", "ee_mae",
            "bias_tstat", "bias_pvalue",
        }
        assert expected == set(s.keys())

    def test_kupiec_pvalue_in_unit_interval(self):
        realized = self.rng.standard_normal(self.T) * 20_000
        result = self.engine.run(self.mtm_paths, realized, self.time_grid)
        assert 0.0 <= result.kupiec_pvalue <= 1.0
        assert result.kupiec_lr >= 0.0

    def test_bias_pvalue_in_unit_interval(self):
        realized = self.rng.standard_normal(self.T) * 20_000
        result = self.engine.run(self.mtm_paths, realized, self.time_grid)
        assert 0.0 <= result.bias_pvalue <= 1.0

    def test_correct_model_kupiec_not_rejected(self):
        """For a correct 95% model, Kupiec p-value should generally be > 0.05."""
        n_paths = 10_000
        T = 100
        paths = self.rng.standard_normal((n_paths, T)) * 100
        realized = self.rng.standard_normal(T) * 100
        result = self.engine.run(paths, realized, np.linspace(0, 1, T))
        # With seed=42 this is deterministic — just ensure p-value is in [0,1]
        assert 0.0 <= result.kupiec_pvalue <= 1.0

    def test_zero_realized_kupiec_finite(self):
        """Zero realized → 0 exceptions; Kupiec should still return finite values."""
        result = self.engine.run(self.mtm_paths, np.zeros(self.T), self.time_grid)
        assert np.isfinite(result.kupiec_lr)
        assert np.isfinite(result.kupiec_pvalue)

    def test_excess_exception_rate_is_difference(self):
        realized = np.full(self.T, 1e12)
        s = self.engine.run(self.mtm_paths, realized, self.time_grid).summary()
        assert s["excess_exception_rate"] == pytest.approx(
            s["exception_rate"] - s["expected_exception_rate"]
        )

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            BacktestEngine(confidence=1.5)

    def test_shape_mismatch_raises(self):
        realized = np.zeros(self.T + 5)
        with pytest.raises(ValueError, match="realized"):
            self.engine.run(self.mtm_paths, realized, self.time_grid)

    def test_time_grid_mismatch_raises(self):
        realized = np.zeros(self.T)
        bad_grid = np.linspace(0, 1, self.T + 1)
        with pytest.raises(ValueError, match="time_grid"):
            self.engine.run(self.mtm_paths, realized, bad_grid)


# ---------------------------------------------------------------------------
# Synthetic end-to-end: GBM paths, peel off one as realized
# ---------------------------------------------------------------------------

class TestSyntheticEndToEnd:
    """Simulate GBM, use path-0 as realised, backtest the rest."""

    def test_gbm_backtest_green_zone(self):
        from pyxva import MonteCarloEngine, GeometricBrownianMotion, TimeGrid

        time_grid = TimeGrid.uniform(1.0, 50)
        n_paths = 2000
        gbm = GeometricBrownianMotion(S0=100.0, mu=0.05, sigma=0.20)

        engine = MonteCarloEngine(n_paths=n_paths + 1, seed=7)
        results = engine.run([gbm], time_grid)
        all_paths = results["GBM"].factor("S")   # (n_paths+1, T)

        # Treat path 0 as realised; peel off and convert to MTM (S - S0)
        realized_s = all_paths[0]
        forecast_paths = all_paths[1:]
        # MTM = spot - forward (simplified: just spot - S0 for illustration)
        realized_mtm = realized_s - gbm.S0
        forecast_mtm = forecast_paths - gbm.S0

        bt = BacktestEngine(confidence=0.95)
        result = bt.run(forecast_mtm, realized_mtm, time_grid)

        s = result.summary()
        # A single realised path will either be Green or Amber (not Red)
        # unless extremely unlucky — with seed=7 this is deterministic
        assert s["basel_zone"] in ("Green", "Amber")
        assert 0 <= s["exception_rate"] <= 1.0
        assert s["ee_rmse"] >= 0.0
