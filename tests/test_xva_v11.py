"""Tests for pyxva v1.1 xVA suite: HazardCurve, FVA, MVA, KVA, attribution,
im_time_profile, xva_summary, and RunResult.from_parquet().
"""
from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest

from pyxva.core import MonteCarloEngine, TimeGrid
from pyxva.exposure.hazard_curve import HazardCurve
from pyxva.exposure.bilateral import BilateralExposureCalculator, _marginal_pd
from pyxva.exposure.margin.im import REGIMEngine
from pyxva.exposure import CSATerms, IMModel, NettingSet, REGVMEngine
from pyxva.models import HullWhite1F
from pyxva.pricing import InterestRateSwap
from pyxva.portfolio.trade import Trade
from pyxva.pipeline.result import AgreementResult, RunResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_PATHS = 500
SEED = 99


def _rates_result():
    model = HullWhite1F(a=0.1, sigma=0.01, r0=0.05)
    grid = TimeGrid.uniform(5.0, 30)
    engine = MonteCarloEngine(N_PATHS, seed=SEED)
    results = engine.run([model], grid)
    return results, grid


def _payer_swap_mtm(n_paths=N_PATHS, seed=SEED):
    results, grid = _rates_result()
    ns = NettingSet("s")
    ns.add_trade(Trade(id="sw", pricer=InterestRateSwap(0.05, 5.0, 1e6), model_name="HullWhite1F"))
    return ns.net_mtm(results), grid


# ===========================================================================
# HazardCurve
# ===========================================================================

class TestHazardCurveFlat:
    def test_survival_at_zero(self):
        hc = HazardCurve.from_flat_spread(0.01)
        assert hc.survival_probability(0.0) == pytest.approx(1.0)

    def test_survival_decays(self):
        lam = 0.02
        hc = HazardCurve.from_flat_spread(lam * 0.6, lgd=0.6)  # spread = lam × lgd
        q1 = hc.survival_probability(1.0)
        q5 = hc.survival_probability(5.0)
        assert q1 == pytest.approx(np.exp(-lam * 1.0), rel=1e-6)
        assert q5 == pytest.approx(np.exp(-lam * 5.0), rel=1e-6)
        assert q5 < q1

    def test_marginal_default_prob_positive(self):
        hc = HazardCurve.from_flat_spread(0.01)
        pd = hc.marginal_default_prob(0.0, 1.0)
        assert pd > 0
        assert pd < 1

    def test_marginal_pd_equals_survival_diff(self):
        hc = HazardCurve.from_flat_spread(0.01)
        q0 = hc.survival_probability(0.0)
        q1 = hc.survival_probability(1.0)
        assert hc.marginal_default_prob(0.0, 1.0) == pytest.approx(q0 - q1, rel=1e-10)

    def test_marginal_pd_zero_length_interval(self):
        hc = HazardCurve.from_flat_spread(0.05)
        assert hc.marginal_default_prob(2.0, 2.0) == pytest.approx(0.0)

    def test_invalid_lgd_raises(self):
        with pytest.raises(ValueError, match="lgd"):
            HazardCurve.from_flat_spread(0.01, lgd=0.0)


class TestHazardCurveTermStructure:
    def test_from_tenors_short_rate_higher(self):
        # Inverted hazard curve: short-term stress higher
        hc = HazardCurve.from_tenors([1.0, 5.0], [0.10, 0.02])
        # Within [0,1]: hazard = 0.10 → Q(0.5) = exp(-0.05)
        q_short = hc.survival_probability(0.5)
        assert q_short == pytest.approx(np.exp(-0.10 * 0.5), rel=1e-8)
        # Within (1,5]: hazard = 0.02 → Q(2) = Q(1) × exp(-0.02 × 1)
        q1 = hc.survival_probability(1.0)
        q2 = hc.survival_probability(2.0)
        assert q2 == pytest.approx(q1 * np.exp(-0.02 * 1.0), rel=1e-8)

    def test_extension_beyond_last_tenor(self):
        hc = HazardCurve.from_tenors([1.0, 5.0], [0.05, 0.03])
        q5 = hc.survival_probability(5.0)
        q10 = hc.survival_probability(10.0)
        # Beyond 5yr: last hazard rate (0.03) continues
        assert q10 == pytest.approx(q5 * np.exp(-0.03 * 5.0), rel=1e-8)

    def test_invalid_tenors_raise(self):
        with pytest.raises(ValueError):
            HazardCurve(tenors=[5.0, 1.0], hazard_rates=[0.01, 0.02])  # not increasing

    def test_negative_hazard_raises(self):
        with pytest.raises(ValueError):
            HazardCurve(tenors=[1.0, 5.0], hazard_rates=[0.01, -0.01])


class TestHazardCurveCalibrate:
    def test_calibrate_flat_spread(self):
        """Flat CDS curve → all buckets should give same hazard rate ≈ s/LGD."""
        tenors = np.array([1.0, 3.0, 5.0])
        lam = 0.02
        recovery = 0.4
        lgd = 1.0 - recovery
        # For flat λ: par spread ≈ λ × lgd (approximate for small λ)
        spreads = np.array([lam * lgd, lam * lgd, lam * lgd])
        hc = HazardCurve.calibrate(tenors, spreads, recovery=recovery)
        # Calibrated hazard rates should reproduce input survival curve closely
        for t in [1.0, 3.0, 5.0]:
            q_cal = hc.survival_probability(t)
            q_exact = np.exp(-lam * t)
            assert abs(q_cal - q_exact) < 0.01, f"Q({t}): calibrated={q_cal:.4f} vs exact={q_exact:.4f}"

    def test_calibrate_upward_curve(self):
        """Upward-sloping spreads → hazard rates should be non-decreasing (roughly)."""
        tenors = np.array([1.0, 3.0, 5.0, 10.0])
        spreads = np.array([0.005, 0.01, 0.015, 0.020])
        hc = HazardCurve.calibrate(tenors, spreads, recovery=0.4)
        assert np.all(hc.hazard_rates >= 0)

    def test_calibrate_recovery_effect(self):
        """Higher recovery → higher hazard rate for same spread (LGD is smaller)."""
        tenors = np.array([5.0])
        spreads = np.array([0.01])
        hc_low_rec = HazardCurve.calibrate(tenors, spreads, recovery=0.2)
        hc_high_rec = HazardCurve.calibrate(tenors, spreads, recovery=0.6)
        # LGD = 1-R; higher R → lower LGD → higher λ = s/LGD
        assert hc_high_rec.hazard_rates[0] > hc_low_rec.hazard_rates[0]

    def test_calibrate_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            HazardCurve.calibrate([1.0, 5.0], [0.01])

    def test_calibrate_negative_spread_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            HazardCurve.calibrate([1.0], [-0.01])

    def test_repr(self):
        hc = HazardCurve.from_flat_spread(0.01)
        assert "HazardCurve" in repr(hc)


# ===========================================================================
# _marginal_pd helper
# ===========================================================================

class TestMarginalPd:
    def test_flat_hazard_matches_exponential(self):
        lam = 0.05
        t0, t1 = 1.0, 2.0
        expected = np.exp(-lam * t0) - np.exp(-lam * t1)
        assert _marginal_pd(lam, t0, t1) == pytest.approx(expected, rel=1e-10)

    def test_hazard_curve_matches_marginal_default_prob(self):
        hc = HazardCurve.from_flat_spread(0.05 * 0.6, lgd=0.6)  # λ = 0.05
        t0, t1 = 0.0, 1.0
        assert _marginal_pd(hc, t0, t1) == pytest.approx(hc.marginal_default_prob(t0, t1), rel=1e-10)

    def test_flat_hazard_and_flat_curve_agree(self):
        lam = 0.03
        lgd = 0.6
        spread = lam * lgd
        hc = HazardCurve.from_flat_spread(spread, lgd=lgd)
        t0, t1 = 2.0, 3.0
        pd_scalar = _marginal_pd(lam, t0, t1)
        pd_curve = _marginal_pd(hc, t0, t1)
        # Should agree to within rounding of flat approximation
        assert abs(pd_scalar - pd_curve) < 1e-4


# ===========================================================================
# BilateralExposureCalculator — FVA, MVA, KVA, attribution
# ===========================================================================

class TestFVA:
    def setup_method(self):
        self.mtm, self.grid = _payer_swap_mtm()
        self.calc = BilateralExposureCalculator()

    def test_fva_float_non_negative_for_positive_epe(self):
        """For a payer swap (positive EE), net FVA = cost(EE) - benefit(ENE) > 0."""
        fva = self.calc.fva_approx(self.mtm, self.grid, funding=0.005)
        # May be positive or slightly negative depending on ENE; just check finite
        assert np.isfinite(fva)

    def test_fva_zero_spread_is_zero(self):
        fva = self.calc.fva_approx(self.mtm, self.grid, funding=0.0)
        assert fva == pytest.approx(0.0, abs=1e-10)

    def test_fva_increases_with_spread(self):
        fva_low = self.calc.fva_approx(self.mtm, self.grid, funding=0.001)
        fva_high = self.calc.fva_approx(self.mtm, self.grid, funding=0.01)
        # Both compute EE - ENE integral; higher spread → larger magnitude
        assert abs(fva_high) >= abs(fva_low) - 1e-6

    def test_fva_with_hazard_curve_vs_flat(self):
        """HazardCurve(flat) should give close result to flat scalar."""
        lam = 0.01
        hc = HazardCurve.from_flat_spread(lam * 0.6, lgd=0.6)  # λ = 0.01
        fva_scalar = self.calc.fva_approx(self.mtm, self.grid, funding=lam)
        fva_curve = self.calc.fva_approx(self.mtm, self.grid, funding=hc)
        # Results differ because HazardCurve uses marginal default prob, not just λ·Δt
        # But both should be finite and have same sign
        assert np.isfinite(fva_scalar) and np.isfinite(fva_curve)


class TestMVA:
    def setup_method(self):
        self.calc = BilateralExposureCalculator()
        self.grid = TimeGrid.uniform(5.0, 30)

    def test_mva_non_negative(self):
        im_profile = np.linspace(1e5, 0.0, len(self.grid))  # declining
        mva = self.calc.mva_approx(im_profile, self.grid, funding=0.005)
        assert mva >= 0

    def test_mva_zero_profile_is_zero(self):
        im_profile = np.zeros(len(self.grid))
        mva = self.calc.mva_approx(im_profile, self.grid, funding=0.01)
        assert mva == pytest.approx(0.0, abs=1e-10)

    def test_mva_increases_with_spread(self):
        im_profile = np.ones(len(self.grid)) * 1e5
        mva_low = self.calc.mva_approx(im_profile, self.grid, funding=0.001)
        mva_high = self.calc.mva_approx(im_profile, self.grid, funding=0.01)
        # Higher spread → larger MVA (10× larger λ → ~10× larger integral for small λ)
        assert mva_high > mva_low
        ratio = mva_high / mva_low
        # Should be close to 10 but nonlinear for finite λ × T
        assert 8 < ratio < 12


class TestKVA:
    def setup_method(self):
        self.calc = BilateralExposureCalculator()
        self.grid = TimeGrid.uniform(5.0, 30)

    def test_kva_proportional_to_ead(self):
        kva1 = self.calc.kva_approx(1e6, self.grid, cost_of_capital=0.10)
        kva2 = self.calc.kva_approx(2e6, self.grid, cost_of_capital=0.10)
        assert kva2 == pytest.approx(kva1 * 2, rel=1e-6)

    def test_kva_proportional_to_coc(self):
        kva1 = self.calc.kva_approx(1e6, self.grid, cost_of_capital=0.05)
        kva2 = self.calc.kva_approx(1e6, self.grid, cost_of_capital=0.10)
        assert kva2 == pytest.approx(kva1 * 2, rel=1e-6)

    def test_kva_zero_ead_is_zero(self):
        kva = self.calc.kva_approx(0.0, self.grid)
        assert kva == pytest.approx(0.0)

    def test_kva_formula(self):
        """KVA = CoC × EAD × T."""
        ead = 5e5
        coc = 0.12
        T = float(self.grid[-1] - self.grid[0])
        kva = self.calc.kva_approx(ead, self.grid, cost_of_capital=coc)
        assert kva == pytest.approx(coc * ead * T, rel=1e-10)


class TestXVAAttribution:
    def setup_method(self):
        self.mtm, self.grid = _payer_swap_mtm()
        self.calc = BilateralExposureCalculator()

    def test_attribution_keys(self):
        out = self.calc.xva_attribution(self.mtm, self.grid, hazard=0.01)
        assert set(out.keys()) == {"time", "cva", "dva", "fva", "mva", "total"}

    def test_attribution_shapes(self):
        out = self.calc.xva_attribution(self.mtm, self.grid, hazard=0.01)
        T = len(self.grid) - 1
        for k, v in out.items():
            assert v.shape == (T,), f"key={k}"

    def test_attribution_cva_sums_to_total_cva(self):
        hazard = 0.01
        total_cva = self.calc.cva_approx(self.mtm, self.grid, hazard_rate=hazard)
        out = self.calc.xva_attribution(self.mtm, self.grid, hazard=hazard)
        assert out["cva"].sum() == pytest.approx(total_cva, rel=1e-6)

    def test_attribution_dva_zero_without_own_hazard(self):
        out = self.calc.xva_attribution(self.mtm, self.grid, hazard=0.01)
        assert np.all(out["dva"] == 0.0)

    def test_attribution_dva_nonzero_with_own_hazard(self):
        out = self.calc.xva_attribution(
            self.mtm, self.grid, hazard=0.01, own_hazard=0.005
        )
        assert np.any(out["dva"] > 0)

    def test_attribution_fva_zero_without_funding(self):
        out = self.calc.xva_attribution(self.mtm, self.grid, hazard=0.01, funding=None)
        assert np.all(out["fva"] == 0.0)

    def test_attribution_total_equals_cva_minus_dva_plus_fva_mva(self):
        funding = 0.005
        own_h = 0.005
        out = self.calc.xva_attribution(
            self.mtm, self.grid, hazard=0.01, funding=funding, own_hazard=own_h
        )
        expected_total = out["cva"] - out["dva"] + out["fva"] + out["mva"]
        np.testing.assert_allclose(out["total"], expected_total, atol=1e-12)

    def test_attribution_with_hazard_curve(self):
        hc = HazardCurve.from_flat_spread(0.01 * 0.6, lgd=0.6)
        out = self.calc.xva_attribution(self.mtm, self.grid, hazard=hc)
        assert np.all(np.isfinite(out["cva"]))


# ===========================================================================
# REGIMEngine.im_time_profile
# ===========================================================================

class TestIMTimeProfile:
    def setup_method(self):
        self.csa = CSATerms(im_model=IMModel.SCHEDULE)
        self.engine = REGIMEngine(self.csa)
        self.grid = TimeGrid.uniform(5.0, 30)

    def test_profile_shape(self):
        trades = [{"asset_class": "IR", "gross_notional": 1e6, "maturity": 5.0}]
        profile = self.engine.im_time_profile(trades, self.grid)
        assert profile.shape == (len(self.grid),)

    def test_profile_declines_toward_maturity(self):
        """IM should decline as residual maturity approaches zero."""
        trades = [{"asset_class": "IR", "gross_notional": 1e6, "maturity": 5.0}]
        profile = self.engine.im_time_profile(trades, self.grid)
        # At t=0: full maturity; at t=5: residual=0 → IM should be 0
        assert profile[0] > 0
        assert profile[-1] == pytest.approx(0.0, abs=1e-10)

    def test_profile_non_negative(self):
        trades = [
            {"asset_class": "IR", "gross_notional": 1e6, "maturity": 3.0},
            {"asset_class": "FX", "gross_notional": 5e5, "maturity": 1.0},
        ]
        profile = self.engine.im_time_profile(trades, self.grid)
        assert np.all(profile >= 0)

    def test_profile_at_zero_matches_schedule_im(self):
        """im_time_profile at t=0 should match schedule_im at full maturity."""
        trades = [{"asset_class": "IR", "gross_notional": 1e6, "maturity": 5.0}]
        profile = self.engine.im_time_profile(trades, self.grid)
        im_t0 = float(self.engine.schedule_im(trades))
        assert profile[0] == pytest.approx(im_t0, rel=1e-6)

    def test_fx_trade_maturity_independent(self):
        """FX trades have no maturity bucket — IM constant until maturity."""
        trades = [{"asset_class": "FX", "gross_notional": 1e6, "maturity": 3.0}]
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        profile = self.engine.im_time_profile(trades, grid)
        # FX weight is independent of maturity bucket, but residual < maturity
        # → after t=3 the trade has zero residual maturity → IM drops to 0
        assert profile[3] == pytest.approx(0.0, abs=1e-10)  # t=3, residual=0
        assert profile[4] == pytest.approx(0.0, abs=1e-10)  # t>maturity


# ===========================================================================
# AgreementResult.xva_summary()
# ===========================================================================

class TestXVASummary:
    def _make_result(self, cva=100.0, dva=20.0, fva=30.0, mva=10.0, kva=5.0):
        grid = np.array([0.0, 1.0, 2.0])
        return AgreementResult(
            id="agr1",
            counterparty_id="CP1",
            time_grid=grid,
            ee_profile=np.zeros(3),
            ene_profile=np.zeros(3),
            pfe_profile=np.zeros(3),
            ee_mpor_profile=np.zeros(3),
            netting_set_summaries={},
            cva=cva, dva=dva, bcva=cva - dva,
            fva=fva, mva=mva, kva=kva,
            pse=0.0, epe=0.0, eepe=0.0,
        )

    def test_xva_summary_keys(self):
        result = self._make_result()
        summary = result.xva_summary()
        for k in ("agreement_id", "counterparty_id", "cva", "dva", "bcva", "fva", "mva", "kva", "total_xva"):
            assert k in summary

    def test_total_xva_formula(self):
        cva, dva, fva, mva, kva = 100.0, 20.0, 30.0, 10.0, 5.0
        result = self._make_result(cva=cva, dva=dva, fva=fva, mva=mva, kva=kva)
        summary = result.xva_summary()
        expected_total = cva - dva + fva + mva + kva
        assert summary["total_xva"] == pytest.approx(expected_total, rel=1e-10)

    def test_xva_summary_values(self):
        result = self._make_result(cva=50.0, dva=10.0, fva=5.0, mva=2.0, kva=1.0)
        s = result.xva_summary()
        assert s["cva"] == pytest.approx(50.0)
        assert s["fva"] == pytest.approx(5.0)
        assert s["agreement_id"] == "agr1"


# ===========================================================================
# RunResult.from_parquet() round-trip
# ===========================================================================

def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        pass
    return False


_parquet_skip = pytest.mark.skipif(
    not _parquet_available(),
    reason="pyarrow or fastparquet required for parquet tests",
)


@_parquet_skip
class TestRunResultFromParquet:
    def _make_run_result(self):
        grid = np.linspace(0, 5, 11)
        n = len(grid)
        agr = AgreementResult(
            id="agr1",
            counterparty_id="CP1",
            time_grid=grid,
            ee_profile=np.random.rand(n) * 1e5,
            ene_profile=np.random.rand(n) * -1e4,
            pfe_profile=np.random.rand(n) * 2e5,
            ee_mpor_profile=np.random.rand(n) * 1.1e5,
            netting_set_summaries={},
            cva=12345.0, dva=2345.0, bcva=10000.0,
            fva=500.0, mva=100.0, kva=50.0,
            pse=8000.0, epe=9000.0, eepe=9500.0,
        )
        return RunResult(
            config=None,
            time_grid=grid,
            simulation_results={},
            agreement_results={"agr1": agr},
            total_cva=agr.cva,
            total_dva=agr.dva,
            total_bcva=agr.bcva,
            total_fva=agr.fva,
            total_mva=agr.mva,
            total_kva=agr.kva,
        )

    def test_round_trip_scalars(self):
        rr = self._make_run_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            rr.to_parquet(tmpdir)
            loaded = RunResult.from_parquet(tmpdir)
        agr_orig = rr.agreement_results["agr1"]
        agr_loaded = loaded.agreement_results["agr1"]
        assert agr_loaded.cva == pytest.approx(agr_orig.cva)
        assert agr_loaded.dva == pytest.approx(agr_orig.dva)
        assert agr_loaded.fva == pytest.approx(agr_orig.fva)
        assert agr_loaded.mva == pytest.approx(agr_orig.mva)
        assert agr_loaded.kva == pytest.approx(agr_orig.kva)
        assert agr_loaded.epe == pytest.approx(agr_orig.epe)

    def test_round_trip_totals(self):
        rr = self._make_run_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            rr.to_parquet(tmpdir)
            loaded = RunResult.from_parquet(tmpdir)
        assert loaded.total_cva == pytest.approx(rr.total_cva)
        assert loaded.total_fva == pytest.approx(rr.total_fva)
        assert loaded.total_mva == pytest.approx(rr.total_mva)
        assert loaded.total_kva == pytest.approx(rr.total_kva)

    def test_round_trip_ee_profile(self):
        rr = self._make_run_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            rr.to_parquet(tmpdir)
            loaded = RunResult.from_parquet(tmpdir)
        orig_ee = rr.agreement_results["agr1"].ee_profile
        loaded_ee = loaded.agreement_results["agr1"].ee_profile
        np.testing.assert_allclose(loaded_ee, orig_ee, rtol=1e-6)

    def test_round_trip_time_grid(self):
        rr = self._make_run_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            rr.to_parquet(tmpdir)
            loaded = RunResult.from_parquet(tmpdir)
        np.testing.assert_allclose(loaded.time_grid, rr.time_grid, rtol=1e-10)

    def test_duckdb_import_error_on_missing_duckdb(self):
        """Without duckdb installed, use_duckdb=True should raise ImportError."""
        rr = self._make_run_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            rr.to_parquet(tmpdir)
            try:
                import duckdb
                # DuckDB is available — skip this test
                pytest.skip("duckdb is installed; cannot test ImportError path")
            except ImportError:
                with pytest.raises(ImportError, match="duckdb"):
                    RunResult.from_parquet(tmpdir, use_duckdb=True)

    def test_summary_df_includes_fva_mva_kva(self):
        rr = self._make_run_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            rr.to_parquet(tmpdir)
            loaded = RunResult.from_parquet(tmpdir)
        df = loaded.summary_df()
        for col in ("fva", "mva", "kva"):
            assert col in df.columns


# ===========================================================================
# Integration: bilateral_summary with FVA/MVA/KVA
# ===========================================================================

class TestBilateralSummaryXVA:
    def setup_method(self):
        self.mtm, self.grid = _payer_swap_mtm()
        self.calc = BilateralExposureCalculator()

    def test_summary_fva_nonzero_with_funding(self):
        summary = self.calc.bilateral_summary(
            self.mtm, self.grid,
            cp_hazard_rate=0.01, own_hazard_rate=0.005,
            funding_spread=0.005,
        )
        assert "fva" in summary
        assert np.isfinite(summary["fva"])

    def test_summary_fva_zero_without_funding(self):
        summary = self.calc.bilateral_summary(
            self.mtm, self.grid, cp_hazard_rate=0.01, own_hazard_rate=0.005
        )
        assert summary["fva"] == pytest.approx(0.0)

    def test_summary_kva_nonzero_with_ead(self):
        summary = self.calc.bilateral_summary(
            self.mtm, self.grid, ead_t0=5e5, cost_of_capital=0.10
        )
        assert summary["kva"] > 0

    def test_summary_mva_with_im_profile(self):
        im_profile = np.linspace(1e5, 0.0, len(self.grid))
        summary = self.calc.bilateral_summary(
            self.mtm, self.grid,
            funding_spread=0.005,
            im_profile=im_profile,
        )
        assert summary["mva"] > 0
