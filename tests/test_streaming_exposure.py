"""Tests for REGVMStepper and StreamingExposureEngine."""
from __future__ import annotations

import numpy as np
import pytest

from risk_analytics.core.paths import SimulationResult
from risk_analytics.exposure.csa import CSATerms, MarginRegime, IMModel
from risk_analytics.exposure.streaming.vm_stepper import REGVMStepper
from risk_analytics.exposure.streaming.engine import StreamingExposureEngine
from risk_analytics.pricing.rates.swap import InterestRateSwap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rate_result(
    n_paths: int = 100,
    n_steps: int = 25,
    r0: float = 0.04,
    r_end: float = 0.04,
) -> SimulationResult:
    time_grid = np.linspace(0, 5.0, n_steps)
    # Fan of paths: r varies linearly from r0 to r_end with cross-sectional spread
    spread = np.linspace(-0.02, 0.02, n_paths)
    r = (
        np.linspace(r0, r_end, n_steps)[None, :]
        + spread[:, None] * np.linspace(0, 1, n_steps)[None, :]
    )
    paths = r[:, :, None]
    return SimulationResult(
        time_grid=time_grid,
        paths=paths,
        model_name="HW",
        factor_names=["r"],
    )


def _csa_zero_th() -> CSATerms:
    return CSATerms(
        counterparty_id="CP",
        margin_regime=MarginRegime.REGVM,
        threshold_party=0.0,
        threshold_counterparty=0.0,
        mta_party=0.0,
        mta_counterparty=0.0,
        mpor=10 / 252,
    )


# ---------------------------------------------------------------------------
# REGVMStepper
# ---------------------------------------------------------------------------

class TestREGVMStepper:
    def test_zero_threshold_fully_collateralised(self):
        """With zero threshold, post-margin exposure should be ~0 after first call."""
        csa = _csa_zero_th()
        stepper = REGVMStepper(csa, n_paths=5)
        net_mtm = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
        exposure = stepper.step(net_mtm)
        # CE = max(V - CSB, 0): after fully collateralised VM call, CE=0 for positive V
        # For positive V: CSB should equal V, so CE = max(V-V,0) = 0
        np.testing.assert_allclose(exposure[net_mtm > 0], 0.0, atol=1e-12)

    def test_csb_updated_after_step(self):
        csa = _csa_zero_th()
        stepper = REGVMStepper(csa, n_paths=3)
        stepper.step(np.array([10.0, -5.0, 0.0]))
        csb = stepper.csb
        # CSB should be ~10 for first path (counterparty posted VM)
        assert csb[0] == pytest.approx(10.0, abs=1e-10)

    def test_mta_suppresses_small_calls(self):
        """Calls below MTA should not update CSB."""
        csa = CSATerms(
            counterparty_id="CP",
            threshold_party=0.0, threshold_counterparty=0.0,
            mta_party=5.0, mta_counterparty=5.0,
        )
        stepper = REGVMStepper(csa, n_paths=2)
        # MTM below MTA
        stepper.step(np.array([2.0, 3.0]))
        np.testing.assert_allclose(stepper.csb, 0.0, atol=1e-10)

    def test_threshold_reduces_csb(self):
        """Threshold reduces delivery amount."""
        csa = CSATerms(
            counterparty_id="CP",
            threshold_party=3.0, threshold_counterparty=3.0,
            mta_party=0.0, mta_counterparty=0.0,
        )
        stepper = REGVMStepper(csa, n_paths=1)
        stepper.step(np.array([10.0]))
        # Should post V - TH = 10 - 3 = 7
        assert stepper.csb[0] == pytest.approx(7.0, abs=1e-10)

    def test_reset(self):
        csa = _csa_zero_th()
        stepper = REGVMStepper(csa, n_paths=3)
        stepper.step(np.array([5.0, 5.0, 5.0]))
        stepper.reset()
        np.testing.assert_allclose(stepper.csb, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# StreamingExposureEngine
# ---------------------------------------------------------------------------

class TestStreamingExposureEngine:
    def test_output_shapes(self):
        result = _rate_result(n_paths=50, n_steps=20)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        trades = [("swap1", swap)]
        csa = _csa_zero_th()
        engine = StreamingExposureEngine(trades, csa, confidence=0.95)
        out = engine.run(result)
        n = len(result.time_grid)
        assert out.ee_profile.shape == (n,)
        assert out.ene_profile.shape == (n,)
        assert out.pfe_profile.shape == (n,)
        assert out.ee_mpor_profile.shape == (n,)

    def test_ee_non_negative(self):
        result = _rate_result(n_paths=100, n_steps=25)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        engine = StreamingExposureEngine([("swap1", swap)], _csa_zero_th())
        out = engine.run(result)
        assert np.all(out.ee_profile >= -1e-10)

    def test_pfe_geq_ee(self):
        """At high confidence, PFE should be >= EE."""
        result = _rate_result(n_paths=200, n_steps=25)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        engine = StreamingExposureEngine([("swap1", swap)], _csa_zero_th(), confidence=0.99)
        out = engine.run(result)
        assert np.all(out.pfe_profile >= out.ee_profile - 1e-8)

    def test_scalars(self):
        result = _rate_result(n_paths=50, n_steps=15)
        swap = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6)
        engine = StreamingExposureEngine([("swap1", swap)], _csa_zero_th())
        out = engine.run(result)
        assert out.peak_ee == pytest.approx(float(np.max(out.ee_profile)))
        assert out.peak_pfe == pytest.approx(float(np.max(out.pfe_profile)))

    def test_two_trades_sum_mtm(self):
        """Two equal offsetting swaps should produce near-zero exposure."""
        result = _rate_result(n_paths=100, n_steps=15)
        pay = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6, payer=True)
        rec = InterestRateSwap(fixed_rate=0.04, maturity=5.0, notional=1e6, payer=False)
        trades = [("pay", pay), ("rec", rec)]
        engine = StreamingExposureEngine(trades, _csa_zero_th())
        out = engine.run(result)
        np.testing.assert_allclose(out.ee_profile, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# REGVMStepper property-based invariants
# ---------------------------------------------------------------------------

class TestREGVMStepperProperties:
    """Property tests over random MTM paths — not hand-crafted inputs."""

    def test_csb_tracks_mtm_exactly_zero_threshold_zero_mta(self):
        """With threshold=0 and mta=0, CSB must equal V(t) after every step."""
        rng = np.random.default_rng(42)
        csa = _csa_zero_th()
        n_paths, n_steps = 500, 60
        stepper = REGVMStepper(csa, n_paths=n_paths)

        for _ in range(n_steps):
            mtm = rng.normal(0, 100, size=n_paths)
            stepper.step(mtm)
            np.testing.assert_allclose(stepper.csb, mtm, atol=1e-10,
                err_msg="CSB must equal V(t) at every step when threshold=mta=0")

    def test_post_margin_ce_zero_for_positive_mtm(self):
        """With threshold=0 and mta=0, CE must be 0 for all positive-MTM paths."""
        rng = np.random.default_rng(99)
        csa = _csa_zero_th()
        n_paths, n_steps = 500, 40
        stepper = REGVMStepper(csa, n_paths=n_paths)

        for _ in range(n_steps):
            mtm = rng.normal(50, 30, size=n_paths)  # mostly positive
            exposure = stepper.step(mtm)
            # CE = max(V - CSB, 0); since CSB=V after step, CE must be 0
            positive = mtm > 0
            np.testing.assert_allclose(exposure[positive], 0.0, atol=1e-10,
                err_msg="Post-margin CE must be 0 for positive MTM paths when fully collateralised")

    def test_initial_csb_equals_net_ia(self):
        """Initial CSB (before any step) must equal ia_counterparty - ia_party."""
        ia_cpty, ia_party = 100.0, 30.0
        csa = CSATerms(
            counterparty_id="CP",
            margin_regime=MarginRegime.REGVM,
            threshold_party=0.0,
            threshold_counterparty=0.0,
            mta_party=0.0,
            mta_counterparty=0.0,
            ia_party=ia_party,
            ia_counterparty=ia_cpty,
        )
        stepper = REGVMStepper(csa, n_paths=50)
        expected_floor = ia_cpty - ia_party
        np.testing.assert_allclose(stepper.csb, expected_floor, atol=1e-12)
