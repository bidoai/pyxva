"""Tests for TradeFactory.register() decorator."""
from __future__ import annotations

import pytest

from risk_analytics.pipeline.config import TradeConfig, TradeFactory
from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult
import numpy as np


class _DummyPricer(Pricer):
    def __init__(self, value: float):
        self.value = value

    def price(self, result: SimulationResult) -> np.ndarray:
        n_paths, n_steps = result.n_paths, result.n_steps
        return np.full((n_paths, n_steps), self.value)


class TestTradeFactoryRegister:
    def setup_method(self):
        # Clean up any leftover custom registrations from previous tests
        TradeFactory._CUSTOM_REGISTRY.clear()

    def test_register_and_build(self):
        @TradeFactory.register("DummyInstrument")
        def _build_dummy(params):
            return _DummyPricer(value=params.get("value", 1.0))

        cfg = TradeConfig(id="t1", type="DummyInstrument", model="m1", params={"value": 42.0})
        trade = TradeFactory.build(cfg)
        assert isinstance(trade.pricer, _DummyPricer)
        assert trade.pricer.value == 42.0

    def test_unknown_type_raises(self):
        cfg = TradeConfig(id="t2", type="AlienProduct", model="m1", params={})
        with pytest.raises(ValueError, match="AlienProduct"):
            TradeFactory.build(cfg)

    def test_custom_appears_in_error_message(self):
        @TradeFactory.register("KnownCustom")
        def _build_known(params):
            return _DummyPricer(1.0)

        cfg = TradeConfig(id="t3", type="Unknown", model="m1", params={})
        with pytest.raises(ValueError, match="KnownCustom"):
            TradeFactory.build(cfg)

    def test_builtin_still_works_after_register(self):
        @TradeFactory.register("CustomType")
        def _build_custom(params):
            return _DummyPricer(0.0)

        cfg = TradeConfig(
            id="swap1", type="InterestRateSwap", model="hw",
            params={"fixed_rate": 0.04, "maturity": 5.0},
        )
        trade = TradeFactory.build(cfg)
        from risk_analytics.pricing.rates.swap import InterestRateSwap
        assert isinstance(trade.pricer, InterestRateSwap)

    def test_register_returns_original_fn(self):
        """Decorator must return the original function unchanged."""
        def my_builder(params):
            return _DummyPricer(0.0)

        result = TradeFactory.register("MyBuilder")(my_builder)
        assert result is my_builder
