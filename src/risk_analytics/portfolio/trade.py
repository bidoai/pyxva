from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from risk_analytics.core.base import Pricer
from risk_analytics.core.paths import SimulationResult


@dataclass
class Trade:
    """
    Lightweight wrapper that binds a Pricer to the model whose
    SimulationResult it consumes.

    A Trade belongs to exactly one NettingSet (determined by legal/CSA scope)
    but may appear in multiple business portfolios (reporting groupings).
    """

    id: str
    pricer: Pricer
    model_name: str   # key into dict[str, SimulationResult]

    def price(self, simulation_results: dict[str, SimulationResult]) -> np.ndarray:
        """Price on the relevant SimulationResult. Returns (n_paths, T)."""
        if self.model_name not in simulation_results:
            raise KeyError(
                f"Trade '{self.id}': model '{self.model_name}' not found in "
                f"simulation results. Available: {list(simulation_results)}"
            )
        return self.pricer.price(simulation_results[self.model_name])

    def cashflow_times(self) -> list:
        """Delegate to the underlying pricer."""
        return self.pricer.cashflow_times()

    def __repr__(self) -> str:
        return f"Trade(id={self.id!r}, model={self.model_name!r}, pricer={type(self.pricer).__name__})"
