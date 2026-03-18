from __future__ import annotations

import numpy as np

from pyxva.core.paths import SimulationResult
from .metrics import ExposureCalculator


class NettingSet:
    """Aggregate exposure across trades under a bilateral netting agreement.

    Under netting, the exposure is applied to the *net* MTM of the portfolio,
    not the sum of individual positive exposures. This reduces credit risk.

    Parameters
    ----------
    name : str
        Identifier for this netting set (e.g. counterparty name).
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._trades: list = []  # list[Trade]

    @property
    def id(self) -> str:
        """Alias for name — used by Agreement for dict keying."""
        return self.name

    def add_trade(self, trade) -> None:
        """Add a Trade to the netting set.

        Parameters
        ----------
        trade : Trade
            A Trade object binding a pricer to its model via ``trade.model_name``.
        """
        from pyxva.portfolio.trade import Trade as _Trade

        if not isinstance(trade, _Trade):
            raise TypeError(
                f"add_trade() requires a Trade object; got {type(trade).__name__}. "
                "Construct one with: Trade(id=..., pricer=..., model_name=...)"
            )
        self._trades.append(trade)

    def net_mtm(self, simulation_results: dict[str, SimulationResult]) -> np.ndarray:
        """Compute the net MTM across all trades in the netting set.

        Each trade's pricer is called with the SimulationResult identified by
        ``trade.model_name``. A KeyError is raised immediately if a trade's
        declared model is not present in ``simulation_results``.

        Parameters
        ----------
        simulation_results : dict[str, SimulationResult]
            Output from MonteCarloEngine.run().

        Returns
        -------
        np.ndarray, shape (n_paths, T)
            Net MTM across all trades.
        """
        if not self._trades:
            raise ValueError("NettingSet has no trades.")

        net = None
        for trade in self._trades:
            if trade.model_name not in simulation_results:
                raise KeyError(
                    f"Trade '{trade.id}': model '{trade.model_name}' not found in "
                    f"simulation results. Available: {list(simulation_results)}"
                )
            mtm = trade.pricer.price(simulation_results[trade.model_name])
            net = mtm if net is None else net + mtm

        return net

    def exposure(
        self,
        simulation_results: dict[str, SimulationResult],
        time_grid: np.ndarray,
        confidence: float = 0.95,
    ) -> dict:
        """Compute full exposure summary for this netting set.

        Returns
        -------
        dict with keys: 'ee_profile', 'pfe_profile', 'pse', 'epe', 'net_mtm'
        """
        net = self.net_mtm(simulation_results)
        calc = ExposureCalculator()
        summary = calc.exposure_summary(net, time_grid, confidence)
        summary["net_mtm"] = net
        summary["netting_set"] = self.name
        return summary

    @property
    def trade_ids(self) -> list[str]:
        return [t.id for t in self._trades]
