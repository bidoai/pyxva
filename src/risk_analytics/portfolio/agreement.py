from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from risk_analytics.exposure.netting import NettingSet
from risk_analytics.exposure.csa import CSATerms
from risk_analytics.portfolio.trade import Trade
from risk_analytics.core.paths import SimulationResult


@dataclass
class Agreement:
    """
    Legal/computation scope: an ISDA Master Agreement covering one or more
    NettingSets under a single CSA.

    Key property: margin (VM) is computed on the *aggregate* MTM across all
    netting sets — the CSA does not floor per netting set.

    Collateralised exposure and CVA/DVA are computed at Agreement level.
    Per-netting-set metrics are available pre-collateral only.
    """

    id: str
    counterparty_id: str
    netting_sets: list[NettingSet] = field(default_factory=list)
    csa: CSATerms = field(default_factory=CSATerms)

    # ------------------------------------------------------------------
    # Trade accessors
    # ------------------------------------------------------------------

    def all_trades(self) -> list[Trade]:
        """Flat list of all Trade objects across all netting sets."""
        trades = []
        for ns in self.netting_sets:
            for trade_id, pricer in ns._trades:
                if isinstance(pricer, Trade):
                    trades.append(pricer)
                else:
                    # backward-compat: bare Pricer stored directly in NettingSet
                    trades.append(Trade(id=trade_id, pricer=pricer, model_name=""))
        return trades

    def all_cashflow_times(self) -> list:
        """
        Union of cashflow times across all trades in all netting sets.
        Used by the pipeline to augment the sparse simulation grid.
        """
        times: set = set()
        for trade in self.all_trades():
            times.update(trade.cashflow_times())
        return sorted(times)

    # ------------------------------------------------------------------
    # MTM aggregation
    # ------------------------------------------------------------------

    def aggregate_mtm(
        self, simulation_results: dict[str, SimulationResult]
    ) -> np.ndarray:
        """
        Sum of net MTM across all netting sets — no per-netting-set floor.
        Shape: (n_paths, T).
        """
        total = None
        for ns in self.netting_sets:
            mtm = ns.net_mtm(simulation_results)
            total = mtm if total is None else total + mtm
        if total is None:
            raise ValueError(f"Agreement '{self.id}' has no netting sets.")
        return total

    def netting_set_mtms(
        self, simulation_results: dict[str, SimulationResult]
    ) -> dict[str, np.ndarray]:
        """
        Pre-collateral net MTM per netting set.
        Returns dict mapping netting_set_id → (n_paths, T).
        """
        return {ns.name: ns.net_mtm(simulation_results) for ns in self.netting_sets}

    def __repr__(self) -> str:
        ns_ids = [ns.name for ns in self.netting_sets]
        return (
            f"Agreement(id={self.id!r}, counterparty={self.counterparty_id!r}, "
            f"netting_sets={ns_ids})"
        )
