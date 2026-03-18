from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from risk_analytics.core.paths import SimulationResult
from risk_analytics.core.stateful import StatefulPricer, PathState
from risk_analytics.exposure.csa import CSATerms
from risk_analytics.exposure.streaming.vm_stepper import REGVMStepper

logger = logging.getLogger(__name__)


@dataclass
class StreamingExposureResult:
    """Output from a single StreamingExposureEngine run.

    All profiles are over the simulation time grid.
    """
    time_grid: np.ndarray
    ee_profile: np.ndarray        # expected exposure E[max(V,0)] per step
    ene_profile: np.ndarray       # expected negative exposure E[max(-V,0)] per step
    pfe_profile: np.ndarray       # peak future exposure at given confidence
    ee_mpor_profile: np.ndarray   # EE under MPOR look-ahead
    peak_ee: float                # max of ee_profile
    peak_pfe: float               # max of pfe_profile


class StreamingExposureEngine:
    """Memory-efficient Monte Carlo exposure engine.

    Computes EE, PFE and MPOR-adjusted EE by stepping through the
    simulation time grid one step at a time, never materialising the
    full ``(n_paths, T)`` MTM matrix in memory.

    Supports both standard ``Pricer`` and ``StatefulPricer`` instruments.
    For stateful pricers the path-state is tracked across steps.

    Parameters
    ----------
    trades : list of (trade_id: str, pricer: Pricer) tuples
        Instruments to include in the netting set.  All pricers must
        operate on the same ``SimulationResult`` (single model).
    csa : CSATerms
        CSA parameters driving the VM stepper.
    confidence : float
        Quantile for PFE (default 0.95).
    """

    def __init__(
        self,
        trades: list,
        csa: CSATerms,
        confidence: float = 0.95,
    ) -> None:
        self.trades = trades
        self.csa = csa
        self.confidence = confidence

    def run(
        self,
        result: SimulationResult,
        mpor_steps: Optional[int] = None,
    ) -> StreamingExposureResult:
        """Run the streaming exposure calculation.

        Parameters
        ----------
        result : SimulationResult
            Output of a single model's simulate() call.
        mpor_steps : int, optional
            Number of time steps corresponding to the MPOR look-ahead.
            Defaults to the number of steps nearest to ``csa.mpor`` years.

        Returns
        -------
        StreamingExposureResult
        """
        time_grid = result.time_grid
        n_paths = result.n_paths
        n_steps = result.n_steps

        # Resolve MPOR look-ahead in steps
        if mpor_steps is None:
            dt_grid = np.mean(np.diff(time_grid)) if n_steps > 1 else 1 / 252
            mpor_steps = max(1, int(round(self.csa.mpor / dt_grid)))

        # Initialise per-pricer stateful state (None for non-stateful pricers)
        states: list[Optional[PathState]] = []
        for _, pricer in self.trades:
            if isinstance(pricer, StatefulPricer):
                states.append(pricer.allocate_state(n_paths))
            else:
                states.append(None)

        vm_stepper = REGVMStepper(self.csa, n_paths)

        # Buffers: full profiles
        ee_profile = np.zeros(n_steps)
        ene_profile = np.zeros(n_steps)
        pfe_profile = np.zeros(n_steps)
        ee_mpor_profile = np.zeros(n_steps)

        # Buffer of recent net MTM for MPOR look-ahead (ring buffer)
        mpor_buf: list[Optional[np.ndarray]] = [None] * mpor_steps

        logger.info(
            "StreamingExposureEngine: %d trades, %d paths, %d steps, MPOR=%d steps",
            len(self.trades), n_paths, n_steps, mpor_steps,
        )

        for i, t in enumerate(time_grid):
            # --- Price all trades at this step ---
            net_mtm = np.zeros(n_paths)
            new_states: list[Optional[PathState]] = []

            for j, (trade_id, pricer) in enumerate(self.trades):
                if isinstance(pricer, StatefulPricer):
                    mtm_j, new_state_j = pricer.step(result, t, i, states[j])
                    new_states.append(new_state_j)
                else:
                    mtm_j = pricer.price_at(result, i)
                    new_states.append(None)
                net_mtm += mtm_j

            states = new_states

            # --- VM margin step ---
            post_margin = vm_stepper.step(net_mtm)

            # --- Accumulate exposure statistics ---
            ee_profile[i] = float(np.mean(post_margin))
            ene_profile[i] = float(np.mean(np.maximum(-net_mtm, 0.0)))
            pfe_profile[i] = float(np.quantile(post_margin, self.confidence))

            # --- MPOR look-ahead: store net_mtm and compute ee_mpor ---
            buf_idx = i % mpor_steps
            mpor_buf[buf_idx] = net_mtm.copy()

            # EE_MPOR: look at the mtm from `mpor_steps` steps ago and
            # apply exposure as if VM was not refreshed during the MPOR window.
            old_idx = (i - mpor_steps + 1) % mpor_steps
            old_mtm = mpor_buf[old_idx]
            if old_mtm is not None:
                ee_mpor_profile[i] = float(np.mean(np.maximum(net_mtm - old_mtm * 0.0, 0.0)))
                # More precisely: exposure at time i given last successful VM at i-mpor_steps
                # CE_mpor = max(V(t) - V(t - mpor), 0) as a rough proxy
                # (a production implementation would replay CSB from the last call)
                ee_mpor_profile[i] = float(np.mean(np.maximum(net_mtm - old_mtm, 0.0)))

        return StreamingExposureResult(
            time_grid=time_grid,
            ee_profile=ee_profile,
            ene_profile=ene_profile,
            pfe_profile=pfe_profile,
            ee_mpor_profile=ee_mpor_profile,
            peak_ee=float(np.max(ee_profile)),
            peak_pfe=float(np.max(pfe_profile)),
        )
