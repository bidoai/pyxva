"""Regulatory Variation Margin (REGVM) engine.

Implements the bilateral VM call calculation per:
- ISDA 2016 Credit Support Annex for Variation Margin (VM CSA)
- EMIR Article 11 / CFTC §23.504 (for REGVM regime)
- ISDA 1994 CSA (for LEGACY regime)
"""
from __future__ import annotations

from math import ceil, floor

import numpy as np
from numba import njit, prange

from pyxva.exposure.csa import CSATerms, MarginRegime


@njit(parallel=True, cache=True)
def _path_csb_nb(
    target: np.ndarray,
    call_mask: np.ndarray,
    mta_p: float,
    mta_c: float,
    rounding_nearest: float,
) -> np.ndarray:
    """Path-dependent CSB loop — compiled with Numba.

    Parallelised over paths (prange).  Each path's time-step loop is sequential
    (CSB depends on the previous step's balance), but paths are fully independent,
    so there are no data races.  The (n_paths, T) row-major layout means each path
    occupies a contiguous row, giving cache-friendly access for the inner time loop.
    """
    n_paths, T = target.shape
    csb = np.empty((n_paths, T))
    for p in prange(n_paths):
        csb[p, 0] = target[p, 0]
        for i in range(1, T):
            if not call_mask[i]:
                csb[p, i] = csb[p, i - 1]
            else:
                excess = target[p, i] - csb[p, i - 1]
                if excess >= mta_p or excess <= -mta_c:
                    new_val = target[p, i]
                    if rounding_nearest > 0.0:
                        unit = rounding_nearest
                        if new_val >= 0.0:
                            new_val = floor(new_val / unit) * unit
                        else:
                            new_val = ceil(new_val / unit) * unit
                    csb[p, i] = new_val
                else:
                    csb[p, i] = csb[p, i - 1]
    return csb


class REGVMEngine:
    """Computes Variation Margin calls and Credit Support Balance on simulation paths.

    Under a bilateral CSA, at each margin call date:
    - Compute the target CSB from current MTM, thresholds and IAs.
    - A transfer occurs only if the gap to the previous balance exceeds the MTA.
    - If the gap exceeds MTA: balance jumps to the target (ISDA delivery/return).
    - If the gap is within MTA: balance is sticky (no transfer this period).

    ``credit_support_balance`` returns the *stationary* target at each step
    (useful for threshold/IA analytics and zero-MTA cases).
    ``path_csb`` returns the *path-dependent* balance that properly accounts
    for MTA gating and margin call frequency. Use ``path_csb`` (via ``lagged_csb``)
    for exposure calculations.

    Parameters
    ----------
    csa : CSATerms
        CSA terms governing this netting set.
    """

    def __init__(self, csa: CSATerms) -> None:
        self.csa = csa

    def credit_support_balance(self, net_mtm: np.ndarray) -> np.ndarray:
        """Stationary target CSB at each time step (no MTA gating).

        CSB(t) = max(V(t) - TH_c, 0)        [VM received from cp]
               - max(-V(t) - TH_p, 0)       [VM posted by us]
               + IA_counterparty - IA_party  [Independent Amounts]

        This is the *unconstrained* target — useful for threshold/IA analytics
        and as a benchmark. When MTA = 0 this equals ``path_csb``. Use
        ``path_csb`` for exposure calculations when MTA > 0.

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        th_c = self.csa.threshold_counterparty
        th_p = self.csa.threshold_party
        ia_net = self.csa.ia_counterparty - self.csa.ia_party

        vm_received = np.maximum(net_mtm - th_c, 0.0)
        vm_posted = np.maximum(-net_mtm - th_p, 0.0)
        csb = vm_received - vm_posted + ia_net

        if self.csa.rounding_nearest > 0:
            csb = self._round_conservative(csb, net_mtm)
        return csb

    def path_csb(self, net_mtm: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """Path-dependent Credit Support Balance with MTA gating.

        At each margin call date the ISDA CSA rule is:
        - Delivery Amount (we receive): if target - CSB_prev >= MTA_party,
          transfer target - CSB_prev and set CSB = target.
        - Return Amount (we post back): if CSB_prev - target >= MTA_counterparty,
          transfer CSB_prev - target and set CSB = target.
        - Otherwise: no transfer, CSB stays at CSB_prev.

        Steps that fall between margin call dates (determined by
        ``csa.margin_call_frequency``) carry the previous balance forward.

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)
        time_grid : np.ndarray, shape (T,)

        Returns
        -------
        np.ndarray, shape (n_paths, T)
            Path-dependent CSB on every simulation path at every time step.
        """
        n_paths, T = net_mtm.shape
        mta_p = self.csa.mta_party
        mta_c = self.csa.mta_counterparty
        mcf = self.csa.margin_call_frequency

        # Stationary target at every step (n_paths, T)
        target = self.credit_support_balance(net_mtm)

        # Determine which time steps trigger a margin call.
        # A call fires at time t if at least mcf years have elapsed since
        # the previous call, so on a fine grid every step qualifies when mcf <= dt.
        call_mask = np.zeros(T, dtype=bool)
        last_call_t = -np.inf
        for i, t in enumerate(time_grid):
            if t - last_call_t >= mcf - 1e-10:
                call_mask[i] = True
                last_call_t = t

        return _path_csb_nb(target, call_mask, mta_p, mta_c, float(self.csa.rounding_nearest))

    def vm_call(self, net_mtm: np.ndarray) -> np.ndarray:
        """Marginal VM call from our perspective (positive = we call from cp).

        Returns the gross delivery/return amount at each step before MTA
        filtering, based on the stationary target. Useful for marginal analytics.

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        th_c = self.csa.threshold_counterparty
        th_p = self.csa.threshold_party

        call = np.maximum(net_mtm - th_c, 0.0) - np.maximum(-net_mtm - th_p, 0.0)

        if self.csa.mta_party > 0 or self.csa.mta_counterparty > 0:
            call = self._apply_mta(call)

        return call

    def lagged_csb(
        self,
        net_mtm: np.ndarray,
        time_grid: np.ndarray,
        lag: float | None = None,
    ) -> np.ndarray:
        """Path-dependent CSB lagged by MPOR to reflect the last-good-margin state.

        Under a default scenario, the last successful margin call was at t - MPOR.
        Exposure at time t is therefore:

            E(t) = max(V(t) - CSB_path(t - MPOR), 0)

        Computes the path-dependent ``path_csb`` and interpolates it backward
        by ``mpor`` on the time axis. For t < MPOR, clamped to t = 0.

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)
        time_grid : np.ndarray, shape (T,)
        lag : float | None
            Override MPOR (years). Defaults to ``csa.mpor``.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        lag = lag if lag is not None else self.csa.mpor
        csb_full = self.path_csb(net_mtm, time_grid)           # path-dependent (n_paths, T)
        lagged_times = np.clip(time_grid - lag, time_grid[0], time_grid[-1])

        lagged = np.empty_like(csb_full)
        for p in range(csb_full.shape[0]):
            lagged[p] = np.interp(lagged_times, time_grid, csb_full[p])
        return lagged

    def uncollateralised_exposure(self, net_mtm: np.ndarray) -> np.ndarray:
        """Positive exposure with no collateral: max(V(t), 0)."""
        return np.maximum(net_mtm, 0.0)

    def collateralised_exposure(
        self,
        net_mtm: np.ndarray,
        time_grid: np.ndarray,
        im_balance: np.ndarray | None = None,
    ) -> np.ndarray:
        """Path-wise positive exposure after VM (with MPOR lag) and IM.

        E_coll(t) = max(V(t) - CSB(t - MPOR) - IM(t), 0)

        Parameters
        ----------
        net_mtm : np.ndarray, shape (n_paths, T)
        time_grid : np.ndarray, shape (T,)
        im_balance : np.ndarray | None, shape (n_paths, T) or broadcastable
            Initial margin held (received from counterparty). If None, no IM.

        Returns
        -------
        np.ndarray, shape (n_paths, T)
        """
        csb = self.lagged_csb(net_mtm, time_grid)
        net = net_mtm - csb
        if im_balance is not None:
            net = net - np.asarray(im_balance)
        return np.maximum(net, 0.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_mta(self, call: np.ndarray) -> np.ndarray:
        """Zero out calls smaller than MTA (both directions)."""
        mta_recv = self.csa.mta_party        # minimum we will call
        mta_send = self.csa.mta_counterparty  # minimum cp will call
        # Positive call (we receive): zero if below mta_recv
        # Negative call (we post): zero if abs below mta_send
        result = call.copy()
        result[(call > 0) & (call < mta_recv)] = 0.0
        result[(call < 0) & (-call < mta_send)] = 0.0
        return result

    def _round_conservative(self, csb: np.ndarray, net_mtm: np.ndarray) -> np.ndarray:
        """Round CSB conservatively per ISDA 2016 VM CSA para 3(b).

        - Delivery amounts (we receive) round DOWN to nearest unit.
        - Return amounts (we return to cp) round UP to nearest unit.
        This is conservative (we keep slightly less than the target CSB).
        """
        unit = self.csa.rounding_nearest
        rounded = np.where(
            csb >= 0,
            np.floor(csb / unit) * unit,   # receiving: round down
            np.ceil(csb / unit) * unit,    # posting: round up (less negative)
        )
        return rounded
