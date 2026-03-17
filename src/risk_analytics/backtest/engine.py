"""BacktestEngine — model-agnostic exposure backtest runner.

Usage
-----
::

    from risk_analytics.backtest import BacktestEngine

    engine = BacktestEngine(confidence=0.95)
    result = engine.run(mtm_paths, realized, time_grid)
    print(result.summary())

Walk-forward pattern
--------------------
For a rolling backtest, call :meth:`BacktestEngine.run` once per window,
collecting the returned :class:`BacktestResult` objects:

::

    results = []
    for t, paths_t, realized_t, grid_t in walk_forward_windows(...):
        results.append(engine.run(paths_t, realized_t, grid_t))
"""
from __future__ import annotations

import logging

import numpy as np

from .metrics import ee_accuracy, ee_profile, exceedance_series, pfe_profile, basel_zone
from .result import BacktestResult

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Model-agnostic exposure forecast backtester.

    Accepts a forecast MTM distribution and a realised MTM series,
    then computes PFE exceedances, the Basel traffic-light zone, and
    EE forecast accuracy metrics.

    Parameters
    ----------
    confidence : float
        PFE confidence level (default 0.95 → 95th-percentile PFE).
    """

    def __init__(self, confidence: float = 0.95) -> None:
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1); got {confidence}")
        self.confidence = confidence

    def run(
        self,
        mtm_paths: np.ndarray,
        realized: np.ndarray,
        time_grid: np.ndarray,
    ) -> BacktestResult:
        """Run a single backtest comparison.

        Parameters
        ----------
        mtm_paths : np.ndarray, shape (n_paths, T)
            Forecast MTM distribution at each time step (from Monte Carlo
            simulation). Can contain negative values (MTM, not exposure).
        realized : np.ndarray, shape (T,)
            Realised MTM values — one observed path used as ground truth.
            May be synthetic (e.g. a single Monte Carlo path held out).
        time_grid : np.ndarray, shape (T,)
            Time points in years corresponding to the columns of mtm_paths
            and entries of realized.

        Returns
        -------
        BacktestResult
            Full backtest output including exceedance series, Basel zone,
            and EE accuracy metrics.

        Raises
        ------
        ValueError
            If array shapes are inconsistent.
        """
        mtm_paths = np.asarray(mtm_paths, dtype=float)
        realized = np.asarray(realized, dtype=float)
        time_grid = np.asarray(time_grid, dtype=float)

        n_paths, T = mtm_paths.shape
        if realized.shape != (T,):
            raise ValueError(
                f"realized must have shape ({T},) to match mtm_paths columns; "
                f"got {realized.shape}"
            )
        if time_grid.shape != (T,):
            raise ValueError(
                f"time_grid must have shape ({T},); got {time_grid.shape}"
            )

        logger.info(
            "BacktestEngine.run: n_paths=%d  T=%d  confidence=%.2f",
            n_paths, T, self.confidence,
        )

        pfe = pfe_profile(mtm_paths, self.confidence)
        ee = ee_profile(mtm_paths)
        exc = exceedance_series(pfe, realized)

        n_exc = int(exc.sum())
        exc_rate = n_exc / T
        expected_rate = 1.0 - self.confidence
        zone = basel_zone(n_exc, T)
        acc = ee_accuracy(ee, realized)

        logger.info(
            "Backtest complete: exceptions=%d/%d (%.1f%% vs %.1f%% expected)  "
            "zone=%s  EE_RMSE=%.4g  EE_bias=%.4g",
            n_exc, T,
            exc_rate * 100, expected_rate * 100,
            zone, acc["rmse"], acc["bias"],
        )

        return BacktestResult(
            time_grid=time_grid,
            realized=realized,
            pfe_profile=pfe,
            ee_profile=ee,
            exceedances=exc,
            confidence=self.confidence,
            n_exceptions=n_exc,
            exception_rate=exc_rate,
            expected_exception_rate=expected_rate,
            basel_zone=zone,
            ee_rmse=acc["rmse"],
            ee_bias=acc["bias"],
            ee_mae=acc["mae"],
        )
