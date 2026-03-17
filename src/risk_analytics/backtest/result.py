"""BacktestResult dataclass — holds all outputs from a single backtest run."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestResult:
    """Full output of a single :class:`BacktestEngine` run.

    Attributes
    ----------
    time_grid : np.ndarray, shape (T,)
        Simulation time points in years.
    realized : np.ndarray, shape (T,)
        The realised MTM series used as ground truth.
    pfe_profile : np.ndarray, shape (T,)
        PFE forecast at each time step.
    ee_profile : np.ndarray, shape (T,)
        Expected positive exposure at each time step.
    exceedances : np.ndarray of bool, shape (T,)
        True where realised MTM exceeded the PFE forecast.
    confidence : float
        PFE confidence level used (e.g. 0.95).
    n_exceptions : int
        Total number of PFE exceedances.
    exception_rate : float
        Observed exceedance rate (n_exceptions / T).
    expected_exception_rate : float
        Theoretical exception rate under a correct model (1 - confidence).
    basel_zone : str
        "Green", "Amber", or "Red" — Basel traffic-light classification.
    ee_rmse : float
        RMSE of EE forecast vs realised MTM.
    ee_bias : float
        Mean signed error of EE forecast (positive = over-prediction).
    ee_mae : float
        Mean absolute error of EE forecast.
    """

    time_grid: np.ndarray
    realized: np.ndarray
    pfe_profile: np.ndarray
    ee_profile: np.ndarray
    exceedances: np.ndarray
    confidence: float
    n_exceptions: int
    exception_rate: float
    expected_exception_rate: float
    basel_zone: str
    ee_rmse: float
    ee_bias: float
    ee_mae: float

    def summary(self) -> dict:
        """Return a flat dict of scalar summary statistics.

        Suitable for printing, logging, or assembling into a DataFrame
        when running many backtest windows.

        Returns
        -------
        dict with keys:
            n_observations, n_exceptions, exception_rate,
            expected_exception_rate, excess_exception_rate,
            basel_zone, ee_rmse, ee_bias, ee_mae
        """
        return {
            "n_observations": int(len(self.time_grid)),
            "n_exceptions": self.n_exceptions,
            "exception_rate": self.exception_rate,
            "expected_exception_rate": self.expected_exception_rate,
            "excess_exception_rate": self.exception_rate - self.expected_exception_rate,
            "basel_zone": self.basel_zone,
            "ee_rmse": self.ee_rmse,
            "ee_bias": self.ee_bias,
            "ee_mae": self.ee_mae,
        }
