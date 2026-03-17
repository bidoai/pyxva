"""Pure metric functions for backtesting exposure forecasts.

All functions are stateless and model-agnostic: they operate on
plain NumPy arrays representing a forecast MTM distribution
(n_paths × T) and a realised MTM time series (T,).
"""
from __future__ import annotations

import numpy as np


def pfe_profile(mtm_paths: np.ndarray, confidence: float) -> np.ndarray:
    """Compute the PFE forecast at each time step.

    PFE(t) = quantile(max(V(t), 0), confidence)  across paths.

    Parameters
    ----------
    mtm_paths : np.ndarray, shape (n_paths, T)
    confidence : float
        E.g. 0.95 for 95th-percentile PFE.

    Returns
    -------
    np.ndarray, shape (T,)
    """
    return np.quantile(np.maximum(mtm_paths, 0.0), confidence, axis=0)


def ee_profile(mtm_paths: np.ndarray) -> np.ndarray:
    """Expected positive exposure at each time step.

    EE(t) = E[max(V(t), 0)]  across paths.

    Parameters
    ----------
    mtm_paths : np.ndarray, shape (n_paths, T)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    return np.maximum(mtm_paths, 0.0).mean(axis=0)


def exceedance_series(pfe: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Boolean array: True where realised MTM exceeds the PFE forecast.

    An exceedance at time t means the actual exposure was worse than
    the model predicted at the chosen confidence level.

    Parameters
    ----------
    pfe : np.ndarray, shape (T,)
        PFE forecast from :func:`pfe_profile`.
    realized : np.ndarray, shape (T,)
        Observed MTM values.

    Returns
    -------
    np.ndarray of bool, shape (T,)
    """
    return realized > pfe


def basel_zone(n_exceptions: int, n_observations: int) -> str:
    """Basel traffic-light zone for a PFE / VaR backtest.

    The standard Basel framework defines zones for a 250-observation
    window calibrated at 99% confidence. Here we scale the observed
    exception count to a 250-observation equivalent, which allows the
    test to be applied over arbitrary-length horizons while preserving
    the regulatory thresholds.

    Zones (scaled to 250 obs):
      Green  — 0–4   exceptions  (model acceptable)
      Amber  — 5–9   exceptions  (investigate)
      Red    — 10+   exceptions  (model inadequate)

    Parameters
    ----------
    n_exceptions : int
        Observed exceedance count.
    n_observations : int
        Total number of time steps compared.

    Returns
    -------
    str  — "Green", "Amber", or "Red"
    """
    if n_observations == 0:
        return "Green"
    scaled = int(round(n_exceptions * 250 / n_observations))
    if scaled <= 4:
        return "Green"
    elif scaled <= 9:
        return "Amber"
    else:
        return "Red"


def ee_accuracy(ee: np.ndarray, realized: np.ndarray) -> dict:
    """Measure accuracy of the EE forecast against realised MTM.

    Compares EE(t) = E[max(V(t), 0)] (cross-sectional mean of positive
    exposures) against the realised mark-to-market at each time step.

    Metrics
    -------
    rmse : float
        Root mean squared error — overall forecast accuracy.
    bias : float
        Mean signed error EE(t) − realised(t). Positive means the model
        systematically over-predicts exposure (conservative); negative
        means under-prediction (dangerous).
    mae : float
        Mean absolute error — robust to outliers.

    Parameters
    ----------
    ee : np.ndarray, shape (T,)
        EE forecast from :func:`ee_profile`.
    realized : np.ndarray, shape (T,)
        Observed MTM values.

    Returns
    -------
    dict with keys: "rmse", "bias", "mae"
    """
    diff = ee - realized
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
    }
