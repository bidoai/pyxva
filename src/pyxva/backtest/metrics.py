"""Pure metric functions for backtesting exposure forecasts.

All functions are stateless and model-agnostic: they operate on
plain NumPy arrays representing a forecast MTM distribution
(n_paths × T) and a realised MTM time series (T,).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


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


def kupiec_pof(n_exceptions: int, n_observations: int, confidence: float) -> dict:
    """Kupiec Proportion of Failures (POF) likelihood ratio test.

    Tests H₀: the true exception probability equals the expected rate
    ``p₀ = 1 - confidence``. Under H₀ the LR statistic is asymptotically
    χ²(1), giving a formal p-value for the exceedance frequency.

    A low p-value (e.g. < 0.05) means the observed exception rate is
    statistically inconsistent with a correct model at the chosen
    confidence level.

    .. note::
        Assumes exceedances are i.i.d. across time steps. MTM paths are
        serially correlated, so the effective sample size is smaller than
        ``n_observations`` and p-values will tend to be anti-conservative
        (appear more significant than they truly are). Use the result as
        a directional signal rather than a strict hypothesis test.

    Parameters
    ----------
    n_exceptions : int
        Observed number of PFE exceedances.
    n_observations : int
        Total number of time steps.
    confidence : float
        PFE confidence level (e.g. 0.95 → p₀ = 0.05).

    Returns
    -------
    dict with keys:
        ``lr_stat`` — Kupiec LR test statistic (non-negative float).
        ``p_value`` — Right-tail p-value from χ²(1) distribution.
    """
    p0 = 1.0 - confidence
    k = n_exceptions
    n = n_observations

    if n == 0:
        return {"lr_stat": float("nan"), "p_value": float("nan")}

    # Degenerate cases where the MLE log-likelihood term is undefined
    if k == 0:
        # log(k/n) → -inf; LR = -2 * [k*log(p0) + (n-k)*log(1-p0) - 0 - 0]
        # = -2 * n * log(1 - p0)
        lr = -2.0 * n * np.log(1.0 - p0)
    elif k == n:
        # log(1 - k/n) → -inf; LR = -2 * n * log(p0)
        lr = -2.0 * n * np.log(p0)
    else:
        p_hat = k / n
        lr = -2.0 * (
            k * np.log(p0 / p_hat)
            + (n - k) * np.log((1.0 - p0) / (1.0 - p_hat))
        )

    lr = max(lr, 0.0)   # clip floating-point rounding artefacts (LR is theoretically ≥ 0)
    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))
    return {"lr_stat": float(lr), "p_value": p_value}


def bias_ttest(ee: np.ndarray, realized: np.ndarray) -> dict:
    """Two-sided t-test for systematic bias in the EE forecast.

    Tests H₀: E[EE(t) − realized(t)] = 0. A significant result means
    the model has a statistically detectable over- or under-prediction
    of expected exposure across the time horizon.

    A significant negative t-statistic (p < 0.05, t < 0) indicates the
    model is *under-predicting* exposure — the more dangerous failure.

    .. note::
        Assumes residuals ``EE(t) − realized(t)`` are i.i.d. over time.
        Serial correlation in MTM paths inflates the apparent sample size,
        making p-values anti-conservative. Treat as a directional signal.

    Parameters
    ----------
    ee : np.ndarray, shape (T,)
        EE forecast from :func:`ee_profile`.
    realized : np.ndarray, shape (T,)
        Observed MTM values.

    Returns
    -------
    dict with keys:
        ``t_stat`` — t-statistic (positive = model over-predicts).
        ``p_value`` — Two-sided p-value from t(T−1) distribution.
    """
    residuals = ee - realized
    T = len(residuals)
    if T < 2:
        return {"t_stat": float("nan"), "p_value": float("nan")}
    if np.std(residuals) == 0.0:
        # All residuals identical: zero bias if they are all zero, otherwise infinite t.
        # Return t=0, p=1 when residuals are uniformly zero (no evidence of bias);
        # return t=nan otherwise (degenerate — constant non-zero offset).
        if residuals[0] == 0.0:
            return {"t_stat": 0.0, "p_value": 1.0}
        return {"t_stat": float("nan"), "p_value": float("nan")}
    t_stat, p_value = stats.ttest_1samp(residuals, popmean=0.0)
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


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
