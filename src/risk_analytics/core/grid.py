from __future__ import annotations

import numpy as np


class TimeGrid:
    """Utility for building simulation time grids."""

    @staticmethod
    def uniform(maturity: float, n_steps: int) -> np.ndarray:
        """Evenly-spaced time grid from 0 to maturity."""
        return np.linspace(0.0, maturity, n_steps + 1)

    @staticmethod
    def from_dates(dates: list[float]) -> np.ndarray:
        """Build a time grid from a list of year fractions (must start at 0)."""
        grid = np.asarray(dates, dtype=float)
        if grid[0] != 0.0:
            grid = np.concatenate([[0.0], grid])
        return np.sort(grid)

    @staticmethod
    def dt(time_grid: np.ndarray) -> np.ndarray:
        """Time steps between grid points, shape (T-1,)."""
        return np.diff(time_grid)


class SparseTimeGrid:
    """
    Sparse simulation grid designed to balance accuracy and memory efficiency.

    Default standard grid:
    - Daily steps for the first 2 weeks  (~10 business days, step = 1/252)
    - Weekly steps for the remainder of year 1  (step = 1/52)
    - Monthly steps from year 1 to maturity  (step = 1/12)

    All grids always include t=0 and t=maturity.
    """

    _DAILY_STEP = 1.0 / 252
    _WEEKLY_STEP = 1.0 / 52
    _MONTHLY_STEP = 1.0 / 12

    _DAILY_HORIZON = 10.0 / 252   # ~2 calendar weeks
    _WEEKLY_HORIZON = 1.0         # 1 year

    @staticmethod
    def standard(maturity: float) -> np.ndarray:
        """Build the standard sparse grid for a given maturity (in years)."""
        points = [0.0]

        # Daily phase: 1/252 steps up to ~2 weeks
        t = SparseTimeGrid._DAILY_STEP
        while t < min(SparseTimeGrid._DAILY_HORIZON, maturity) - 1e-9:
            points.append(t)
            t += SparseTimeGrid._DAILY_STEP

        # Weekly phase: 1/52 steps up to 1 year
        if maturity > SparseTimeGrid._DAILY_HORIZON:
            # Start from where daily left off, snapped to nearest weekly boundary
            t = SparseTimeGrid._DAILY_HORIZON
            while t < min(SparseTimeGrid._WEEKLY_HORIZON, maturity) - 1e-9:
                points.append(t)
                t += SparseTimeGrid._WEEKLY_STEP

        # Monthly phase: 1/12 steps to maturity
        if maturity > SparseTimeGrid._WEEKLY_HORIZON:
            t = SparseTimeGrid._WEEKLY_HORIZON
            while t < maturity - 1e-9:
                points.append(t)
                t += SparseTimeGrid._MONTHLY_STEP

        points.append(maturity)
        return np.unique(np.array(points))

    @staticmethod
    def custom(anchor_points: list) -> np.ndarray:
        """Build a grid from user-supplied anchor points. Always includes t=0."""
        pts = list(anchor_points)
        if 0.0 not in pts:
            pts = [0.0] + pts
        return np.unique(np.array(pts, dtype=float))

    @staticmethod
    def merge_cashflows(
        grid: np.ndarray,
        cashflow_times: list,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Merge cashflow/jump dates into an existing grid as hard nodes.

        Cashflow dates that are within `tol` of an existing node are snapped
        to that node (deduplication). Dates outside the grid range are ignored
        with a warning.
        """
        if not cashflow_times:
            return grid

        import logging
        logger = logging.getLogger(__name__)

        t_min, t_max = grid[0], grid[-1]
        extra = []
        for t in cashflow_times:
            if t < t_min - tol or t > t_max + tol:
                logger.debug("Cashflow time %.6f is outside grid range [%.6f, %.6f]; skipping.", t, t_min, t_max)
                continue
            # Only add if not already within tol of an existing node
            if np.min(np.abs(grid - t)) > tol:
                extra.append(t)

        if not extra:
            return grid

        merged = np.concatenate([grid, np.array(extra, dtype=float)])
        return np.unique(merged)

    @staticmethod
    def dt(grid: np.ndarray) -> np.ndarray:
        """Time step sizes between consecutive grid points."""
        return np.diff(grid)
