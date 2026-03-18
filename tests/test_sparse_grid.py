"""Tests for SparseTimeGrid."""
from __future__ import annotations

import numpy as np
import pytest

from risk_analytics.core.grid import SparseTimeGrid


class TestStandard:
    def test_standard_starts_at_zero(self):
        grid = SparseTimeGrid.standard(0.5)
        assert grid[0] == pytest.approx(0.0)

    def test_standard_ends_at_maturity(self):
        for maturity in [0.1, 0.5, 1.0, 5.0, 10.0]:
            grid = SparseTimeGrid.standard(maturity)
            assert grid[-1] == pytest.approx(maturity)

    def test_standard_is_monotone(self):
        grid = SparseTimeGrid.standard(5.0)
        assert np.all(np.diff(grid) > 0)

    def test_standard_half_year_has_daily_steps_at_start(self):
        grid = SparseTimeGrid.standard(0.5)
        # First step should be approximately 1/252
        first_step = grid[1] - grid[0]
        assert first_step == pytest.approx(1.0 / 252, rel=1e-5)

    def test_standard_5yr_has_three_phases(self):
        grid = SparseTimeGrid.standard(5.0)
        # Daily phase: steps ~ 1/252
        daily_end = 10.0 / 252
        daily_mask = grid[grid < daily_end]
        if len(daily_mask) > 1:
            first_steps = np.diff(daily_mask[:5])
            assert np.allclose(first_steps, 1.0 / 252, rtol=1e-4)

        # Weekly phase: steps ~ 1/52 between daily_end and 1yr
        # The last step may be smaller (residual gap to the 1yr anchor)
        weekly_pts = grid[(grid >= daily_end) & (grid <= 1.0)]
        if len(weekly_pts) > 2:
            # All interior steps should be 1/52; the last may be a residual
            interior_steps = np.diff(weekly_pts[:-1])
            assert np.allclose(interior_steps, 1.0 / 52, rtol=1e-4)

        # Monthly phase: steps ~ 1/12 beyond 1yr
        # The last step may be smaller (residual gap to maturity)
        monthly_pts = grid[grid >= 1.0]
        if len(monthly_pts) > 2:
            interior_steps = np.diff(monthly_pts[:-1])
            assert np.allclose(interior_steps, 1.0 / 12, rtol=1e-4)

    def test_standard_short_maturity_no_weekly_phase(self):
        # For maturity < DAILY_HORIZON, only daily + maturity endpoint
        grid = SparseTimeGrid.standard(0.03)
        # Should contain t=0 and t=maturity at minimum
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(0.03)


class TestCustom:
    def test_custom_includes_t0(self):
        grid = SparseTimeGrid.custom([1.0, 2.0, 5.0])
        assert grid[0] == pytest.approx(0.0)

    def test_custom_with_explicit_t0(self):
        grid = SparseTimeGrid.custom([0.0, 1.0, 2.0])
        assert grid[0] == pytest.approx(0.0)
        assert len(grid) == 3  # no duplicate 0.0

    def test_custom_is_sorted(self):
        grid = SparseTimeGrid.custom([5.0, 1.0, 3.0])
        assert np.all(np.diff(grid) > 0)

    def test_custom_preserves_all_points(self):
        pts = [1.0, 2.0, 5.0]
        grid = SparseTimeGrid.custom(pts)
        for p in pts:
            assert p in grid


class TestMergeCashflows:
    def test_merge_adds_new_times(self):
        grid = np.array([0.0, 0.5, 1.0, 2.0])
        cf_times = [0.25, 0.75, 1.5]
        merged = SparseTimeGrid.merge_cashflows(grid, cf_times)
        for t in cf_times:
            assert np.any(np.abs(merged - t) < 1e-9)

    def test_merge_deduplicates_near_existing_nodes(self):
        grid = np.array([0.0, 0.5, 1.0, 2.0])
        # 0.5000001 is within default tol=1e-6 of 0.5
        cf_times = [0.5 + 5e-7]
        merged = SparseTimeGrid.merge_cashflows(grid, cf_times)
        # Length should be same as original — nothing added
        assert len(merged) == len(grid)

    def test_merge_ignores_out_of_range(self):
        grid = np.array([0.0, 0.5, 1.0])
        cf_times = [2.0, -0.1]  # both outside grid
        merged = SparseTimeGrid.merge_cashflows(grid, cf_times)
        assert len(merged) == len(grid)

    def test_merge_empty_cashflows_returns_same_grid(self):
        grid = np.array([0.0, 1.0, 2.0])
        merged = SparseTimeGrid.merge_cashflows(grid, [])
        np.testing.assert_array_equal(merged, grid)

    def test_merge_result_is_sorted(self):
        grid = np.array([0.0, 1.0, 3.0])
        cf_times = [2.0, 0.5]
        merged = SparseTimeGrid.merge_cashflows(grid, cf_times)
        assert np.all(np.diff(merged) > 0)

    def test_dt_helper(self):
        grid = np.array([0.0, 0.5, 1.0, 2.0])
        dt = SparseTimeGrid.dt(grid)
        np.testing.assert_allclose(dt, [0.5, 0.5, 1.0])
