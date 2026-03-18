"""Tests for day-count conventions, calendars, and schedules."""
from __future__ import annotations

import datetime
from datetime import date

import numpy as np
import pytest

from pyxva.core import (
    BusinessDayConvention,
    DayCountConvention,
    Frequency,
    NullCalendar,
    Schedule,
    TARGET,
    USCalendar,
)
from pyxva.core.conventions import _easter


# ---------------------------------------------------------------------------
# Day-count conventions
# ---------------------------------------------------------------------------

class TestDayCountConvention:
    def test_act360_simple(self):
        d1 = date(2024, 1, 1)
        d2 = date(2024, 7, 1)  # 182 days
        yf = DayCountConvention.ACT_360.year_fraction(d1, d2)
        assert yf == pytest.approx(182 / 360)

    def test_act365_simple(self):
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)  # 366 days (2024 is leap)
        yf = DayCountConvention.ACT_365.year_fraction(d1, d2)
        assert yf == pytest.approx(366 / 365)

    def test_act_act_isda_same_year(self):
        d1 = date(2024, 1, 1)
        d2 = date(2024, 7, 1)  # 182 days in a 366-day year
        yf = DayCountConvention.ACT_ACT_ISDA.year_fraction(d1, d2)
        assert yf == pytest.approx(182 / 366)

    def test_act_act_isda_spans_year(self):
        d1 = date(2024, 7, 1)   # leap year; 184 days remaining
        d2 = date(2025, 7, 1)   # 181 days into 2025
        yf = DayCountConvention.ACT_ACT_ISDA.year_fraction(d1, d2)
        expected = 184 / 366 + 181 / 365
        assert yf == pytest.approx(expected)

    def test_thirty_360_full_year(self):
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)
        yf = DayCountConvention.THIRTY_360.year_fraction(d1, d2)
        assert yf == pytest.approx(1.0)

    def test_thirty_360_quarter(self):
        d1 = date(2024, 1, 1)
        d2 = date(2024, 4, 1)
        yf = DayCountConvention.THIRTY_360.year_fraction(d1, d2)
        assert yf == pytest.approx(0.25)

    def test_thirty_e_360_end_of_month(self):
        """30E/360 always caps D2 at 30."""
        d1 = date(2024, 1, 31)
        d2 = date(2024, 3, 31)
        yf = DayCountConvention.THIRTY_E_360.year_fraction(d1, d2)
        # D1=30, D2=30 → (0*360 + 2*30 + 0)/360 = 60/360 = 1/6
        assert yf == pytest.approx(60 / 360)

    def test_negative_year_fraction(self):
        d1 = date(2024, 6, 1)
        d2 = date(2024, 1, 1)
        yf = DayCountConvention.ACT_360.year_fraction(d1, d2)
        assert yf < 0
        assert yf == pytest.approx(-DayCountConvention.ACT_360.year_fraction(d2, d1))

    def test_zero_days(self):
        d = date(2024, 3, 15)
        assert DayCountConvention.ACT_360.year_fraction(d, d) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------

class TestNullCalendar:
    def setup_method(self):
        self.cal = NullCalendar()

    def test_weekday_is_business_day(self):
        assert self.cal.is_business_day(date(2024, 3, 13))  # Wednesday

    def test_saturday_not_business_day(self):
        assert not self.cal.is_business_day(date(2024, 3, 16))  # Saturday

    def test_sunday_not_business_day(self):
        assert not self.cal.is_business_day(date(2024, 3, 17))  # Sunday

    def test_no_holidays(self):
        assert not self.cal.is_holiday(date(2024, 1, 1))

    def test_adjust_following(self):
        sat = date(2024, 3, 16)
        adjusted = self.cal.adjust(sat, BusinessDayConvention.FOLLOWING)
        assert adjusted == date(2024, 3, 18)  # Monday

    def test_adjust_preceding(self):
        sat = date(2024, 3, 16)
        adjusted = self.cal.adjust(sat, BusinessDayConvention.PRECEDING)
        assert adjusted == date(2024, 3, 15)  # Friday

    def test_adjust_modified_following_crosses_month(self):
        # Saturday April 29 2023 → Modified Following → Thursday April 27 2023
        # (May 1 would cross month boundary, so go back to Friday April 28)
        sat = date(2023, 4, 29)
        adjusted = self.cal.adjust(sat, BusinessDayConvention.MODIFIED_FOLLOWING)
        assert adjusted.month == 4   # Must stay in April

    def test_adjust_unadjusted(self):
        sat = date(2024, 3, 16)
        assert self.cal.adjust(sat, BusinessDayConvention.UNADJUSTED) == sat


class TestEaster:
    @pytest.mark.parametrize("year,expected", [
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2023, date(2023, 4,  9)),
        (2000, date(2000, 4, 23)),
    ])
    def test_known_easter_dates(self, year, expected):
        assert _easter(year) == expected


class TestTARGET:
    def setup_method(self):
        self.cal = TARGET()

    def test_new_years_day(self):
        assert self.cal.is_holiday(date(2024, 1, 1))

    def test_labour_day(self):
        assert self.cal.is_holiday(date(2024, 5, 1))

    def test_christmas(self):
        assert self.cal.is_holiday(date(2024, 12, 25))
        assert self.cal.is_holiday(date(2024, 12, 26))

    def test_good_friday_2024(self):
        assert self.cal.is_holiday(date(2024, 3, 29))

    def test_easter_monday_2024(self):
        assert self.cal.is_holiday(date(2024, 4, 1))

    def test_normal_weekday_not_holiday(self):
        assert not self.cal.is_holiday(date(2024, 6, 5))

    def test_is_not_business_day_on_holiday(self):
        assert not self.cal.is_business_day(date(2024, 1, 1))


class TestUSCalendar:
    def setup_method(self):
        self.cal = USCalendar()

    def test_thanksgiving_2024(self):
        # 4th Thursday of November 2024
        assert self.cal.is_holiday(date(2024, 11, 28))

    def test_independence_day_2024(self):
        assert self.cal.is_holiday(date(2024, 7, 4))

    def test_mlk_day_2024(self):
        # 3rd Monday of January 2024 = Jan 15
        assert self.cal.is_holiday(date(2024, 1, 15))

    def test_juneteenth_since_2021(self):
        # June 19 2021 was a Saturday → observed Friday June 18
        assert self.cal.is_holiday(date(2021, 6, 18))
        # June 19 2023 is a Monday → observed on the day itself
        assert self.cal.is_holiday(date(2023, 6, 19))

    def test_normal_weekday_not_holiday(self):
        assert not self.cal.is_holiday(date(2024, 6, 5))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_quarterly_schedule_length(self):
        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2026, 1, 1),
            Frequency.QUARTERLY,
        )
        assert len(sched) == 8  # 2 years × 4 quarters

    def test_semi_annual_schedule_length(self):
        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2029, 1, 1),
            Frequency.SEMI_ANNUAL,
        )
        assert len(sched) == 10  # 5 years × 2

    def test_payment_times_increasing(self):
        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2025, 1, 1),
            Frequency.QUARTERLY,
        )
        assert np.all(np.diff(sched.payment_times) > 0)

    def test_payment_times_last_near_maturity(self):
        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2026, 1, 1),
            Frequency.ANNUAL, day_count=DayCountConvention.ACT_365,
        )
        # Last payment ~2 years from start
        assert abs(sched.payment_times[-1] - 2.0) < 0.01

    def test_day_count_fractions_sum_to_total(self):
        """Sum of accrual periods should approximately equal total maturity."""
        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2026, 1, 1),
            Frequency.QUARTERLY,
            day_count=DayCountConvention.ACT_365,
        )
        total = DayCountConvention.ACT_365.year_fraction(date(2024, 1, 1), date(2026, 1, 1))
        assert abs(sched.day_count_fractions.sum() - total) < 1e-10

    def test_business_day_adjustment_skips_weekend(self):
        # Jan 1 2024 is Monday; quarterly dates should avoid weekends
        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2025, 1, 1),
            Frequency.QUARTERLY,
            calendar=NullCalendar(),
            bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        )
        for d in sched.payment_dates:
            assert d.weekday() < 5, f"{d} is a weekend"

    def test_target_calendar_adjusts_holiday(self):
        """Payment falling on TARGET holiday should be moved."""
        # May 1 (Labour Day) is a TARGET holiday
        sched = Schedule.from_dates(
            date(2024, 2, 1), date(2024, 8, 1),
            Frequency.QUARTERLY,
            calendar=TARGET(),
            bdc=BusinessDayConvention.MODIFIED_FOLLOWING,
        )
        for d in sched.payment_dates:
            assert TARGET().is_business_day(d), f"{d} is not a TARGET business day"

    def test_integer_frequency(self):
        """Integer frequency (4 = quarterly) should work like Frequency.QUARTERLY."""
        sched_enum = Schedule.from_dates(
            date(2024, 1, 1), date(2025, 1, 1), Frequency.QUARTERLY,
        )
        sched_int = Schedule.from_dates(
            date(2024, 1, 1), date(2025, 1, 1), 4,
        )
        assert np.allclose(sched_enum.payment_times, sched_int.payment_times)

    def test_repr(self):
        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2025, 1, 1), Frequency.ANNUAL,
        )
        assert "Schedule" in repr(sched)


# ---------------------------------------------------------------------------
# Integration: Schedule-driven swap vs plain swap
# ---------------------------------------------------------------------------

class TestScheduleDrivenPricers:
    def test_swap_with_schedule_has_nonuniform_deltas(self):
        from pyxva.pricing import InterestRateSwap

        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2026, 1, 1),
            Frequency.QUARTERLY,
            calendar=TARGET(),
            day_count=DayCountConvention.ACT_360,
        )
        swap = InterestRateSwap(fixed_rate=0.04, schedule=sched)
        # ACT/360 quarterly accruals are not all exactly 0.25
        assert not np.allclose(swap.deltas, swap.deltas[0])

    def test_swap_maturity_from_schedule(self):
        from pyxva.pricing import InterestRateSwap

        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2029, 1, 1),
            Frequency.ANNUAL, day_count=DayCountConvention.ACT_365,
        )
        swap = InterestRateSwap(fixed_rate=0.05, schedule=sched)
        assert abs(swap.maturity - 5.0) < 0.01

    def test_bond_coupon_amounts_from_schedule(self):
        from pyxva.pricing import FixedRateBond

        sched = Schedule.from_dates(
            date(2024, 1, 1), date(2026, 1, 1),
            Frequency.SEMI_ANNUAL, day_count=DayCountConvention.THIRTY_360,
        )
        bond = FixedRateBond(coupon_rate=0.05, face_value=1000.0, schedule=sched)
        # 30/360 semi-annual → each δ ≈ 0.5 → coupon ≈ 25
        assert np.allclose(bond.coupon_amounts, 25.0, atol=0.5)
