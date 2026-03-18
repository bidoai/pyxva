"""Payment schedule generation with calendar and day-count support."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np

from .conventions import (
    BusinessDayConvention,
    Calendar,
    DayCountConvention,
    NullCalendar,
)


class Frequency(Enum):
    """Number of payment periods per year."""

    MONTHLY     = 12
    QUARTERLY   =  4
    SEMI_ANNUAL =  2
    ANNUAL      =  1

    @property
    def months(self) -> int:
        """Months between successive payment dates."""
        return 12 // self.value


@dataclass
class Schedule:
    """Fully adjusted payment schedule for a fixed-income instrument.

    Attributes
    ----------
    start_date : datetime.date
        Effective (start) date of the schedule.
    end_date : datetime.date
        Final maturity date (unadjusted).
    payment_dates : list[datetime.date]
        Business-day-adjusted payment dates.
    payment_times : np.ndarray, shape (n,)
        Year-fractions from ``start_date`` to each adjusted payment date,
        computed under ``day_count``.
    day_count_fractions : np.ndarray, shape (n,)
        Accrual period length (δᵢ) for each coupon: year-fraction between the
        start of the period (previous payment date or start_date) and the
        adjusted payment date. Used as the coupon multiplier for the fixed leg.
    day_count : DayCountConvention
        Convention used to compute both ``payment_times`` and ``day_count_fractions``.
    calendar : Calendar
        Calendar used for business-day adjustment.
    """

    start_date:           date
    end_date:             date
    payment_dates:        list[date]
    payment_times:        np.ndarray
    day_count_fractions:  np.ndarray
    day_count:            DayCountConvention
    calendar:             Calendar

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dates(
        cls,
        start: date,
        end: date,
        frequency: Frequency | int,
        *,
        calendar: Calendar | None = None,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        bdc: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    ) -> "Schedule":
        """Build an adjusted payment schedule from start/end dates.

        Parameters
        ----------
        start : datetime.date
            Effective (start) date. Not itself a payment date.
        end : datetime.date
            Final maturity / termination date.
        frequency : Frequency | int
            Payment frequency enum or integer number of payments per year
            (e.g. 4 = quarterly).
        calendar : Calendar | None
            Holiday calendar for business-day adjustment.
            Defaults to ``NullCalendar`` (weekends only).
        day_count : DayCountConvention
            Year-fraction convention.
        bdc : BusinessDayConvention
            Business-day adjustment rule.

        Returns
        -------
        Schedule
        """
        cal = calendar or NullCalendar()

        if isinstance(frequency, int):
            months_per_period = 12 // frequency
        else:
            months_per_period = frequency.months

        # Generate unadjusted stub dates
        unadjusted: list[date] = []
        d = start
        while True:
            d = _add_months(d, months_per_period)
            if d >= end:
                unadjusted.append(end)
                break
            unadjusted.append(d)

        # Business-day adjust
        adjusted = [cal.adjust(d, bdc) for d in unadjusted]

        # Year fractions from start_date (simulation t=0) to each payment
        payment_times = np.array([
            day_count.year_fraction(start, d) for d in adjusted
        ])

        # Accrual fractions: δᵢ = year_fraction(prev, curr) per period
        period_starts = [start] + adjusted[:-1]
        day_count_fractions = np.array([
            day_count.year_fraction(ps, pe)
            for ps, pe in zip(period_starts, adjusted)
        ])

        return cls(
            start_date=start,
            end_date=end,
            payment_dates=adjusted,
            payment_times=payment_times,
            day_count_fractions=day_count_fractions,
            day_count=day_count,
            calendar=cal,
        )

    def __len__(self) -> int:
        return len(self.payment_dates)

    def __repr__(self) -> str:
        return (
            f"Schedule(start={self.start_date}, end={self.end_date}, "
            f"n={len(self)}, day_count={self.day_count.value})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """Add an integer number of months to a date, clamping to month-end."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    # Clamp day to the last day of the target month
    import calendar as _cal
    last_day = _cal.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))
