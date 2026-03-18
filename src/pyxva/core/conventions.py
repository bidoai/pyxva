"""Day-count conventions, business-day conventions, and market calendars."""
from __future__ import annotations

import datetime
from datetime import date
from enum import Enum


# ---------------------------------------------------------------------------
# Day-count conventions
# ---------------------------------------------------------------------------

class DayCountConvention(Enum):
    """Standard ISDA day-count conventions for year-fraction computation."""

    ACT_360      = "ACT/360"
    ACT_365      = "ACT/365"
    ACT_ACT_ISDA = "ACT/ACT ISDA"
    THIRTY_360   = "30/360"       # US Bond Basis
    THIRTY_E_360 = "30E/360"      # Eurobond Basis

    def year_fraction(self, d1: date, d2: date) -> float:
        """Compute the year fraction between two dates under this convention.

        Parameters
        ----------
        d1, d2 : datetime.date
            Start and end dates (d1 <= d2; sign is flipped otherwise).

        Returns
        -------
        float
        """
        if d1 > d2:
            return -self.year_fraction(d2, d1)
        if self is DayCountConvention.ACT_360:
            return (d2 - d1).days / 360.0
        if self is DayCountConvention.ACT_365:
            return (d2 - d1).days / 365.0
        if self is DayCountConvention.ACT_ACT_ISDA:
            return _act_act_isda(d1, d2)
        if self is DayCountConvention.THIRTY_360:
            return _thirty_360(d1, d2, eurobond=False)
        if self is DayCountConvention.THIRTY_E_360:
            return _thirty_360(d1, d2, eurobond=True)
        raise NotImplementedError(self)  # pragma: no cover


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_in_year(year: int) -> int:
    return 366 if _is_leap(year) else 365


def _act_act_isda(d1: date, d2: date) -> float:
    """ACT/ACT ISDA: split period across calendar years.

    For each calendar year spanned, count actual days and divide by the
    actual number of days in that year (365 or 366).
    """
    if d1.year == d2.year:
        return (d2 - d1).days / _days_in_year(d1.year)

    total = 0.0
    # Partial first year
    year_end = date(d1.year + 1, 1, 1)
    total += (year_end - d1).days / _days_in_year(d1.year)
    # Full intermediate years
    for y in range(d1.year + 1, d2.year):
        total += 1.0
    # Partial last year
    year_start = date(d2.year, 1, 1)
    total += (d2 - year_start).days / _days_in_year(d2.year)
    return total


def _thirty_360(d1: date, d2: date, *, eurobond: bool) -> float:
    """30/360 (US Bond) or 30E/360 (Eurobond) day count.

    US Bond Basis (ISDA 2006 §4.16(f)):
        D1 = min(D1, 30)
        D2 = min(D2, 30) only if D1 == 30

    Eurobond Basis (ISDA 2006 §4.16(g)):
        D1 = min(D1, 30)
        D2 = min(D2, 30)  (always)
    """
    y1, m1, d_1 = d1.year, d1.month, d1.day
    y2, m2, d_2 = d2.year, d2.month, d2.day

    if eurobond:
        d_1 = min(d_1, 30)
        d_2 = min(d_2, 30)
    else:
        d_1 = min(d_1, 30)
        if d_1 == 30:
            d_2 = min(d_2, 30)

    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d_2 - d_1)) / 360.0


# ---------------------------------------------------------------------------
# Business-day conventions
# ---------------------------------------------------------------------------

class BusinessDayConvention(Enum):
    """ISDA business-day adjustment rules."""

    FOLLOWING          = "Following"
    MODIFIED_FOLLOWING = "ModifiedFollowing"
    PRECEDING          = "Preceding"
    UNADJUSTED         = "Unadjusted"


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

class Calendar:
    """Market holiday calendar.

    Parameters
    ----------
    holidays : set[datetime.date] | None
        Explicit holiday dates (weekends are always excluded regardless).
    """

    def __init__(self, holidays: set[date] | None = None) -> None:
        self._holidays: frozenset[date] = frozenset(holidays or set())

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and not self.is_holiday(d)

    def holidays_for_year(self, year: int) -> frozenset[date]:
        """Return the set of holidays in a given year.

        Subclasses override this to compute rule-based holidays. The base
        implementation returns only the explicitly-provided fixed holidays.
        """
        return frozenset(h for h in self._holidays if h.year == year)

    def adjust(self, d: date, convention: BusinessDayConvention) -> date:
        """Adjust a date to the nearest business day per the given convention."""
        if convention is BusinessDayConvention.UNADJUSTED:
            return d
        if self.is_business_day(d):
            return d
        if convention is BusinessDayConvention.FOLLOWING:
            return self._advance(d, 1)
        if convention is BusinessDayConvention.PRECEDING:
            return self._advance(d, -1)
        if convention is BusinessDayConvention.MODIFIED_FOLLOWING:
            candidate = self._advance(d, 1)
            if candidate.month != d.month:
                return self._advance(d, -1)
            return candidate
        raise NotImplementedError(convention)  # pragma: no cover

    def _advance(self, d: date, step: int) -> date:
        d = d + datetime.timedelta(days=step)
        while not self.is_business_day(d):
            d = d + datetime.timedelta(days=step)
        return d

    def _compute_holidays(self, year: int) -> set[date]:
        """Override in subclasses to compute rule-based holidays for ``year``."""
        return set()


class NullCalendar(Calendar):
    """Weekend-only calendar — no public holidays."""

    def is_holiday(self, d: date) -> bool:
        return False

    def holidays_for_year(self, year: int) -> frozenset[date]:
        return frozenset()


# ---------------------------------------------------------------------------
# Easter algorithm (Meeus / Jones / Butcher)
# ---------------------------------------------------------------------------

def _easter(year: int) -> date:
    """Return the date of Easter Sunday for the given year (Gregorian)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence (1-indexed) of *weekday* in month/year.

    weekday: 0=Monday … 6=Sunday (same as date.weekday()).
    """
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + datetime.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last occurrence of weekday in month/year."""
    if month == 12:
        last = date(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - datetime.timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - datetime.timedelta(days=offset)


# ---------------------------------------------------------------------------
# TARGET calendar (ECB / euro area)
# ---------------------------------------------------------------------------

class TARGET(Calendar):
    """ECB TARGET2 settlement calendar.

    Fixed holidays:
    - New Year's Day (Jan 1)
    - Good Friday (Easter − 2)
    - Easter Monday (Easter + 1)
    - Labour Day (May 1)
    - Christmas (Dec 25)
    - Boxing Day (Dec 26)
    """

    def is_holiday(self, d: date) -> bool:
        return d in self._get_year_holidays(d.year)

    def holidays_for_year(self, year: int) -> frozenset[date]:
        return frozenset(self._get_year_holidays(year))

    @staticmethod
    def _get_year_holidays(year: int) -> set[date]:
        easter = _easter(year)
        return {
            date(year, 1, 1),                              # New Year's Day
            easter - datetime.timedelta(days=2),           # Good Friday
            easter + datetime.timedelta(days=1),           # Easter Monday
            date(year, 5, 1),                              # Labour Day
            date(year, 12, 25),                            # Christmas
            date(year, 12, 26),                            # Boxing Day
        }


# ---------------------------------------------------------------------------
# US calendar (Federal Reserve / NYSE)
# ---------------------------------------------------------------------------

class USCalendar(Calendar):
    """US Federal Reserve / NYSE settlement calendar.

    Fixed and floating holidays:
    - New Year's Day (Jan 1, nearest Mon if weekend)
    - MLK Day (3rd Monday of January)
    - Presidents Day (3rd Monday of February)
    - Good Friday (Easter − 2) — NYSE only; included here
    - Memorial Day (last Monday of May)
    - Juneteenth (June 19, nearest Mon if weekend; since 2021)
    - Independence Day (July 4, nearest Mon/Fri if weekend)
    - Labor Day (1st Monday of September)
    - Thanksgiving (4th Thursday of November)
    - Christmas (Dec 25, nearest Mon/Fri if weekend)
    """

    def is_holiday(self, d: date) -> bool:
        return d in self._get_year_holidays(d.year)

    def holidays_for_year(self, year: int) -> frozenset[date]:
        return frozenset(self._get_year_holidays(year))

    @staticmethod
    def _nearest_monday(d: date) -> date:
        """Observe on nearest Monday (Fri → prev Mon if Sat, Mon if Sun)."""
        w = d.weekday()
        if w == 5:  # Saturday → Friday
            return d - datetime.timedelta(days=1)
        if w == 6:  # Sunday → Monday
            return d + datetime.timedelta(days=1)
        return d

    @classmethod
    def _get_year_holidays(cls, year: int) -> set[date]:
        easter = _easter(year)
        holidays = {
            cls._nearest_monday(date(year, 1, 1)),          # New Year's Day
            _nth_weekday(year, 1, 0, 3),                    # MLK Day (3rd Mon Jan)
            _nth_weekday(year, 2, 0, 3),                    # Presidents Day (3rd Mon Feb)
            easter - datetime.timedelta(days=2),            # Good Friday
            _last_weekday(year, 5, 0),                      # Memorial Day (last Mon May)
            _nth_weekday(year, 9, 0, 1),                    # Labor Day (1st Mon Sep)
            _nth_weekday(year, 11, 3, 4),                   # Thanksgiving (4th Thu Nov)
            cls._nearest_monday(date(year, 7, 4)),          # Independence Day
            cls._nearest_monday(date(year, 12, 25)),        # Christmas
        }
        if year >= 2021:
            holidays.add(cls._nearest_monday(date(year, 6, 19)))  # Juneteenth
        return holidays
