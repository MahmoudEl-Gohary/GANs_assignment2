"""Shared utility functions for date validation and evaluation."""

from datetime import datetime


# Mapping constants used across modules
MONTHS_MAP: dict[str, int] = {
    '[JAN]': 1, '[FEB]': 2, '[MAR]': 3, '[APR]': 4,
    '[MAY]': 5, '[JUN]': 6, '[JUL]': 7, '[AUG]': 8,
    '[SEP]': 9, '[OCT]': 10, '[NOV]': 11, '[DEC]': 12,
}

DAYS_MAP: dict[str, int] = {
    '[MON]': 0, '[TUE]': 1, '[WED]': 2, '[THU]': 3,
    '[FRI]': 4, '[SAT]': 5, '[SUN]': 6,
}


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year per the Gregorian calendar."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def validate_date(
    date_str: str,
    cond_day: str,
    cond_month: str,
    cond_leap: str,
    cond_decade: str,
) -> bool:
    """Validate a generated date string against all four calendar conditions.

    Args:
        date_str: Date in dd-mm-yyyy format.
        cond_day: Day condition token, e.g. '[MON]'.
        cond_month: Month condition token, e.g. '[JAN]'.
        cond_leap: Leap year condition, '[True]' or '[False]'.
        cond_decade: Decade condition token, e.g. '[192]'.

    Returns:
        True if the date satisfies all four conditions.
    """
    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")

        year = dt.year
        if is_leap_year(year) != (cond_leap == "[True]"):
            return False

        expected_decade_start = int(cond_decade[1:-1]) * 10
        if not (expected_decade_start <= year < expected_decade_start + 10):
            return False

        if dt.month != MONTHS_MAP[cond_month]:
            return False

        if dt.weekday() != DAYS_MAP[cond_day]:
            return False

        return True
    except (ValueError, KeyError):
        return False


def check_individual_conditions(
    date_str: str,
    cond_day: str,
    cond_month: str,
    cond_leap: str,
    cond_decade: str,
) -> dict[str, bool]:
    """Check each condition individually for per-condition accuracy tracking.

    Args:
        date_str: Date in dd-mm-yyyy format.
        cond_day: Day condition token.
        cond_month: Month condition token.
        cond_leap: Leap year condition.
        cond_decade: Decade condition token.

    Returns:
        Dict with keys 'valid_date', 'day', 'month', 'leap', 'decade'
        indicating which conditions passed.
    """
    results = {
        "valid_date": False,
        "day": False,
        "month": False,
        "leap": False,
        "decade": False,
    }

    try:
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        results["valid_date"] = True
    except ValueError:
        return results

    year = dt.year

    results["leap"] = is_leap_year(year) == (cond_leap == "[True]")

    expected_decade_start = int(cond_decade[1:-1]) * 10
    results["decade"] = expected_decade_start <= year < expected_decade_start + 10

    results["month"] = dt.month == MONTHS_MAP.get(cond_month, -1)
    results["day"] = dt.weekday() == DAYS_MAP.get(cond_day, -1)

    return results
