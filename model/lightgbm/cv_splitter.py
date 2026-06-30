"""Expanding-window time-series CV with gap months to reduce minute-level leakage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from data_loader import normalize_date_str


@dataclass(frozen=True)
class CVFold:
    fold_id: int
    train_dates: list[str]
    gap_dates: list[str]
    val_dates: list[str]


def _date_to_period(d: str) -> str:
    d = normalize_date_str(d)
    return f"{d[:4]}-{d[4:6]}"


def _filter_by_periods(dates: Sequence[str], periods: set[str]) -> list[str]:
    return sorted(d for d in dates if _date_to_period(d) in periods)


def _periods_inclusive(start_ym: str, end_ym: str) -> set[str]:
    """Return YYYY-MM strings from start_ym through end_ym inclusive."""
    start = pd.Period(start_ym, freq="M")
    end = pd.Period(end_ym, freq="M")
    return {str(p) for p in pd.period_range(start, end, freq="M")}


DEFAULT_FOLD_SPECS: list[tuple[str, str, str, str]] = [
    ("2025-01", "2025-06", "2025-07", "2025-08"),
    ("2025-01", "2025-07", "2025-08", "2025-09"),
    ("2025-01", "2025-08", "2025-09", "2025-10"),
    ("2025-01", "2025-09", "2025-10", "2025-11"),
    ("2025-01", "2025-10", "2025-11", "2025-12"),
]


def build_expanding_folds(
    all_dates: Sequence[str],
    *,
    n_folds: int = 5,
    gap_months: int = 1,
    fold_specs: Sequence[tuple[str, str, str, str]] | None = None,
) -> list[CVFold]:
    """
    Build expanding-window folds with a gap month between train and validation.
    Default specs follow the strategy document (5 folds, 1-month gap).
    """
    specs = list(fold_specs or DEFAULT_FOLD_SPECS)[:n_folds]
    if gap_months != 1 and fold_specs is None:
        raise NotImplementedError("Custom gap_months requires explicit fold_specs.")

    folds: list[CVFold] = []
    for i, (train_start, train_end, gap_ym, val_ym) in enumerate(specs, start=1):
        train_periods = _periods_inclusive(train_start, train_end)
        gap_periods = {gap_ym}
        val_periods = {val_ym}

        train_dates = _filter_by_periods(all_dates, train_periods)
        gap_dates = _filter_by_periods(all_dates, gap_periods)
        val_dates = _filter_by_periods(all_dates, val_periods)

        if not train_dates or not val_dates:
            raise ValueError(
                f"Fold {i}: insufficient dates (train={len(train_dates)}, val={len(val_dates)}).",
            )
        folds.append(CVFold(i, train_dates, gap_dates, val_dates))
    return folds
