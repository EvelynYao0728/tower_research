"""Load factor panels and labels with strict train/test date boundaries."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

_MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from config import (  # noqa: E402
    DEFAULT_FACTOR_ROOT,
    DEFAULT_LABEL_ROOT,
    DEFAULT_REGISTRY,
    DEFAULT_TRADE_DATE,
)
from data import (  # noqa: E402
    available_dates,
    discover_factor_names,
    load_panel,
    load_trade_dates,
)

RESEARCH = _MODEL_ROOT.parent
DEFAULT_TEST_LABEL_ROOT = RESEARCH / "final" / "label"

TRAIN_START = "20250101"
TRAIN_END = "20251231"
TEST_START = "20260101"
TEST_END = "20260430"

META_COLS = ("date", "sym_root", "sym_suffix", "minute", "ticker")
DEFAULT_LABEL_COL = "ex_log_ret_10m"


@dataclass
class DataPaths:
    factor_root: Path = DEFAULT_FACTOR_ROOT
    train_label_root: Path = DEFAULT_LABEL_ROOT
    test_label_root: Path = DEFAULT_TEST_LABEL_ROOT
    trade_date_csv: Path = DEFAULT_TRADE_DATE
    registry: Path = DEFAULT_REGISTRY
    label_col: str = DEFAULT_LABEL_COL


def normalize_date_str(d: str) -> str:
    return str(d).replace("-", "")


def filter_dates_by_range(dates: Sequence[str], start: str, end: str) -> list[str]:
    start, end = normalize_date_str(start), normalize_date_str(end)
    return sorted(d for d in dates if start <= normalize_date_str(d) <= end)


def discover_all_factors(paths: DataPaths) -> list[str]:
    factors = discover_factor_names(paths.factor_root, paths.registry)
    if not factors:
        raise ValueError(f"No factor directories found under {paths.factor_root}")
    return factors


def get_available_dates(
    paths: DataPaths,
    factor_cols: Sequence[str],
    label_root: Path,
    candidate_dates: Sequence[str] | None = None,
) -> list[str]:
    dates = available_dates(
        paths.factor_root,
        label_root,
        factor_cols,
        candidate_dates,
        show_progress=True,
    )
    return dates


def load_train_dates(paths: DataPaths, factor_cols: Sequence[str]) -> list[str]:
    calendar = load_trade_dates(paths.trade_date_csv)
    dates = get_available_dates(paths, factor_cols, paths.train_label_root, calendar)
    return filter_dates_by_range(dates, TRAIN_START, TRAIN_END)


def load_test_dates(paths: DataPaths, factor_cols: Sequence[str]) -> list[str]:
    # Test period (2026) is not in trade_date.csv; discover dates from label files.
    dates = get_available_dates(paths, factor_cols, paths.test_label_root, None)
    return filter_dates_by_range(dates, TEST_START, TEST_END)


def load_panel_for_dates(
    dates: Sequence[str],
    paths: DataPaths,
    factor_cols: Sequence[str],
    label_root: Path,
    *,
    winsorize_q: float = 0.01,
    session_start: int = 931,
    session_end: int = 1559,
    workers: int = 4,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load preprocessed panel (factor winsorize+zscore per minute, raw label kept)."""
    if not dates:
        raise ValueError("No dates provided for panel loading.")
    panel = load_panel(
        dates,
        factor_root=paths.factor_root,
        label_root=label_root,
        factor_cols=factor_cols,
        label_col=paths.label_col,
        winsorize_q=winsorize_q,
        session_start=session_start,
        session_end=session_end,
        workers=workers,
        desc="Loading panel",
        show_progress=True,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    panel = ensure_panel_schema(panel)
    return panel


def ensure_panel_schema(panel: pd.DataFrame) -> pd.DataFrame:
    """Add stock_code alias and datetime for ordering; keep ticker for compatibility."""
    out = panel.copy()
    out["date"] = out["date"].astype(str).str.replace("-", "", regex=False)
    if "ticker" not in out.columns:
        out["ticker"] = out["sym_root"].astype(str) + "." + out["sym_suffix"].astype(str)
    out["stock_code"] = out["ticker"]
    out["datetime"] = pd.to_datetime(
        out["date"].astype(str) + out["minute"].astype(str).str.zfill(4),
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    return out.sort_values(["stock_code", "date", "minute"]).reset_index(drop=True)


def split_panel_by_date_list(panel: pd.DataFrame, dates: Sequence[str]) -> pd.DataFrame:
    date_set = {normalize_date_str(d) for d in dates}
    mask = panel["date"].astype(str).map(normalize_date_str).isin(date_set)
    return panel.loc[mask].reset_index(drop=True)


def diagnose_test_coverage(
    paths: DataPaths,
    factor_cols: Sequence[str],
    test_dates: Sequence[str],
) -> dict:
    """
    Explain why the usable test window may be shorter than TEST_START~TEST_END.
    Pipeline requires every factor + label file to exist for each date.
    """
    label_dates = sorted(
        p.stem.replace("-", "")
        for p in paths.test_label_root.glob("*.parquet")
        if normalize_date_str(p.stem) >= TEST_START
    )
    label_in_range = [d for d in label_dates if normalize_date_str(d) <= TEST_END]

    factor_last: dict[str, str | None] = {}
    for name in factor_cols:
        files = sorted(p.stem.replace("-", "") for p in (paths.factor_root / name).glob("2026*.parquet"))
        factor_last[name] = files[-1] if files else None

    last_dates = [last for last in factor_last.values() if last]
    if last_dates:
        min_last = min(last_dates)
        bottleneck = sorted(name for name, last in factor_last.items() if last == min_last)
    else:
        bottleneck = list(factor_cols)

    return {
        "expected_start": TEST_START,
        "expected_end": TEST_END,
        "label_days_in_range": len(label_in_range),
        "label_range": (label_in_range[0], label_in_range[-1]) if label_in_range else ("", ""),
        "actual_start": test_dates[0] if test_dates else "",
        "actual_end": test_dates[-1] if test_dates else "",
        "actual_days": len(test_dates),
        "bottleneck_factors": bottleneck,
        "factor_last_2026": factor_last,
    }
