"""
Load per-factor metrics from ``research/backtest/output/summary.csv``.

Trajectory pools and evolution ranking use the core backtest metrics only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from quantaalpha.backtest.research_bt import DEFAULT_RESEARCH_ROOT

# Must match ``backtest.single_factor_bt.metrics.CORE_SUMMARY_METRIC_COLUMNS``.
SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "IC",
    "ICIR",
    "RankIC",
    "RankICIR",
    "long_ret",
    "short_ret",
    "long_short_ret",
    "volatility",
    "sharpe",
    "turnover",
)


def is_summary_metric_column(col: str) -> bool:
    """Whether a column is a published backtest summary metric."""
    return col in SUMMARY_METRIC_COLUMNS


def default_summary_csv_path() -> Path:
    env = os.environ.get("QUANTALPHA_BT_SUMMARY")
    if env:
        return Path(env).expanduser().resolve()
    out = os.environ.get(
        "QUANTALPHA_BT_OUTPUT",
        str(DEFAULT_RESEARCH_ROOT / "backtest" / "output"),
    )
    return Path(out).expanduser().resolve() / "summary.csv"


def load_research_summary(path: Path | str | None = None) -> pd.DataFrame:
    p = Path(path).expanduser().resolve() if path else default_summary_csv_path()
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "factor" not in df.columns:
        return pd.DataFrame()
    return df


def _float_or_none(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def summary_row_to_metrics(row: pd.Series) -> dict[str, Optional[float]]:
    """One summary.csv row -> JSON-safe metrics dict (research column names)."""
    return {
        col: _float_or_none(row[col]) for col in SUMMARY_METRIC_COLUMNS if col in row.index
    }


def _merge_metric_dicts(
    *parts: dict[str, Optional[float]],
) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for part in parts:
        for key, val in part.items():
            if val is not None or key not in out:
                out[key] = val
    return out


def metrics_from_summary_dataframe(
    summary_df: pd.DataFrame,
    factor_names: list[str],
) -> dict[str, dict[str, Optional[float]]]:
    """Build per-factor metric dicts from an in-memory backtest summary table."""
    if summary_df.empty or "factor" not in summary_df.columns:
        return {}
    names = {n for n in factor_names if n}
    if not names:
        return {}
    subset = summary_df[summary_df["factor"].astype(str).isin(names)]
    per_factor: dict[str, dict[str, Optional[float]]] = {}
    for _, row in subset.iterrows():
        per_factor[str(row["factor"])] = summary_row_to_metrics(row)
    return per_factor


def metrics_for_factor_names(
    factor_names: list[str],
    *,
    summary_path: Path | str | None = None,
) -> dict[str, dict[str, Optional[float]]]:
    """
    Return ``{factor_name: metrics_dict}`` for names present in summary.csv.

    Missing names are omitted (caller may treat as no backtest yet).
    """
    names = [n for n in factor_names if n]
    if not names:
        return {}

    return metrics_from_summary_dataframe(
        load_research_summary(summary_path), names
    )


def experiment_result_frame_for_factors(
    summary_df: pd.DataFrame,
    factor_names: list[str],
    *,
    metric: str = "RankIC",
) -> pd.Series | None:
    """
    Build ``experiment.result`` from summary rows: best factor among *factor_names*.

    Returns a metric-indexed ``Series`` (compatible with ``process_results`` /
    ``metrics_from_experiment_result``), or ``None`` when no matching rows exist.
    """
    if summary_df is None or summary_df.empty or not factor_names:
        return None
    per_factor = metrics_from_summary_dataframe(summary_df, factor_names)
    if not per_factor:
        return None
    best = pick_best_factor_metrics(per_factor, metric=metric)
    if not best:
        return None
    ordered = {col: best.get(col) for col in SUMMARY_METRIC_COLUMNS if col in best}
    if not ordered:
        return None
    return pd.Series(ordered)


def pick_best_factor_metrics(
    per_factor: dict[str, dict[str, Optional[float]]],
    *,
    metric: str = "RankIC",
) -> dict[str, Optional[float]]:
    """Choose the factor with largest ``abs(metric)`` among *per_factor* entries."""
    if not per_factor:
        return {}

    best_name: str | None = None
    best_score = -1.0
    for name, metrics in per_factor.items():
        val = metrics.get(metric)
        if val is None:
            continue
        score = abs(float(val))
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        return next(iter(per_factor.values()))
    return dict(per_factor[best_name])


def metrics_from_experiment_result(result: Any) -> dict[str, Optional[float]]:
    """Parse ``experiment.result`` when it is already a research-style metric vector."""
    if result is None:
        return {}

    series: pd.Series | None = None
    if isinstance(result, pd.Series):
        series = result
    elif isinstance(result, pd.DataFrame):
        if result.shape[1] >= 1:
            series = result.iloc[:, 0]
        elif result.shape[0] >= 1:
            series = result.iloc[0]

    if series is None:
        return {}

    idx = {str(i) for i in series.index}
    if not idx.intersection(set(SUMMARY_METRIC_COLUMNS)):
        return {}

    out: dict[str, Optional[float]] = {}
    for col in SUMMARY_METRIC_COLUMNS:
        if col in series.index:
            out[col] = _float_or_none(series[col])
    return out


def resolve_trajectory_backtest_metrics(
    *,
    factor_names: list[str],
    experiment: Any = None,
    summary_path: Path | str | None = None,
) -> tuple[dict[str, Optional[float]], dict[str, dict[str, Optional[float]]]]:
    """
    Metrics for evolution / ``trajectory_pool.json``.

    Priority (per factor, merged left-to-right; later non-None wins):
    1. ``experiment.backtest_summary`` from the current run (freshest).
    2. Rows in ``summary.csv`` matching ``factor_names``.
    3. ``experiment.result`` for the trajectory-level best-factor view only.
    """
    names = [n for n in factor_names if n]
    per_factor: dict[str, dict[str, Optional[float]]] = {}

    if experiment is not None:
        run_summary = getattr(experiment, "backtest_summary", None)
        if isinstance(run_summary, pd.DataFrame) and not run_summary.empty:
            per_factor = metrics_from_summary_dataframe(run_summary, names)

    disk = metrics_for_factor_names(names, summary_path=summary_path)
    for fname, metrics in disk.items():
        per_factor[fname] = _merge_metric_dicts(per_factor.get(fname, {}), metrics)

    traj_best = metrics_from_experiment_result(
        getattr(experiment, "result", None) if experiment is not None else None
    )
    best = pick_best_factor_metrics(per_factor) if per_factor else traj_best
    if not best and traj_best:
        best = traj_best

    return best, per_factor
