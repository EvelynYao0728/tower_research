"""
Thin wrapper around ``/home/yzyao.25/research/backtest/single_factor_bt``.

Agents obtain structured metrics (summary rows, paths) without shelling out to CLI.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RESEARCH_ROOT = Path(
    os.environ.get("QUANTALPHA_RESEARCH_ROOT", "/home/yzyao.25/research")
).expanduser().resolve()
DEFAULT_BACKTEST_PKG = DEFAULT_RESEARCH_ROOT / "backtest"
DEFAULT_LABEL = DEFAULT_RESEARCH_ROOT / "data" / "label"
DEFAULT_TRADE_DATES = DEFAULT_RESEARCH_ROOT / "data" / "trade_date.csv"


def backtest_output_artifact_paths(output_root: Path, factor_name: str) -> dict[str, Path]:
    """
    ``single_factor_bt`` 默认落盘布局（与 ``backtest/output/F1_relative_spread`` 等一致）。

    Returns
    -------
    dict
        含 ``factor_output_dir``、``metrics_per_minute_csv``、``decile_inner_long_short_csv``、
        ``intraday_ic_profile_csv``、根目录 ``summary_csv``。
    """
    root = Path(output_root).expanduser().resolve()
    sub = root / factor_name
    return {
        "factor_output_dir": sub,
        "metrics_per_minute_csv": sub / "metrics_per_minute.csv",
        "decile_inner_long_short_csv": sub / "decile_inner_long_short.csv",
        "intraday_ic_profile_csv": sub / "intraday_ic_profile.csv",
        "summary_csv": root / "summary.csv",
    }


def _ensure_backtest_on_path() -> None:
    root = str(DEFAULT_BACKTEST_PKG)
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class BacktestResult:
    """Outcome of :func:`run_backtest` for agent consumption."""

    summary: pd.DataFrame
    output_dir: Path
    factor_names: list[str]

    def artifacts_for(self, factor_name: str) -> dict[str, Path]:
        """磁盘上该因子对应的标准输出路径（见 :func:`backtest_output_artifact_paths`）。"""
        return backtest_output_artifact_paths(self.output_dir, factor_name)

    def best_row_by_abs_rank_ic(self) -> pd.Series:
        if self.summary.empty:
            return pd.Series(dtype=float)
        idx = self.summary["RankIC"].abs().idxmax()
        return self.summary.loc[idx]

    def to_feedback_table(self) -> str:
        return self.summary.to_string(index=False)


def default_paths() -> dict[str, Path]:
    return {
        "research_root": DEFAULT_RESEARCH_ROOT,
        "backtest_pkg": DEFAULT_BACKTEST_PKG,
        "label": DEFAULT_LABEL,
        "trade_date_csv": DEFAULT_TRADE_DATES,
    }


def run_backtest(
    factor_path: Path | str,
    *,
    label_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    factor_col: str | None = None,
    label_col: str | None = None,
    trade_date_csv: Path | str | None = DEFAULT_TRADE_DATES,
    workers: int | None = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> BacktestResult:
    """
    Run the private single-factor backtest package.

    ``factor_path`` may be a directory of daily long parquets or a single file;
    see ``research/backtest/README.md``.

    Extra ``kwargs`` are forwarded to ``single_factor_bt.engine.run_backtest``
    (e.g. ``n_groups``, ``inner_q``, ``session_start``).
    """
    _ensure_backtest_on_path()
    from single_factor_bt.engine import run_backtest as _engine_run
    from single_factor_bt.safety import assert_safe_output

    factor_path = Path(factor_path).expanduser().resolve()
    label_path = Path(
        label_path or os.environ.get(
            "QUANTALPHA_LABEL_ROOT", str(DEFAULT_LABEL)
        )
    ).expanduser().resolve()
    out = assert_safe_output(
        Path(
            output_dir
            or os.environ.get(
                "QUANTALPHA_BT_OUTPUT",
                str(DEFAULT_RESEARCH_ROOT / "backtest" / "output"),
            )
        ).expanduser().resolve()
    )

    summary, out_root = _engine_run(
        factor_path=factor_path,
        label_path=label_path,
        output_dir=out,
        factor_col=factor_col,
        label_col=label_col,
        workers=workers,
        use_cache=use_cache,
        trade_date_csv=Path(trade_date_csv).expanduser().resolve()
        if trade_date_csv
        else None,
        **kwargs,
    )

    f_cols = list(summary["factor"]) if "factor" in summary.columns else (
        [factor_col] if factor_col else []
    )
    return BacktestResult(summary=summary, output_dir=out_root, factor_names=f_cols)


def run_backtest_from_agent_workspace(
    workspace_root: Path,
    **kwargs: Any,
) -> BacktestResult:
    """
    Convenience: pick up merged factor parquet(s) written under an experiment workspace.

    The :class:`PrivateFactorRunner` writes ``factor_bt_input/``; this helper
    points the engine at that folder.
    """
    bt_dir = Path(workspace_root) / "factor_bt_input"
    if not bt_dir.is_dir():
        raise FileNotFoundError(f"Missing factor_bt_input directory: {bt_dir}")
    return run_backtest(bt_dir, **kwargs)
