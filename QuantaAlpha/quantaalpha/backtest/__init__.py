"""Bridge to the private ``research/backtest`` single-factor engine."""

from quantaalpha.backtest.research_bt import (
    BacktestResult,
    backtest_output_artifact_paths,
    default_paths,
    run_backtest,
)

__all__ = [
    "BacktestResult",
    "backtest_output_artifact_paths",
    "default_paths",
    "run_backtest",
]
