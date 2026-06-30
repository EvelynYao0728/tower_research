"""
Factor execution + evaluation.

The historical Qlib/Docker runner has been replaced by
:class:`quantaalpha.factors.private_runner.PrivateFactorRunner`, which writes
long-format parquet shards and calls ``research/backtest``.
"""

from quantaalpha.factors.private_runner import PrivateFactorRunner

# Backward-compatible alias for notebooks or external imports
QlibFactorRunner = PrivateFactorRunner

__all__ = ["PrivateFactorRunner", "QlibFactorRunner"]
