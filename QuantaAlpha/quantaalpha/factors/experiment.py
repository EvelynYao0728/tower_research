"""
Factor experiment container for the private data + research backtest stack.

The historical name ``QlibFactorExperiment`` is kept so imports across the pipeline
remain stable, but rdagent / Qlib workspaces are no longer required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from quantaalpha.core.experiment import Experiment

if TYPE_CHECKING:
    import pandas as pd
from quantaalpha.factors.coder.factor import FactorTask


class QlibFactorExperiment(Experiment[FactorTask, None, None]):
    """Holds factor tasks, per-task workspaces (filled by the coder), and backtest results."""

    def __init__(
        self,
        sub_tasks: Sequence[FactorTask] | None = None,
        based_experiments: Sequence["QlibFactorExperiment"] | None = None,
    ) -> None:
        st = list(sub_tasks) if sub_tasks is not None else []
        be = list(based_experiments) if based_experiments is not None else []
        super().__init__(sub_tasks=st, based_experiments=be)
        self.experiment_workspace = None
        self.sub_workspace_list = [None] * len(st)
        #: ``research/backtest`` 落盘后的标准产物路径（按因子名索引），由 :class:`PrivateFactorRunner` 填充。
        self.backtest_artifacts: dict[str, dict[str, str]] | None = None
        #: 本轮回测 ``summary`` 表（IC / RankIC / long-short 等核心列），写入 trajectory_pool 时优先使用。
        self.backtest_summary: "pd.DataFrame | None" = None

    @property
    def tasks(self) -> list[FactorTask]:
        return list(self.sub_tasks)

    @tasks.setter
    def tasks(self, value: Sequence[FactorTask]) -> None:
        value_list = list(value)
        self.sub_tasks = value_list
        self.sub_workspace_list = [None] * len(value_list)
