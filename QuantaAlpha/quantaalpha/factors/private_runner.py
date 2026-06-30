"""
因子回测：优先消费 ``QUANTALPHA_PER_FEATURE_ROOT`` / ``QUANTALPHA_LEGACY_PANEL_ROOT`` 下的日分片数据，
直接调用 ``research/backtest``（``single_factor_bt``），结果落盘到 ``backtest/output/<因子名>/`` 等标准路径，
并在 ``QlibFactorExperiment.backtest_artifacts`` 中记录这些路径。

若磁盘上找不到该因子列/目录，则回退到工作区 ``factor_output.parquet`` / ``execute`` 合并后再跑同一套回测。
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, List

import pandas as pd

from quantaalpha.backtest.research_bt import backtest_output_artifact_paths, run_backtest
from quantaalpha.backtest.summary_metrics import (
    experiment_result_frame_for_factors,
    load_research_summary,
)
from quantaalpha.components.runner import CachedRunner
from quantaalpha.core.conf import RD_AGENT_SETTINGS
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.core.utils import cache_with_pickle, multiprocessing_wrapper
from quantaalpha.data.private_catalog import LONG_KEYS, PrivateDataConfig
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.experiment import QlibFactorExperiment
from quantaalpha.factors.factor_materialize import materialize_workspace_full_panel
from quantaalpha.factors.coder.factor import FactorFBWorkspace
from quantaalpha.log import logger

_DATA_SUFFIXES = {".parquet", ".fea", ".feather", ".pq"}


def _dir_has_feature_shards(d: Path) -> bool:
    if not d.is_dir():
        return False
    try:
        return any(
            p.is_file() and p.suffix.lower() in _DATA_SUFFIXES
            for p in d.iterdir()
        )
    except OSError:
        return False


def _resolve_disk_backtest_input(cfg: PrivateDataConfig, factor_name: str) -> tuple[Any, ...] | None:
    """
    在 ``QUANTALPHA_PER_FEATURE_ROOT/<因子名>`` 或 ``QUANTALPHA_LEGACY_PANEL_ROOT`` 宽表中
    定位 ``research/backtest/single_factor_bt`` 可直接读取的输入（日分片 parquet/feather）。
    """
    per = cfg.per_feature_root / factor_name
    if _dir_has_feature_shards(per):
        return ("per_feature", per)
    root = cfg.legacy_panel_root
    if not root.is_dir():
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    for pattern in ("*.parquet", "*.pq", "*.fea", "*.feather"):
        files = sorted(root.glob(pattern))
        if not files:
            continue
        try:
            cols = set(pq.ParquetFile(files[0]).schema_arrow.names)
        except Exception:
            continue
        if factor_name in cols:
            return ("legacy_col", root, factor_name)
    return None


def _use_disk_panel_first() -> bool:
    return os.environ.get("QUANTALPHA_BT_USE_DISK_PANEL", "1").lower() not in ("0", "false", "no")


def _norm_suffix_series(s: pd.Series) -> pd.Series:
    def one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "None"
        t = str(v).strip()
        return t if t else "None"

    return s.map(one)


def _read_workspace_long(ws) -> pd.DataFrame:
    from quantaalpha.factors.coder.factor import FactorFBWorkspace

    if not isinstance(ws, FactorFBWorkspace):
        raise TypeError(f"Expected FactorFBWorkspace, got {type(ws)}")
    task = ws.target_task
    name = task.factor_name
    parq = ws.workspace_path / "factor_output.parquet"
    if parq.exists():
        df = pd.read_parquet(parq)
        missing = [c for c in LONG_KEYS if c not in df.columns]
        if missing:
            raise ValueError(f"factor_output.parquet missing columns {missing}")
        if name not in df.columns:
            raise ValueError(f"factor_output.parquet must contain column {name!r}")
        df = df[list(LONG_KEYS) + [name]].copy()
        df["sym_suffix"] = _norm_suffix_series(df["sym_suffix"])
        return df

    try:
        names = sorted(p.name for p in ws.workspace_path.iterdir())
    except OSError:
        names = []
    hint = ""
    if (ws.workspace_path / "execution_stdout.txt").exists():
        hint = " 查看同目录 execution_stdout.txt（子进程 stdout/stderr）。"
    elif (ws.workspace_path / "factor.py").exists():
        hint = " 已生成 factor.py 但未写出结果；请修正脚本使其在当前目录写入 factor_output.parquet。"
    raise FactorEmptyError(
        f"No factor_output.parquet in {ws.workspace_path}. "
        f"已有文件: {names}.{hint}"
    )


def _long_from_execute_return(ws, exec_out: Any) -> pd.DataFrame:
    """
    ``FactorFBWorkspace.execute`` 被 pickle 缓存命中时不会写盘，工作区可能为空；
    此处用 execute 返回的 ``(feedback, dataframe)`` 恢复长表，并尽量落盘便于排查。
    """
    from quantaalpha.factors.coder.factor import FactorFBWorkspace

    if not isinstance(ws, FactorFBWorkspace):
        raise TypeError(f"Expected FactorFBWorkspace, got {type(ws)}")
    if not isinstance(exec_out, tuple) or len(exec_out) < 2:
        raise FactorEmptyError(f"execute 返回异常（非 tuple）: {type(exec_out)}")
    feedback, df = exec_out[0], exec_out[1]
    name = ws.target_task.factor_name
    if df is None:
        fb = str(feedback).strip() if feedback is not None else ""
        fb_short = fb[:1200] if fb else "(无)"
        ws_path = ws.workspace_path
        lines = [
            f"因子「{name}」: execute 返回的因子表为 None（第二项为空），无法写入回测输入。",
            f"工作目录: {ws_path}",
            f"execute 反馈: {fb_short}",
        ]
        low = fb.lower()
        if "code is not set" in low or "缺少 factor.py" in fb:
            lines.append(
                "说明: 当前工作区尚未注入可执行的 factor.py；流水线应先跑通 "
                "factor_construct → factor_calculate（生成代码），再进入 factor_backtest / PrivateFactorRunner。"
            )
        elif not fb:
            lines.append(
                "说明: 无文本反馈且未落盘 factor_output.parquet；"
                "请查看同目录 execution_stdout.txt 或检查子进程是否超时/崩溃。"
            )
        else:
            lines.append(
                "说明: 子进程可能执行失败或未写出 factor_output.parquet；"
                "请结合 execution_stdout.txt 排查。"
            )
        raise FactorEmptyError("\n".join(lines))

    if isinstance(df, pd.Series):
        frame: pd.DataFrame | pd.Series = df
    elif isinstance(df, pd.DataFrame):
        frame = df
    else:
        raise FactorEmptyError(f"execute 返回的第二项类型不支持: {type(df)}")

    if isinstance(frame, pd.DataFrame) and all(c in frame.columns for c in LONG_KEYS) and name in frame.columns:
        out = frame[list(LONG_KEYS) + [name]].copy()
        out["sym_suffix"] = _norm_suffix_series(out["sym_suffix"])
    else:
        raise FactorEmptyError(
            f"execute 返回值须为含 {list(LONG_KEYS)} 与因子列 {name!r} 的长表 DataFrame，"
            f"或在工作区写出 factor_output.parquet；实际列: {getattr(frame, 'columns', frame)}"
        )

    try:
        ws.workspace_path.mkdir(parents=True, exist_ok=True)
        out.to_parquet(ws.workspace_path / "factor_output.parquet", index=False)
    except OSError:
        pass
    return out


def _merge_on_keys(parts: List[pd.DataFrame]) -> pd.DataFrame:
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=list(LONG_KEYS), how="outer")
    return out


def _write_daily_shards(merged: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    factor_cols = [c for c in merged.columns if c not in LONG_KEYS]
    for dkey, g in merged.groupby("date"):
        stem = str(int(dkey))
        path = out_dir / f"{stem}.parquet"
        g[list(LONG_KEYS) + factor_cols].to_parquet(path, index=False)


class PrivateFactorRunner(CachedRunner[QlibFactorExperiment]):
    """优先从磁盘面板调用 ``research/backtest``；否则从工作区 materialize 后再调用。"""

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:
        del use_local  # Docker path removed for private deployment

        if exp.based_experiments and exp.based_experiments[-1].result is None:
            tail = exp.based_experiments[-1]
            if tail.sub_tasks:
                exp.based_experiments[-1] = self.develop(tail)
            else:
                tail.result = None

        if not exp.sub_workspace_list:
            raise FactorEmptyError("No factor workspaces to execute.")

        cfg = PrivateDataConfig()
        exp.backtest_artifacts = {}
        summary_parts: List[pd.DataFrame] = []
        covered_indices: set[int] = set()

        if _use_disk_panel_first():
            for i, task in enumerate(exp.sub_tasks):
                name = task.factor_name
                resolved = _resolve_disk_backtest_input(cfg, name)
                if resolved is None:
                    continue
                covered_indices.add(i)
                if resolved[0] == "per_feature":
                    factor_dir = resolved[1]
                    logger.info(
                        "回测（磁盘 per-feature）: 因子 %r -> run_backtest(%s)",
                        name,
                        factor_dir,
                    )
                    bt = run_backtest(factor_dir, use_cache=True)
                else:
                    legacy_root, col = resolved[1], resolved[2]
                    logger.info(
                        "回测（磁盘 legacy 列）: 因子 %r -> run_backtest(%s, factor_col=%r)",
                        name,
                        legacy_root,
                        col,
                    )
                    bt = run_backtest(legacy_root, factor_col=col, use_cache=True)

                if not bt.summary.empty:
                    summary_parts.append(bt.summary)
                for k, p in backtest_output_artifact_paths(bt.output_dir, name).items():
                    exp.backtest_artifacts.setdefault(name, {})[k] = str(p)

        uncovered = [i for i in range(len(exp.sub_tasks)) if i not in covered_indices]

        if uncovered:
            for i in uncovered:
                ws = exp.sub_workspace_list[i]
                if ws is not None and hasattr(ws, "hydrate_code_dict_from_disk"):
                    if ws.hydrate_code_dict_from_disk():
                        logger.debug(
                            "Hydrated factor.py from disk into code_dict before execute: %s",
                            getattr(ws, "workspace_path", ws),
                        )
                if (
                    FACTOR_COSTEER_SETTINGS.promote_full_panel_after_debug
                    and isinstance(ws, FactorFBWorkspace)
                ):
                    try:
                        materialize_workspace_full_panel(ws)
                    except Exception as e:
                        logger.warning(
                            "回测前全量物化失败，将回退 execute(All): %s",
                            e,
                        )

            exec_outputs = multiprocessing_wrapper(
                [(exp.sub_workspace_list[i].execute, ("All",)) for i in uncovered],
                n=RD_AGENT_SETTINGS.multi_proc_n,
            )

            long_parts: List[pd.DataFrame] = []
            for j, i in enumerate(uncovered):
                ws = exp.sub_workspace_list[i]
                exec_out = exec_outputs[j]
                try:
                    long_parts.append(_read_workspace_long(ws))
                except FactorEmptyError:
                    long_parts.append(_long_from_execute_return(ws, exec_out))

            merged = _merge_on_keys(long_parts)
            if merged.empty:
                raise FactorEmptyError("Merged factor frame is empty.")

            anchor = exp.sub_workspace_list[uncovered[0]].workspace_path
            bt_root = anchor.parent / f"factor_bt_input_{uuid.uuid4().hex[:8]}"
            _write_daily_shards(merged, bt_root)
            logger.info(f"Wrote backtest input parquet shards under {bt_root}")

            bt = run_backtest(bt_root, use_cache=True)
            logger.info(f"Backtest summary (workspace merge):\n{bt.summary.to_string(index=False)}")
            if not bt.summary.empty:
                summary_parts.append(bt.summary)
                for _, row in bt.summary.iterrows():
                    fn = str(row["factor"])
                    for k, p in backtest_output_artifact_paths(bt.output_dir, fn).items():
                        exp.backtest_artifacts.setdefault(fn, {})[k] = str(p)

        if not summary_parts:
            exp.backtest_summary = None
            exp.result = None
            if not exp.backtest_artifacts:
                exp.backtest_artifacts = None
            return exp

        full_summary = pd.concat(summary_parts, ignore_index=True)
        if not full_summary.empty and "factor" in full_summary.columns:
            full_summary = full_summary.drop_duplicates(subset=["factor"], keep="last")
        exp.backtest_summary = full_summary if not full_summary.empty else None
        logger.info(f"Backtest summary (combined):\n{full_summary.to_string(index=False)}")

        factor_names = [t.factor_name for t in exp.sub_tasks if getattr(t, "factor_name", None)]
        disk_summary = load_research_summary()
        exp.result = experiment_result_frame_for_factors(disk_summary, factor_names)
        if exp.result is None:
            exp.result = experiment_result_frame_for_factors(full_summary, factor_names)
        if exp.result is None and not full_summary.empty:
            logger.warning(
                "No summary rows for this experiment's factors %s; experiment.result left unset.",
                factor_names,
            )

        return exp
