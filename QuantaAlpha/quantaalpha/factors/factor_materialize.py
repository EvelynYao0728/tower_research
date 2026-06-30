"""
Debugging 结束后将因子从「少量交易日」提升为「全部交易日」，并写入回测可读路径。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantaalpha.data.private_catalog import LONG_KEYS, PrivateDataConfig
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.coder.factor import FactorFBWorkspace, _resolve_expr_and_name
from quantaalpha.factors.coder.factor_eval import (
    calculate_factor_to_parquet,
    resolve_factor_eval_workers,
)
from quantaalpha.log import logger


def _write_daily_shards(merged: pd.DataFrame, out_dir: Path, factor_col: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dkey, g in merged.groupby("date"):
        stem = str(int(dkey))
        g[list(LONG_KEYS) + [factor_col]].to_parquet(out_dir / f"{stem}.parquet", index=False)


def materialize_workspace_full_panel(ws: FactorFBWorkspace) -> Path:
    """
    在 CoSTEER Debugging（少量交易日）通过后，按**全部可用交易日**重算因子。

    写出 ``workspace/factor_output.parquet``，并可选写入
    ``QUANTALPHA_PER_FEATURE_ROOT/<因子名>/YYYYMMDD.parquet`` 供 ``PrivateFactorRunner`` 直接回测。
    """
    ws.hydrate_code_dict_from_disk()
    code = (ws.code_dict or {}).get("factor.py", "")
    expr, name = _resolve_expr_and_name(ws.target_task, code)
    if not expr or not name:
        raise ValueError(
            f"无法全量物化：缺少 factor_expression / factor_name（workspace={ws.workspace_path}）"
        )

    ws.workspace_path.mkdir(parents=True, exist_ok=True)
    out_parq = ws.workspace_path / "factor_output.parquet"
    marker = ws.workspace_path / ".full_panel_promoted"
    if marker.is_file() and out_parq.is_file():
        logger.info("已全量物化，跳过: %s", out_parq)
        return out_parq

    workers = resolve_factor_eval_workers()
    logger.info(
        "全量物化因子 %r（全部交易日，Debugging 后，workers=%d）-> %s",
        name,
        workers,
        out_parq,
    )
    calculate_factor_to_parquet(
        expr,
        name,
        out_parq,
        workers=workers,
        panel_max_days=0,
    )
    df = pd.read_parquet(out_parq)
    ws._last_execute_df = df  # noqa: SLF001
    marker.write_text(f"days={df['date'].nunique()}\n", encoding="utf-8")

    if FACTOR_COSTEER_SETTINGS.write_per_feature_shards_on_promote:
        cfg = PrivateDataConfig()
        shard_dir = cfg.per_feature_root / name
        _write_daily_shards(df, shard_dir, name)
        logger.info("全量日分片 -> %s (%d days)", shard_dir, df["date"].nunique())

    return out_parq


def promote_experiment_after_debug(exp) -> None:
    """对实验中每个有效 workspace 做全量物化（在 factor_backtest 之前调用）。"""
    if not FACTOR_COSTEER_SETTINGS.promote_full_panel_after_debug:
        return
    for ws in getattr(exp, "sub_workspace_list", []) or []:
        if ws is None or not isinstance(ws, FactorFBWorkspace):
            continue
        if not (ws.workspace_path / "factor.py").is_file() and "factor.py" not in (ws.code_dict or {}):
            continue
        try:
            materialize_workspace_full_panel(ws)
        except Exception as e:
            logger.warning("全量物化跳过 %s: %s", getattr(ws, "workspace_path", ws), e)
