"""
为 ``template.jinjia2`` 因子执行提供数据：从私有 parquet（legacy / per-feature）加载，
**不**使用 ``daily_pv.h5``。

通过表达式中的 ``$字段`` 推断要加载的列，调用 :func:`quantaalpha.data.private_catalog.load_feature_long`，
再转为 ``function_lib`` 所需的 ``instrument`` + ``datetime`` 列结构。
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from quantaalpha.data.private_catalog import LONG_KEYS, PrivateDataConfig, PrivateMarketCatalog, _iter_day_files, load_feature_long


def normalize_expr_dollar_fields(expr: str, cfg: PrivateDataConfig | None = None) -> str:
    """
    将表达式里与私有目录一致的**裸**字段名补成 ``$字段``，便于 ``load_feature_long`` 推断列；
    不替换已为 ``$xxx`` 的写法，且跳过与 ``function_lib`` 中大写因子函数同名的目录字段，避免误伤 ``RANK(...)`` 等调用。
    """
    try:
        catalog = PrivateMarketCatalog(cfg or PrivateDataConfig())
        fields = catalog.all_fields()
    except Exception:
        return expr
    try:
        from quantaalpha.factors.coder import function_lib as fl

        reserved = {
            n
            for n, v in vars(fl).items()
            if callable(v) and not n.startswith("_") and n[:1].isalpha() and n[0].isupper()
        }
    except Exception:
        reserved = set()
    out = expr
    for bare in sorted(
        ((f[1:] if str(f).startswith("$") else f) for f in fields), key=len, reverse=True
    ):
        if not bare or bare in reserved:
            continue
        out = re.sub(rf"(?<!\$)\b{re.escape(bare)}\b", f"${bare}", out)
    return out


def _resolve_panel_dates(cfg: PrivateDataConfig, fields: list[str]) -> list[str] | None:
    """
    决定 ``load_feature_long`` 的 ``dates`` 参数。

    默认加载**全部**可用交易日。仅当 ``QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS`` 为正整数时，
    才截取最近 N 天（用于快速调试）。
    """
    raw = os.environ.get("QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS", "0").strip()
    if raw.lower() in ("", "0", "all", "none", "full"):
        return None
    try:
        max_days = int(raw)
    except ValueError:
        return None
    if max_days <= 0:
        return None
    return _recent_day_stems_for_fields(cfg, fields, max_days)


def _collect_day_stems(cfg: PrivateDataConfig, fields: list[str]) -> list[str]:
    """合并 legacy / per-feature 源上的全部交易日 stem（YYYYMMDD）。"""
    stems: set[str] = set()
    cat_root = cfg.legacy_panel_root
    if cat_root.is_dir():
        for fp in _iter_day_files(cat_root):
            stems.add(fp.stem.replace("-", ""))
    pf_root = cfg.per_feature_root
    for f in fields:
        sub = pf_root / f
        if sub.is_dir():
            for fp in _iter_day_files(sub):
                stems.add(fp.stem.replace("-", ""))
    return sorted(stems)


def list_day_stems_for_fields(cfg: PrivateDataConfig, fields: list[str]) -> list[str]:
    """表达式涉及字段的全部交易日（受 ``QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS`` 限制时可截断）。"""
    limited = _resolve_panel_dates(cfg, fields)
    if limited is not None:
        return limited
    return _collect_day_stems(cfg, fields)


def _recent_day_stems_for_fields(cfg: PrivateDataConfig, fields: list[str], max_days: int) -> list[str] | None:
    """取最近 ``max_days`` 个交易日 stem。"""
    ordered = _collect_day_stems(cfg, fields)
    if not ordered:
        return None
    if len(ordered) > max_days:
        return ordered[-max_days:]
    return ordered


def _long_to_eval_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """长表 (date, minute, sym_*, 特征列) -> 带 ``instrument`` / ``datetime`` 的宽表行。"""
    if raw.empty:
        raise ValueError("load_feature_long returned an empty frame")

    missing = [k for k in LONG_KEYS if k not in raw.columns]
    if missing:
        raise ValueError(f"Missing long keys {missing} in loaded frame")

    # date may be YYYYMMDD int/str or ISO "YYYY-MM-DD" strings in parquet
    dt = pd.to_datetime(raw["date"], errors="coerce")
    hh = raw["minute"].astype(np.int64) // 100
    mm = raw["minute"].astype(np.int64) % 100
    dt = dt + pd.to_timedelta(hh * 60 + mm, unit="m")

    root = raw["sym_root"].astype(str)
    sfx = raw["sym_suffix"].astype(str)
    instrument = np.where(
        sfx.str.upper().isin(("NONE", "")) | sfx.isna(),
        root,
        root + "." + sfx.str.upper(),
    )

    feat_cols = [c for c in raw.columns if c not in LONG_KEYS]
    out = raw[feat_cols].copy()
    out["instrument"] = instrument
    out["datetime"] = dt
    # 与表达式中 ``$xxx`` 对齐：列名统一为带 ``$`` 前缀
    ren = {c: f"${c}" if not str(c).startswith("$") else c for c in feat_cols}
    out = out.rename(columns=ren)
    return out


def load_frame_for_expression(expr: str) -> pd.DataFrame:
    """
    根据 ``expr`` 加载全年面板（``instrument`` + ``datetime`` + ``$字段``）。

    **慢**：约 5000 万行。因子生成请用 :func:`quantaalpha.factors.coder.factor_eval.calculate_factor_to_parquet`。
    """
    from quantaalpha.data.private_catalog import validate_factor_expression_fields

    ok, msg = validate_factor_expression_fields(expr)
    if not ok:
        raise ValueError(msg)

    expr_for_fields = normalize_expr_dollar_fields(expr)
    fields = sorted({m.group(1) for m in re.finditer(r"\$([A-Za-z_][A-Za-z0-9_]*)", expr_for_fields)})
    if not fields:
        raise ValueError(f"No $feature tokens found in expression: {expr[:200]!r}")

    cfg = PrivateDataConfig()
    dates = _resolve_panel_dates(cfg, fields)

    raw = load_feature_long(fields, dates=dates, cfg=cfg)
    return _long_to_eval_dataframe(raw)
