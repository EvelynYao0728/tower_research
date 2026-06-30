"""
按交易日 eval 因子（与模板 ``factor.py`` 语义一致，避免全年整表 ~5000 万行一次 ``groupby``）。

默认多进程按日计算后合并为 ``factor_output.parquet`` 长表格式。
环境变量：

* ``QUANTALPHA_FACTOR_EVAL_WORKERS``：并行进程数，默认 ``4``（硬上限 ``4``，更大值会被截断）
* ``QUANTALPHA_FACTOR_EVAL_BULK=1``：强制全年整表 eval（慢，仅调试）
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantaalpha.data.private_catalog import LONG_KEYS, PrivateDataConfig, load_feature_long
from quantaalpha.factors.coder.template_panel_loader import (
    _long_to_eval_dataframe,
    list_day_stems_for_fields,
    normalize_expr_dollar_fields,
)

# 与 LowEntropy 默认式等价的向量化快路径
_FAST_LOW_ENTROPY_NORM = re.sub(
    r"\s+",
    "",
    "(-1 * $imbalance_entropy) * SIGN(MEDIAN($mid_price_std) - $mid_price_std)",
)


def _norm_suffix_series(s: pd.Series) -> pd.Series:
    def one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "None"
        t = str(v).strip()
        return t if t else "None"

    return s.map(one) if hasattr(s, "map") else s


def extract_dollar_fields(expr: str) -> list[str]:
    expr_n = normalize_expr_dollar_fields(expr)
    return sorted({m.group(1) for m in re.finditer(r"\$([A-Za-z_][A-Za-z0-9_]*)", expr_n)})


def _norm_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr.strip())


def is_fast_low_entropy(expr: str) -> bool:
    return _norm_expr(expr) == _FAST_LOW_ENTROPY_NORM


def compile_eval_code(expr: str, columns: Sequence[str]) -> str:
    from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol

    parsed = parse_symbol(expr, list(columns))
    parsed = parse_expression(parsed)
    for col in columns:
        parsed = parsed.replace(col[1:], f"df['{col}']")
    return parsed


def eval_globals() -> dict:
    import quantaalpha.factors.coder.function_lib as fl

    g: dict = {}
    for key in dir(fl):
        if key.startswith("_"):
            continue
        g[key] = getattr(fl, key)
    return g


def _long_from_panel(df: pd.DataFrame, name: str) -> pd.DataFrame:
    sym = df["instrument"].astype(str).str.split(".", n=1, expand=True)
    sym_root = sym[0]
    sym_suffix = (
        sym[1]
        if sym.shape[1] > 1
        else pd.Series(["None"] * len(df), index=df.index)
    )
    sym_suffix = _norm_suffix_series(sym_suffix.fillna("None"))
    return pd.DataFrame(
        {
            "date": df["datetime"].dt.strftime("%Y%m%d").astype(int),
            "sym_root": sym_root.astype(str),
            "sym_suffix": sym_suffix,
            "minute": (df["datetime"].dt.hour * 100 + df["datetime"].dt.minute).astype(
                int
            ),
            name: df[name].astype("float32", copy=False),
        }
    )


def eval_one_day_panel(
    stem: str,
    expr: str,
    name: str,
    fields: Sequence[str],
    eval_code: str,
    cfg: PrivateDataConfig,
) -> pd.DataFrame:
    from quantaalpha.factors.coder.function_lib import clear_ts_panel_meta, set_ts_panel_meta

    raw = load_feature_long(fields, dates=[stem], cfg=cfg)
    if raw.empty:
        return pd.DataFrame()
    df = _long_to_eval_dataframe(raw)
    set_ts_panel_meta(df)
    try:
        g = eval_globals()
        g["df"] = df
        df[name] = eval(eval_code, g)  # noqa: S307
    finally:
        clear_ts_panel_meta()
    return _long_from_panel(df, name)


def fast_low_entropy_one_day(stem: str, name: str, cfg: PrivateDataConfig) -> pd.DataFrame:
    ent_path = cfg.per_feature_root / "imbalance_entropy" / f"{stem}.parquet"
    simp_path = cfg.legacy_panel_root / f"{stem}.parquet"
    if not ent_path.exists() or not simp_path.exists():
        return pd.DataFrame()

    ent = pd.read_parquet(ent_path, columns=list(LONG_KEYS) + ["imbalance_entropy"])
    simp = pd.read_parquet(simp_path, columns=list(LONG_KEYS) + ["mid_price_std"])

    vol = simp["mid_price_std"].to_numpy(dtype=np.float64)
    med = simp.groupby("minute", observed=True)["mid_price_std"].transform("median")
    sign = np.sign(med.to_numpy(dtype=np.float64) - vol)
    sign_df = simp[list(LONG_KEYS)].copy()
    sign_df["_sign"] = sign.astype(np.float32)

    merged = ent.merge(sign_df, on=list(LONG_KEYS), how="left", validate="many_to_one")
    merged[name] = (
        (-1.0 * merged["imbalance_entropy"].to_numpy(dtype=np.float64))
        * merged["_sign"].to_numpy(dtype=np.float64)
    ).astype(np.float32)

    out = merged[list(LONG_KEYS) + [name]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y%m%d").astype(
        int
    )
    return out


_DEFAULT_FACTOR_EVAL_WORKERS = 4
_MAX_FACTOR_EVAL_WORKERS = 4


def resolve_factor_eval_workers(workers: int | None = None) -> int:
    """解析 factor_eval 并行进程数（环境变量 ``QUANTALPHA_FACTOR_EVAL_WORKERS``，默认 4，硬上限 4）。"""
    if workers is not None and workers > 0:
        n = workers
    else:
        raw = os.environ.get("QUANTALPHA_FACTOR_EVAL_WORKERS", "").strip()
        if raw:
            n = max(1, int(raw))
        else:
            n = _DEFAULT_FACTOR_EVAL_WORKERS
    return min(max(1, n), _MAX_FACTOR_EVAL_WORKERS)


def _worker(
    stem: str,
    expr: str,
    name: str,
    fields: Tuple[str, ...],
    eval_code: Optional[str],
    use_fast: bool,
) -> Tuple[str, pd.DataFrame]:
    cfg = PrivateDataConfig()
    if use_fast:
        df = fast_low_entropy_one_day(stem, name, cfg)
    else:
        if not eval_code:
            raise ValueError("eval_code required")
        df = eval_one_day_panel(stem, expr, name, fields, eval_code, cfg)
    return stem, df


def compute_factor_long(
    expr: str,
    name: str,
    *,
    workers: int | None = None,
    panel_max_days: int | None = None,
) -> pd.DataFrame:
    """
    按日计算因子长表（date, sym_*, minute, <name>），语义与模板整表 eval 一致。
    """
    from quantaalpha.data.private_catalog import validate_factor_expression_fields

    ok, msg = validate_factor_expression_fields(expr)
    if not ok:
        raise ValueError(msg)

    fields = extract_dollar_fields(expr)
    if not fields:
        raise ValueError(f"表达式中无 $字段: {expr[:200]!r}")

    cfg = PrivateDataConfig()
    saved_panel_env = os.environ.get("QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS")
    try:
        if panel_max_days is not None:
            os.environ["QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS"] = (
                str(panel_max_days) if panel_max_days > 0 else "0"
            )
        stems = list_day_stems_for_fields(cfg, fields)
    finally:
        if panel_max_days is not None:
            if saved_panel_env is None:
                os.environ.pop("QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS", None)
            else:
                os.environ["QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS"] = saved_panel_env
    if not stems:
        raise ValueError("无可用交易日")

    use_fast = is_fast_low_entropy(expr)
    eval_code: Optional[str] = None
    if not use_fast:
        sample_cols = [f"${f}" for f in fields] + ["instrument", "datetime"]
        eval_code = compile_eval_code(expr, sample_cols)

    workers = resolve_factor_eval_workers(workers)
    os.environ["QUANTALPHA_FACTOR_EVAL_WORKERS"] = str(workers)

    parts: List[pd.DataFrame] = []
    t0 = time.perf_counter()

    if workers <= 1:
        for stem in stems:
            _, df = _worker(stem, expr, name, tuple(fields), eval_code, use_fast)
            if not df.empty:
                parts.append(df)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _worker, stem, expr, name, tuple(fields), eval_code, use_fast
                ): stem
                for stem in stems
            }
            for fut in as_completed(futs):
                _, df = fut.result()
                if not df.empty:
                    parts.append(df)

    if not parts:
        return pd.DataFrame(columns=list(LONG_KEYS) + [name])

    out = pd.concat(parts, ignore_index=True)
    elapsed = time.perf_counter() - t0
    mode = "fast" if use_fast else "eval"
    print(
        f"[factor_eval] mode={mode} days={len(parts)}/{len(stems)} "
        f"rows={len(out)} elapsed={elapsed:.1f}s workers={workers}"
    )
    return out


def compute_factor_long_bulk(expr: str, name: str) -> pd.DataFrame:
    """全年整表 eval（慢）。"""
    from quantaalpha.factors.coder.function_lib import clear_ts_panel_meta, set_ts_panel_meta
    from quantaalpha.factors.coder.template_panel_loader import load_frame_for_expression

    df = load_frame_for_expression(expr)
    set_ts_panel_meta(df)
    try:
        code = compile_eval_code(expr, list(df.columns))
        g = eval_globals()
        g["df"] = df
        df[name] = eval(code, g)  # noqa: S307
    finally:
        clear_ts_panel_meta()
    return _long_from_panel(df, name)


def calculate_factor_to_parquet(
    expr: str,
    name: str,
    output_path: str | Path = "factor_output.parquet",
    *,
    workers: int | None = None,
    panel_max_days: int | None = None,
) -> Path:
    """模板 ``factor.py`` 入口：写出 ``factor_output.parquet``。"""
    out = Path(output_path)
    if os.environ.get("QUANTALPHA_FACTOR_EVAL_BULK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        long_df = compute_factor_long_bulk(expr, name)
    else:
        long_df = compute_factor_long(
            expr, name, workers=workers, panel_max_days=panel_max_days
        )
    if long_df.empty:
        raise ValueError("因子结果为空")
    long_df.to_parquet(out, index=False)
    return out
