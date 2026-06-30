#!/usr/bin/env python3
"""
单因子：低 entropy × 低波动（相对截面 median mid_price_std）

表达式（与 LLM 因子库一致）::
    (-1 * $imbalance_entropy) * SIGN(MEDIAN($mid_price_std) - $mid_price_std)

说明
----
* ``$imbalance_entropy``：日频指标（同一交易日内各分钟取值相同）
* ``MEDIAN($mid_price_std)``：每分钟截面上的 mid_price_std 中位数
* ``SIGN(median - vol)``：波动低于截面中位时为 +1，否则 -1
* 整体：低 entropy（更定向）且相对低波动时信号为正

用法
----
    cd QuantaAlpha
    source .venv/bin/activate   # 或已安装 quantaalpha 的环境
    python scripts/backtest_low_entropy_low_vol.py

    # 仅重跑回测（因子 parquet 已存在）
    python scripts/backtest_low_entropy_low_vol.py --only-backtest

    # 强制重算 + 回测
    python scripts/backtest_low_entropy_low_vol.py --force-calc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 项目根：scripts/ -> QuantaAlpha/
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

_env = _PROJECT_ROOT / ".env"
if _env.exists():
    load_dotenv(_env, override=True)

FACTOR_EXPR = "(-1 * $imbalance_entropy) * SIGN(MEDIAN($mid_price_std) - $mid_price_std)"
FACTOR_NAME = "low_entropy_low_vol"


def _write_daily_shards(long_df, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_days = 0
    for dkey, g in long_df.groupby("date"):
        stem = str(int(dkey))
        g.to_parquet(out_dir / f"{stem}.parquet", index=False)
        n_days += 1
    return n_days


def main(argv: list[str] | None = None) -> int:
    from quantaalpha.backtest.research_bt import (
        DEFAULT_RESEARCH_ROOT,
        backtest_output_artifact_paths,
        default_paths,
        run_backtest,
    )
    from quantaalpha.data.private_catalog import PrivateDataConfig
    from quantaalpha.factors.coder.factor_eval import compute_factor_long, resolve_factor_eval_workers

    ap = argparse.ArgumentParser(description="计算并回测 low_entropy_low_vol 单因子")
    ap.add_argument("--expr", default=FACTOR_EXPR, help="因子表达式")
    ap.add_argument("--name", default=FACTOR_NAME, help="因子列名 / 回测输出目录名")
    ap.add_argument(
        "--factor-dir",
        type=Path,
        default=None,
        help="日分片 parquet 输出目录（默认 research/data/<name>）",
    )
    ap.add_argument(
        "--bt-output",
        type=Path,
        default=None,
        help="回测图表/ summary 根目录（默认 research/backtest/output）",
    )
    ap.add_argument("-j", "--workers", type=int, default=0, help="并行计算进程数")
    ap.add_argument("--only-backtest", action="store_true", help="跳过计算，仅回测")
    ap.add_argument("--skip-backtest", action="store_true", help="仅计算，不回测")
    ap.add_argument("--force-calc", action="store_true", help="即使已有分片也重算")
    ap.add_argument("--no-cache", action="store_true", help="回测不使用缓存")
    args = ap.parse_args(argv)

    research = DEFAULT_RESEARCH_ROOT
    cfg = PrivateDataConfig()
    cfg.validate_roots_exist()

    factor_dir = (args.factor_dir or research / "data" / args.name).resolve()
    bt_out = (args.bt_output or research / "backtest" / "output").resolve()
    workers = resolve_factor_eval_workers(args.workers or None)

    has_shards = factor_dir.is_dir() and any(factor_dir.glob("*.parquet"))

    if not args.only_backtest:
        if has_shards and not args.force_calc:
            print(f"[calc] 跳过：{factor_dir} 已有日分片（加 --force-calc 重算）")
        else:
            print(f"[calc] 表达式: {args.expr}")
            print(f"[calc] 因子名: {args.name}  workers={workers}")
            long_df = compute_factor_long(args.expr, args.name, workers=workers)
            if long_df.empty:
                print("[calc] 错误：计算结果为空，请检查 imbalance_entropy / mid_price_std 数据路径")
                return 1
            n_days = _write_daily_shards(long_df, factor_dir)
            print(f"[calc] 完成：{n_days} 个交易日 -> {factor_dir}")

    if args.skip_backtest:
        return 0

    if not factor_dir.is_dir() or not any(factor_dir.glob("*.parquet")):
        print(f"[bt] 错误：未找到因子分片 {factor_dir}")
        return 1

    paths = default_paths()
    trade_csv = paths["trade_date_csv"] if paths["trade_date_csv"].is_file() else None

    print(f"[bt] 回测输入: {factor_dir}")
    print(f"[bt] 输出目录: {bt_out}")
    bt = run_backtest(
        factor_dir,
        factor_col=args.name,
        output_dir=bt_out,
        label_col="ex_log_ret_10m",
        trade_date_csv=trade_csv,
        use_cache=not args.no_cache,
    )

    print("\n=== summary ===")
    print(bt.summary.to_string(index=False))
    arts = backtest_output_artifact_paths(bt.output_dir, args.name)
    print(f"\n图表目录: {arts['factor_output_dir']}")
    print(f"summary.csv: {arts['summary_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
