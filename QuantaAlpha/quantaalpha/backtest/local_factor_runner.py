#!/usr/bin/env python3
"""本地因子计算 + research/backtest（无 LLM）。实现见 ``factors.coder.factor_eval``。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quantaalpha.backtest.research_bt import (
    DEFAULT_RESEARCH_ROOT,
    backtest_output_artifact_paths,
    default_paths,
    run_backtest,
)
from quantaalpha.factors.coder.factor_eval import compute_factor_long


def main(argv=None) -> int:
    import os

    ap = argparse.ArgumentParser(description="本地 factor_eval + 回测")
    ap.add_argument("--expr", default="(-1 * $imbalance_entropy) * SIGN(MEDIAN($mid_price_std) - $mid_price_std)")
    ap.add_argument("--name", default="LowEntropy_LowVol_Signal")
    ap.add_argument("--factor-dir", type=Path, default=None)
    ap.add_argument("--bt-output", type=Path, default=None)
    ap.add_argument("-j", "--workers", type=int, default=0)
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--only-backtest", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    research = DEFAULT_RESEARCH_ROOT
    factor_dir = (args.factor_dir or research / "data" / args.name).resolve()
    bt_out = (args.bt_output or research / "backtest" / "output").resolve()
    workers = args.workers or None

    if not args.only_backtest:
        if factor_dir.is_dir() and any(factor_dir.glob("*.parquet")):
            print(f"[calc] skip: {factor_dir} already has shards (use --only-backtest)")
        else:
            long_df = compute_factor_long(args.expr, args.name, workers=workers)
            factor_dir.mkdir(parents=True, exist_ok=True)
            for dkey, g in long_df.groupby("date"):
                stem = str(int(dkey))
                g.to_parquet(factor_dir / f"{stem}.parquet", index=False)
            print(f"[calc] wrote {long_df['date'].nunique()} days -> {factor_dir}")

    if args.skip_backtest:
        return 0

    bt = run_backtest(
        factor_dir,
        factor_col=args.name,
        output_dir=bt_out,
        label_col="ex_log_ret_10m",
        trade_date_csv=default_paths()["trade_date_csv"]
        if default_paths()["trade_date_csv"].is_file()
        else None,
        use_cache=not args.no_cache,
    )
    print(bt.summary.to_string())
    print(backtest_output_artifact_paths(bt.output_dir, args.name)["summary_csv"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
