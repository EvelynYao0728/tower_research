#!/usr/bin/env python3
"""Generate 2026 simple_factors from public Quote_2026 + Trade_2026.

Examples
--------
# 补全所有 Quote_2026 中尚未写入 simple_factors 的日期
python -m base_data_process.run_simple_factors --all

# 指定日期，8 线程并行 ticker 计算
python -m base_data_process.run_simple_factors --dates 20260102 20260105 -w 8

# 强制重跑
python -m base_data_process.run_simple_factors --dates 20260102 --overwrite -w 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from base_data_process.compute_simple_factors import compute_one_day, write_day
from base_data_process.config import OUTPUT_DIR, QUOTE_DIR_2026, TRADE_DIR_2026


def _default_workers() -> int:
    return max(1, min(16, (os.cpu_count() or 4)))


def _list_quote_dates(quote_dir: Path) -> list[str]:
    return sorted(p.stem for p in quote_dir.glob("*.parquet"))


def _pending_dates(dates: list[str], output_dir: Path, overwrite: bool) -> list[str]:
    if overwrite:
        return dates
    return [d for d in dates if not (output_dir / f"{d}.parquet").is_file()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build 2026 simple_factors from TAQ quote+trade")
    p.add_argument("--dates", nargs="*", default=None, help="YYYYMMDD 列表")
    p.add_argument("--all", action="store_true", help="处理 Quote_2026 全部日期")
    p.add_argument(
        "--quote-dir",
        type=Path,
        default=QUOTE_DIR_2026,
        help="Quote parquet 目录",
    )
    p.add_argument(
        "--trade-dir",
        type=Path,
        default=TRADE_DIR_2026,
        help="Trade parquet 目录",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="输出目录（默认 data/simple_factors）",
    )
    p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=_default_workers(),
        help="ticker 级多线程并行度（默认 min(16, CPU 核数)）",
    )
    p.add_argument("--overwrite", action="store_true", help="覆盖已有 parquet")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.all:
        dates = _list_quote_dates(args.quote_dir)
    elif args.dates:
        dates = [d.replace("-", "") for d in args.dates]
    else:
        print("[error] 请指定 --dates 或 --all", file=sys.stderr)
        return 1

    pending = _pending_dates(dates, args.output, args.overwrite)
    if not pending:
        print(f"[done] 无需处理，{len(dates)} 天均已在 {args.output}")
        return 0

    ok = skip = fail = 0
    t0 = time.perf_counter()
    print(f"[info] tick_workers={args.workers} pending_days={len(pending)}")
    for date_str in tqdm(pending, desc="days", unit="day", disable=args.no_progress):
        quote_path = args.quote_dir / f"{date_str}.parquet"
        trade_path = args.trade_dir / f"{date_str}.parquet"
        out_path = args.output / f"{date_str}.parquet"

        if not quote_path.is_file() or not trade_path.is_file():
            skip += 1
            print(f"[skip] {date_str}: missing quote or trade", file=sys.stderr)
            continue

        try:
            df = compute_one_day(quote_path, trade_path, tick_workers=args.workers)
            if df.is_empty():
                skip += 1
                print(f"[skip] {date_str}: empty result", file=sys.stderr)
                continue
            write_day(df, out_path)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[fail] {date_str}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(
        f"\n[done] ok={ok} skip={skip} fail={fail} "
        f"elapsed={elapsed:.1f}s output={args.output}"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
