#!/usr/bin/env python3
"""Unified entry point: compute all 28 因子库_final factors.

Examples
--------
# 全量生成（断点续跑，跳过已有 parquet）
python 因子库_final/compute/run.py --all

# 指定日期 + 因子族
python 因子库_final/compute/run.py --dates 20260102 --groups bcde mb

# 仅 O 族
python 因子库_final/compute/run.py --dates 20260102 --groups o
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_LIBROOT = Path(__file__).resolve().parent.parent
if str(_LIBROOT) not in sys.path:
    sys.path.insert(0, str(_LIBROOT))

from compute.compute import factors_for_groups, generate_day, pending_dates  # noqa: E402
from compute.config import (  # noqa: E402
    ALL_FACTORS,
    FACTOR_GROUPS,
    LIBRARY_ROOT,
    QUOTE_DIR,
    SIMPLE_FACTORS_DIR,
    TRADE_DATE_CSV,
    TRADE_DIR,
)


def _list_dates(trade_date_csv: Path | None, quote_dir: Path, trade_dir: Path) -> list[str]:
    if trade_date_csv and trade_date_csv.is_file():
        df = pd.read_csv(trade_date_csv)
        col = "date" if "date" in df.columns else df.columns[0]
        return [str(d).replace("-", "") for d in df[col].tolist()]
    stems = sorted(
        {p.stem for p in trade_dir.glob("*.parquet")}
        & {p.stem for p in quote_dir.glob("*.parquet")}
    )
    return stems


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute 28 factors into 因子库_final")
    p.add_argument("--dates", nargs="*", default=None, help="YYYYMMDD 列表")
    p.add_argument("--all", action="store_true", help="trade_date.csv 中全部日期")
    p.add_argument(
        "--groups",
        nargs="*",
        choices=tuple(FACTOR_GROUPS.keys()),
        default=list(FACTOR_GROUPS.keys()),
        help="因子族：bcde / mb / o（默认全部）",
    )
    p.add_argument(
        "--factors",
        nargs="*",
        default=None,
        help="指定因子名（自动推断所需 groups）",
    )
    p.add_argument("-o", "--output", type=Path, default=LIBRARY_ROOT)
    p.add_argument("--sf-dir", type=Path, default=SIMPLE_FACTORS_DIR)
    p.add_argument("--quote-dir", type=Path, default=QUOTE_DIR)
    p.add_argument("--trade-dir", type=Path, default=TRADE_DIR)
    p.add_argument("--trade-date-csv", type=Path, default=TRADE_DATE_CSV)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tick-workers", type=int, default=4, help="O 族 tick 并行度")
    p.add_argument(
        "--tick-pool",
        choices=("process", "thread"),
        default="thread",
    )
    p.add_argument("--write-threads", type=int, default=4)
    p.add_argument("--no-progress", action="store_true")
    return p


def _resolve_groups(args: argparse.Namespace) -> tuple[str, ...]:
    if args.factors:
        unknown = set(args.factors) - set(ALL_FACTORS)
        if unknown:
            print(f"[error] 未知因子: {sorted(unknown)}", file=sys.stderr)
            raise SystemExit(1)
        groups: list[str] = []
        for g, names in FACTOR_GROUPS.items():
            if any(f in names for f in args.factors):
                groups.append(g)
        return tuple(groups)
    return tuple(args.groups)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    groups = _resolve_groups(args)

    if args.all:
        dates = _list_dates(args.trade_date_csv, args.quote_dir, args.trade_dir)
    elif args.dates:
        dates = [d.replace("-", "") for d in args.dates]
    else:
        print("[error] 请指定 --dates 或 --all", file=sys.stderr)
        return 1

    target_factors = factors_for_groups(groups)
    todo = pending_dates(dates, args.output, target_factors, overwrite=args.overwrite)
    if not todo:
        print(f"[done] 无需处理，{len(dates)} 天均已在 {args.output}")
        return 0

    ok = fail = skip = 0
    t0 = time.perf_counter()
    print(f"[info] groups={groups} pending={len(todo)} output={args.output}")

    for date_str in tqdm(todo, desc="factors", unit="day", disable=args.no_progress):
        if "mb" in groups:
            sf_path = args.sf_dir / f"{date_str}.parquet"
            if not sf_path.is_file() and groups == ("mb",):
                skip += 1
                print(f"[skip] {date_str}: missing {sf_path}", file=sys.stderr)
                continue
        try:
            paths = generate_day(
                date_str,
                args.output,
                sf_dir=args.sf_dir,
                quote_dir=args.quote_dir,
                trade_dir=args.trade_dir,
                groups=groups,
                ref_root=args.output,
                tick_workers=args.tick_workers,
                tick_pool=args.tick_pool,
                write_threads=args.write_threads,
            )
            if paths:
                ok += 1
            else:
                skip += 1
        except Exception as exc:
            fail += 1
            print(f"[fail] {date_str}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(f"\n[done] ok={ok} skip={skip} fail={fail} elapsed={elapsed:.1f}s")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
