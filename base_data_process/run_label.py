#!/usr/bin/env python3
"""Generate labels from simple_factors mid_price_last (10-row mid-to-mid forward return).

Examples
--------
# 验证 20250102 与 public/label 完全一致
python -m base_data_process.run_label --dates 20250102 \\
  --factors-dir /home/yzyao.25/research/data/simple_factors \\
  --compare-dir /home/yzyao.25/research/public/label

# 用 base_data_process/simple_factors 生成 2026 全部 label
python -m base_data_process.run_label --all \\
  --factors-dir /home/yzyao.25/research/base_data_process/simple_factors
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

from base_data_process.compute_label import compare_label_frames, compute_one_day, write_day
from base_data_process.config import (
    LABEL_OUTPUT_DIR,
    LABEL_REF_DIR,
    SIMPLE_FACTORS_DIR_2026,
)


def _list_factor_dates(factors_dir: Path) -> list[str]:
    return sorted(p.stem for p in factors_dir.glob("*.parquet"))


def _pending_dates(dates: list[str], output_dir: Path, overwrite: bool) -> list[str]:
    if overwrite:
        return dates
    return [d for d in dates if not (output_dir / f"{d}.parquet").is_file()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build labels from simple_factors mid_price_last (10-row mid-to-mid)"
    )
    p.add_argument("--dates", nargs="*", default=None, help="YYYYMMDD 列表")
    p.add_argument("--all", action="store_true", help="处理 factors-dir 下全部日期")
    p.add_argument(
        "--factors-dir",
        type=Path,
        default=SIMPLE_FACTORS_DIR_2026,
        help="simple_factors 目录（2026 默认 base_data_process/simple_factors）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=LABEL_OUTPUT_DIR,
        help="label 输出目录（默认 base_data_process/label）",
    )
    p.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="参考 label 目录；指定后对每个生成日做逐行比对",
    )
    p.add_argument("--overwrite", action="store_true", help="覆盖已有 parquet")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.all:
        dates = _list_factor_dates(args.factors_dir)
    elif args.dates:
        dates = [d.replace("-", "") for d in args.dates]
    else:
        print("[error] 请指定 --dates 或 --all", file=sys.stderr)
        return 1

    pending = _pending_dates(dates, args.output, args.overwrite)
    if not pending:
        print(f"[done] 无需处理，{len(dates)} 天均已在 {args.output}")
        return 0

    ok = skip = fail = compare_fail = 0
    t0 = time.perf_counter()
    print(f"[info] pending_days={len(pending)} factors={args.factors_dir}")

    for date_str in tqdm(pending, desc="label", unit="day", disable=args.no_progress):
        factors_path = args.factors_dir / f"{date_str}.parquet"
        out_path = args.output / f"{date_str}.parquet"

        if not factors_path.is_file():
            skip += 1
            print(f"[skip] {date_str}: missing {factors_path}", file=sys.stderr)
            continue

        try:
            df = compute_one_day(factors_path)
            if df.empty:
                skip += 1
                print(f"[skip] {date_str}: empty result", file=sys.stderr)
                continue

            if args.compare_dir is not None:
                ref_path = args.compare_dir / f"{date_str}.parquet"
                if not ref_path.is_file():
                    compare_fail += 1
                    print(f"[fail] {date_str}: missing reference {ref_path}", file=sys.stderr)
                    continue
                import pandas as pd

                ref = pd.read_parquet(ref_path)
                errs = compare_label_frames(ref, df)
                if errs:
                    compare_fail += 1
                    print(f"[fail] {date_str} compare:", file=sys.stderr)
                    for e in errs:
                        print(f"  - {e}", file=sys.stderr)
                    continue
                print(f"[ok] {date_str}: identical to {ref_path}")

            write_day(df, out_path)
            ok += 1
        except Exception as exc:
            fail += 1
            print(f"[fail] {date_str}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(
        f"\n[done] ok={ok} skip={skip} fail={fail} compare_fail={compare_fail} "
        f"elapsed={elapsed:.1f}s output={args.output}"
    )
    return 0 if fail == 0 and compare_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
