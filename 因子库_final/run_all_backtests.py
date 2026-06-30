#!/usr/bin/env python3
"""批量回测因子库_final 中所有因子，结果写入 output/。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent
_RESEARCH = _ROOT.parent
if str(_RESEARCH / "backtest") not in sys.path:
    sys.path.insert(0, str(_RESEARCH / "backtest"))

from single_factor_bt.engine import run_backtest  # noqa: E402

DEFAULT_LABEL = _RESEARCH / "data" / "label"
DEFAULT_TRADE_DATE = _RESEARCH / "data" / "trade_date.csv"
DEFAULT_OUTPUT = _ROOT / "output"
REGISTRY = _ROOT / "factor_registry.yaml"


def _load_factor_names(registry: Path) -> list[str]:
    with registry.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [item["name"] for item in data["factors"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="批量回测因子库_final")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--trade-date-csv", type=Path, default=DEFAULT_TRADE_DATE)
    parser.add_argument("-w", "--workers", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--factors",
        nargs="*",
        default=None,
        help="仅回测指定因子目录名；默认 registry 中全部",
    )
    args = parser.parse_args()

    names = args.factors or _load_factor_names(REGISTRY)
    trade_csv = args.trade_date_csv if args.trade_date_csv.is_file() else None

    summaries = []
    for name in names:
        factor_dir = _ROOT / name
        if not factor_dir.is_dir():
            print(f"[skip] 目录不存在: {factor_dir}")
            continue
        print(f"\n{'=' * 60}\n[run] {name}\n{'=' * 60}")
        summary, _ = run_backtest(
            factor_path=factor_dir,
            label_path=args.label,
            output_dir=args.output,
            factor_col=name,
            trade_date_csv=trade_csv,
            workers=args.workers,
            use_cache=not args.no_cache,
        )
        summaries.append(summary)

    if summaries:
        import pandas as pd

        all_df = pd.concat(summaries, ignore_index=True)
        print("\n========== 因子库_final 回测汇总 ==========")
        print(all_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
