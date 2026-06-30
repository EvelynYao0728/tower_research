#!/usr/bin/env python3
"""Train a cross-sectional Ridge linear model on 因子库_final and evaluate on label."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DEFAULT_PANEL_CACHE, LinearModelConfig, default_workers  # noqa: E402
from linear.evaluate import build_evaluation_report, save_evaluation_outputs  # noqa: E402
from linear.plot import generate_all_plots  # noqa: E402
from linear.trainer import save_model_artifacts, train_linear_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Ridge linear model: all factors → forward return label",
    )
    p.add_argument("--factor-root", type=Path, default=LinearModelConfig().factor_root)
    p.add_argument("--label-root", type=Path, default=LinearModelConfig().label_root)
    p.add_argument("--trade-date-csv", type=Path, default=LinearModelConfig().trade_date_csv)
    p.add_argument("--registry", type=Path, default=LinearModelConfig().registry)
    p.add_argument("--output", type=Path, default=LinearModelConfig().output_dir)
    p.add_argument("--label-col", default="ex_log_ret_10m")
    p.add_argument("--train-ratio", type=float, default=0.60)
    p.add_argument("--valid-ratio", type=float, default=0.20)
    p.add_argument("--max-days", type=int, default=None, help="Limit days for quick runs")
    p.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help=f"Parallel day loaders (default: min(16, cpu_count()) = {default_workers()})",
    )
    p.add_argument("--skip-eval", action="store_true", help="Skip IC/decile evaluation")
    p.add_argument("--skip-plots", action="store_true", help="Skip figure generation")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Preprocessed panel cache directory (default: model/cache/panels)",
    )
    p.add_argument("--no-cache", action="store_true", help="Disable preprocessed panel cache")
    p.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild panel cache even when sources are unchanged",
    )
    p.add_argument("--session-start", type=int, default=931)
    p.add_argument("--session-end", type=int, default=1559)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = LinearModelConfig(
        factor_root=args.factor_root,
        label_root=args.label_root,
        trade_date_csv=args.trade_date_csv,
        registry=args.registry,
        output_dir=args.output,
        label_col=args.label_col,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        max_days=args.max_days,
        workers=args.workers if args.workers is not None else default_workers(),
        skip_eval=args.skip_eval,
        skip_plots=args.skip_plots,
        show_progress=not args.no_progress,
        cache_dir=None if args.no_cache else (args.cache_dir or DEFAULT_PANEL_CACHE),
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        session_start=args.session_start,
        session_end=args.session_end,
    )

    t0 = time.perf_counter()
    print("=" * 60)
    print("Linear Model Training")
    print("=" * 60)
    print(f"Factors : {cfg.factor_root}")
    print(f"Label   : {cfg.label_root} [{cfg.label_col}]")
    print(f"Output  : {cfg.output_dir}")

    trained, panels = train_linear_model(cfg)
    print(f"\n[train] {len(trained.train_dates)} days, {trained.train_rows:,} rows")
    print(f"[valid] {len(trained.valid_dates)} days, {trained.valid_rows:,} rows")
    print(f"[test ] {len(trained.test_dates)} days, {trained.test_rows:,} rows")
    print(f"[model] Ridge alpha = {trained.alpha:.6g}, factors = {len(trained.factor_cols)}")

    save_model_artifacts(trained, cfg.output_dir)

    report = None
    if not cfg.skip_eval:
        report = build_evaluation_report(
            panels=panels,
            model=trained.model,
            factor_cols=trained.factor_cols,
            label_col=trained.label_col,
            n_deciles=cfg.n_deciles,
            annualization_days=cfg.annualization_days,
            show_progress=cfg.show_progress,
        )
        save_evaluation_outputs(report, cfg.output_dir)

    if not cfg.skip_plots:
        if report is None:
            report = build_evaluation_report(
                panels=panels,
                model=trained.model,
                factor_cols=trained.factor_cols,
                label_col=trained.label_col,
                n_deciles=cfg.n_deciles,
                annualization_days=cfg.annualization_days,
                show_progress=cfg.show_progress,
            )
        coef_df = pd.read_csv(cfg.output_dir / "coefficients.csv")
        generate_all_plots(
            report=report,
            coef_df=coef_df,
            panels=panels,
            model=trained.model,
            factor_cols=trained.factor_cols,
            label_col=trained.label_col,
            output_dir=cfg.output_dir,
            n_deciles=cfg.n_deciles,
            show_progress=cfg.show_progress,
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "factor_cols": trained.factor_cols,
                "label_col": trained.label_col,
                "alpha": trained.alpha,
                "train_dates": trained.train_dates,
                "valid_dates": trained.valid_dates,
                "test_dates": trained.test_dates,
                **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    elapsed = time.perf_counter() - t0
    if report is not None:
        print("\n========== Evaluation Summary ==========")
        print(report["combined_summary"].to_string(index=False))
    else:
        skipped = []
        if cfg.skip_eval:
            skipped.append("evaluation")
        if cfg.skip_plots:
            skipped.append("plots")
        if skipped:
            print(f"\nSkipped: {', '.join(skipped)}")
    print(f"\nArtifacts saved to: {cfg.output_dir}")
    if not cfg.skip_plots:
        print(f"Figures saved to:   {cfg.output_dir / 'figures'}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
