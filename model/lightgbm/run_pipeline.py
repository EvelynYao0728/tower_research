#!/usr/bin/env python3
"""Main entry point: minute-frequency multi-factor LGBM strategy pipeline."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cv_splitter import build_expanding_folds  # noqa: E402
from data_loader import (  # noqa: E402
    DataPaths,
    TEST_END,
    TEST_START,
    diagnose_test_coverage,
    discover_all_factors,
    load_panel_for_dates,
    load_test_dates,
    load_train_dates,
    split_panel_by_date_list,
)
from evaluator import build_predictions_df, evaluate_predictions  # noqa: E402
from feature_engineer import create_lagged_features, get_factor_columns  # noqa: E402
from label_processor import attach_training_label  # noqa: E402
from trainer import (  # noqa: E402
    LGBParams,
    predict_panel,
    save_model,
    train_lgbm,
    train_lgbm_full,
    tune_hyperparameters_optuna,
)
from visualizer import TestCoverage, generate_all_figures  # noqa: E402

_RESEARCH = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output" / "lgbm_strategy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minute-frequency LGBM multi-factor strategy pipeline")
    p.add_argument("--data_path", type=Path, default=_RESEARCH, help="Research root (factor/label paths)")
    p.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--lags", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--gap_months", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_train_days", type=int, default=None, help="Limit train days for quick runs")
    p.add_argument("--max_test_days", type=int, default=None, help="Limit test days for quick runs")
    p.add_argument("--optuna_trials", type=int, default=20, help="Optuna trials (0=skip, default 20)")
    p.add_argument("--winsorize_q", type=float, default=0.01)
    p.add_argument("--no_cache", action="store_true")
    return p.parse_args()


def _build_paths(data_path: Path) -> DataPaths:
    return DataPaths(
        factor_root=data_path / "因子库_final",
        train_label_root=data_path / "data" / "label",
        test_label_root=data_path / "final" / "label",
        trade_date_csv=data_path / "data" / "trade_date.csv",
        registry=data_path / "因子库_final" / "factor_registry.yaml",
    )


def run_cv(
    train_panel: pd.DataFrame,
    folds: list,
    factor_cols: list[str],
    label_col: str,
    params: LGBParams,
    *,
    winsorize_q: float,
    models_dir: Path,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], int]:
    """Run expanding-window CV; return cv_results, fold daily ICs, median best_iteration."""
    cv_rows: list[dict] = []
    fold_daily_ics: dict[int, pd.DataFrame] = {}
    best_iters: list[int] = []

    for fold in folds:
        print(f"\n--- Fold {fold.fold_id}: train={len(fold.train_dates)}d, val={len(fold.val_dates)}d ---")
        train_sub = split_panel_by_date_list(train_panel, fold.train_dates)
        val_sub = split_panel_by_date_list(train_panel, fold.val_dates)

        result = train_lgbm(
            train_sub, val_sub, factor_cols, label_col, params, winsorize_q=winsorize_q,
        )
        save_model(result, models_dir / f"lgbm_fold{fold.fold_id}.pkl")
        best_iters.append(result.best_iteration)

        pred = predict_panel(result.model, val_sub, factor_cols)
        eval_out = evaluate_predictions(val_sub, pred, label_col)
        fold_daily_ics[fold.fold_id] = eval_out["daily_metrics"]
        s = eval_out["summary"]
        cv_rows.append({
            "fold": fold.fold_id,
            "train_days": len(fold.train_dates),
            "val_days": len(fold.val_dates),
            "best_iteration": result.best_iteration,
            "val_mean_ic": s["mean_ic"],
            "val_icir": s["icir"],
            "val_mean_rank_ic": s["mean_rank_ic"],
            "val_rankicir": s["rankicir"],
            "val_rmse": s["rmse"],
            "val_r2": s["r2"],
        })
        print(
            f"  Val Rank IC={s['mean_rank_ic']:.4f}, ICIR={s['rankicir']:.4f}, "
            f"best_iter={result.best_iteration}",
        )

    cv_df = pd.DataFrame(cv_rows)
    median_iter = int(np.median(best_iters))
    return cv_df, fold_daily_ics, median_iter


def run_lag_analysis(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    params: LGBParams,
    best_iteration: int,
    lags: list[int],
    *,
    winsorize_q: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for lag in lags:
        print(f"\n--- Lag analysis: lag={lag} ---")
        train_lag = create_lagged_features(train_panel, lag, factor_cols)
        test_lag = create_lagged_features(test_panel, lag, factor_cols)

        result = train_lgbm_full(
            train_lag, factor_cols, label_col, params, best_iteration, winsorize_q=winsorize_q,
        )
        pred = predict_panel(result.model, test_lag, factor_cols)
        eval_out = evaluate_predictions(test_lag, pred, label_col)
        s = eval_out["summary"]
        rows.append({
            "lag": lag,
            "mean_ic": s["mean_rank_ic"],
            "icir": s["rankicir"],
        })
        print(f"  Test Rank IC={s['mean_rank_ic']:.4f}, ICIR={s['rankicir']:.4f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        ref = df.loc[df["lag"] == df["lag"].min(), "mean_ic"].iloc[0]
        df["ic_decay_ratio"] = df["mean_ic"] / ref if ref != 0 else np.nan
    return df


def main() -> int:
    args = parse_args()
    use_cache = not args.no_cache
    paths = _build_paths(args.data_path)
    out = args.output_path
    models_dir = out / "models"
    results_dir = out / "results"
    figures_dir = out / "figures"
    for d in (models_dir, results_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LGBM Minute-Frequency Multi-Factor Strategy Pipeline")
    print("Label mode: winsorize only (exp1)")
    print("=" * 60)

    factor_cols = discover_all_factors(paths)
    print(f"Factors discovered: {len(factor_cols)}")

    train_dates = load_train_dates(paths, factor_cols)
    test_dates = load_test_dates(paths, factor_cols)
    if not test_dates:
        raise ValueError(
            f"No test dates found in {paths.test_label_root} for {TEST_START}~{TEST_END}. "
            "Check that final/label and factor files overlap for 2026.",
        )
    if args.max_train_days:
        train_dates = train_dates[: args.max_train_days]
    if args.max_test_days:
        test_dates = test_dates[: args.max_test_days]

    coverage_info = diagnose_test_coverage(paths, factor_cols, test_dates)
    if coverage_info["actual_days"] < coverage_info["label_days_in_range"]:
        print(
            f"\nWARNING: Test data coverage is partial.\n"
            f"  Target window : {TEST_START} ~ {TEST_END} "
            f"({coverage_info['label_days_in_range']} label days available)\n"
            f"  Actual overlap: {coverage_info['actual_start']} ~ {coverage_info['actual_end']} "
            f"({coverage_info['actual_days']} days)\n"
            f"  Bottleneck factors ({len(coverage_info['bottleneck_factors'])}): "
            f"{', '.join(coverage_info['bottleneck_factors'][:8])}"
            f"{'...' if len(coverage_info['bottleneck_factors']) > 8 else ''}"
        )

    print(f"Train dates: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Test dates : {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} days)")

    cache_dir = Path(__file__).resolve().parents[1] / "cache" / "panels"
    print("\nLoading train panel...")
    train_panel = load_panel_for_dates(
        train_dates, paths, factor_cols, paths.train_label_root,
        workers=args.workers, cache_dir=cache_dir, use_cache=use_cache,
    )
    print("Loading test panel...")
    test_panel = load_panel_for_dates(
        test_dates, paths, factor_cols, paths.test_label_root,
        workers=args.workers, cache_dir=cache_dir, use_cache=use_cache,
    )
    label_col = paths.label_col
    factor_cols = get_factor_columns(train_panel, label_col)
    print(f"Feature columns: {len(factor_cols)}")

    folds = build_expanding_folds(train_dates, n_folds=args.n_folds, gap_months=args.gap_months)
    params = LGBParams()

    if args.optuna_trials > 0 and folds:
        print(f"\nOptuna hyperparameter search ({args.optuna_trials} trials) on Fold 1...")
        f0 = folds[0]
        train_sub = split_panel_by_date_list(train_panel, f0.train_dates)
        val_sub = split_panel_by_date_list(train_panel, f0.val_dates)
        params, optuna_meta = tune_hyperparameters_optuna(
            train_sub, val_sub, factor_cols, label_col,
            n_trials=args.optuna_trials, winsorize_q=args.winsorize_q,
        )
        if optuna_meta:
            pd.DataFrame([optuna_meta]).to_csv(results_dir / "optuna_best.csv", index=False)
            print(
                f"  Best val Rank IC={optuna_meta.get('best_val_rank_ic', float('nan')):.4f} | "
                f"lr={params.learning_rate}, leaves={params.num_leaves}, "
                f"min_child={params.min_child_samples}, reg_lambda={params.reg_lambda:.3f}, "
                f"colsample={params.colsample_bytree:.3f}"
            )

    print("\n========== Expanding Window CV ==========")
    cv_df, fold_daily_ics, median_iter = run_cv(
        train_panel, folds, factor_cols, label_col, params,
        winsorize_q=args.winsorize_q, models_dir=models_dir,
    )
    cv_df.to_csv(results_dir / "cv_results.csv", index=False)
    print("\nCV Results:")
    print(cv_df.to_string(index=False))

    print("\n========== Final Model (Full 2025) ==========")
    final_result = train_lgbm_full(
        train_panel, factor_cols, label_col, params, median_iter,
        winsorize_q=args.winsorize_q,
    )
    save_model(final_result, models_dir / "lgbm_final.pkl")

    print("\n========== Test Evaluation (lag=0) ==========")
    test_pred = predict_panel(final_result.model, test_panel, factor_cols)
    test_eval = evaluate_predictions(test_panel, test_pred, label_col)

    test_prep, test_label = attach_training_label(
        test_panel, label_col, winsorize_q=args.winsorize_q,
    )
    norm_eval = evaluate_predictions(
        test_panel, test_pred, label_col, normalized_label=test_prep[test_label].to_numpy(),
    )
    if np.isfinite(norm_eval["summary"].get("r2", np.nan)) and abs(norm_eval["summary"]["r2"]) > 100:
        warnings.warn("Extreme R² detected — verify label normalization alignment.", stacklevel=2)

    test_daily = test_eval["daily_metrics"]
    test_daily.to_csv(results_dir / "test_daily_ic.csv", index=False)
    pred_df = build_predictions_df(test_panel, test_pred, label_col)
    pred_df.to_csv(results_dir / "test_predictions.csv", index=False)
    if not test_eval["quantile_returns"].empty:
        test_eval["quantile_returns"].to_csv(results_dir / "quantile_returns.csv", index=False)

    print("\n========== Lag Decay Analysis ==========")
    decay_df = run_lag_analysis(
        train_panel, test_panel, factor_cols, label_col, params, median_iter,
        args.lags, winsorize_q=args.winsorize_q,
    )
    decay_df.to_csv(results_dir / "lag_decay_analysis.csv", index=False)

    print("\n========== Generating Figures ==========")
    importance = final_result.model.booster_.feature_importance(importance_type="gain")
    train_daily_agg = pd.DataFrame()
    if fold_daily_ics:
        train_daily_agg = pd.concat(fold_daily_ics.values(), ignore_index=True)

    generate_all_figures(
        test_daily=test_daily,
        train_daily=train_daily_agg if not train_daily_agg.empty else None,
        quantile_returns=test_eval["quantile_returns"],
        decay_df=decay_df,
        fold_daily_ics=fold_daily_ics,
        factor_cols=factor_cols,
        importance=importance,
        figures_dir=figures_dir,
        test_icir=test_eval["summary"]["rankicir"],
        coverage=TestCoverage(
            expected_start=TEST_START,
            expected_end=TEST_END,
            actual_start=coverage_info["actual_start"],
            actual_end=coverage_info["actual_end"],
            n_days=coverage_info["actual_days"],
            bottleneck_factors=coverage_info["bottleneck_factors"],
        ),
    )

    pd.DataFrame([{
        "expected_start": coverage_info["expected_start"],
        "expected_end": coverage_info["expected_end"],
        "actual_start": coverage_info["actual_start"],
        "actual_end": coverage_info["actual_end"],
        "actual_days": coverage_info["actual_days"],
        "label_days_in_range": coverage_info["label_days_in_range"],
        "bottleneck_factors": ",".join(coverage_info["bottleneck_factors"]),
    }]).to_csv(results_dir / "test_coverage.csv", index=False)

    print("\n========== Summary ==========")
    s = test_eval["summary"]
    print(f"Test Mean Rank IC : {s['mean_rank_ic']:.4f}")
    print(f"Test Rank ICIR    : {s['rankicir']:.4f}")
    print(f"Test RMSE (winsorized label) : {norm_eval['summary']['rmse']:.6f}")
    print(f"Test R² (winsorized label)   : {norm_eval['summary']['r2']:.4f}")
    print(f"\nAll outputs saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
