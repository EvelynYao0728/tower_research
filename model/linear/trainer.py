"""Train a regularized cross-sectional linear model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from config import LinearModelConfig
from data import (
    available_dates,
    discover_factor_names,
    load_and_split_panels,
    load_trade_dates,
    split_dates,
    to_xy,
)


@dataclass
class TrainedLinearModel:
    model: Ridge
    factor_cols: list[str]
    label_col: str
    alpha: float
    train_dates: list[str]
    valid_dates: list[str]
    test_dates: list[str]
    train_rows: int
    valid_rows: int
    test_rows: int


def _center_xy(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    return x - x_mean, y - y_mean, x_mean, y_mean


def _ridge_solve_from_gram(
    gram: np.ndarray,
    xy: np.ndarray,
    alpha: float,
) -> np.ndarray:
    n_features = gram.shape[0]
    reg = gram + alpha * np.eye(n_features, dtype=gram.dtype)
    return np.linalg.solve(reg, xy)


def _ridge_predict(
    x: np.ndarray,
    weights: np.ndarray,
    x_mean: np.ndarray,
    y_mean: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    intercept = y_mean - float(x_mean @ weights)
    return x @ weights + intercept


def _make_ridge_model(
    weights: np.ndarray,
    x_mean: np.ndarray,
    y_mean: float,
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.coef_ = weights.astype(np.float64, copy=False)
    model.intercept_ = float(y_mean - x_mean @ weights)
    return model


def _fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    alphas: tuple[float, ...],
    show_progress: bool = True,
) -> tuple[Ridge, float]:
    """Select alpha on validation IC via precomputed Gram matrix, then refit."""
    x_c, y_c, x_mean, y_mean = _center_xy(x_train, y_train)
    gram = x_c.T @ x_c
    xy = x_c.T @ y_c

    best_alpha = alphas[0]
    best_score = -np.inf
    alpha_iter = tqdm(
        alphas,
        desc="Ridge alpha search",
        disable=not show_progress,
    )
    for alpha in alpha_iter:
        weights = _ridge_solve_from_gram(gram, xy, alpha)
        pred = _ridge_predict(x_valid, weights, x_mean, y_mean)
        score = _mean_rank_ic(pred, y_valid)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_alpha = alpha
        alpha_iter.set_postfix(alpha=f"{alpha:.0g}", rank_ic=f"{score:.4f}")

    x_all = np.vstack([x_train, x_valid])
    y_all = np.concatenate([y_train, y_valid])
    x_c, y_c, x_mean, y_mean = _center_xy(x_all, y_all)
    gram = x_c.T @ x_c
    xy = x_c.T @ y_c
    weights = _ridge_solve_from_gram(gram, xy, best_alpha)
    return _make_ridge_model(weights, x_mean, y_mean, best_alpha), float(best_alpha)


def _mean_rank_ic(pred: np.ndarray, y: np.ndarray) -> float:
    if pred.size < 3:
        return np.nan
    from scipy.stats import spearmanr
    corr, _ = spearmanr(pred, y, nan_policy="omit")
    return float(corr)


def train_linear_model(cfg: LinearModelConfig) -> tuple[TrainedLinearModel, dict[str, pd.DataFrame]]:
    """Load data, fit Ridge, and return model plus train/valid/test panels."""
    factor_cols = discover_factor_names(cfg.factor_root, cfg.registry)
    if not factor_cols:
        raise ValueError(f"No factor directories found under {cfg.factor_root}")

    calendar_dates = load_trade_dates(cfg.trade_date_csv)
    all_dates = available_dates(
        cfg.factor_root,
        cfg.label_root,
        factor_cols,
        calendar_dates,
        show_progress=cfg.show_progress,
    )
    if not all_dates:
        raise ValueError("No overlapping dates across label and all factor files.")
    if cfg.max_days is not None:
        all_dates = all_dates[: cfg.max_days]
    train_dates, valid_dates, test_dates = split_dates(
        all_dates, cfg.train_ratio, cfg.valid_ratio,
    )

    common_kwargs = dict(
        factor_root=cfg.factor_root,
        label_root=cfg.label_root,
        factor_cols=factor_cols,
        label_col=cfg.label_col,
        winsorize_q=cfg.winsorize_quantile,
        session_start=cfg.session_start,
        session_end=cfg.session_end,
        workers=cfg.workers,
        show_progress=cfg.show_progress,
        cache_dir=cfg.cache_dir,
        use_cache=cfg.use_cache,
        refresh_cache=cfg.refresh_cache,
    )

    panels = load_and_split_panels(
        train_dates,
        valid_dates,
        test_dates,
        **common_kwargs,
    )

    train_panel = panels["train"]
    valid_panel = panels["valid"]
    test_panel = panels["test"]

    x_train, y_train, _ = to_xy(train_panel, factor_cols, cfg.label_col)
    x_valid, y_valid, _ = to_xy(valid_panel, factor_cols, cfg.label_col)
    model, alpha = _fit_ridge(
        x_train,
        y_train,
        x_valid,
        y_valid,
        cfg.ridge_alphas,
        show_progress=cfg.show_progress,
    )

    trained = TrainedLinearModel(
        model=model,
        factor_cols=factor_cols,
        label_col=cfg.label_col,
        alpha=alpha,
        train_dates=train_dates,
        valid_dates=valid_dates,
        test_dates=test_dates,
        train_rows=len(y_train),
        valid_rows=len(y_valid),
        test_rows=len(to_xy(test_panel, factor_cols, cfg.label_col)[1]),
    )
    return trained, panels


def save_model_artifacts(trained: TrainedLinearModel, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    coef_df = pd.DataFrame(
        {
            "factor": trained.factor_cols,
            "coefficient": trained.model.coef_.astype(float),
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coef_df.to_csv(output_dir / "coefficients.csv", index=False)

    meta = pd.DataFrame(
        [
            {
                "label_col": trained.label_col,
                "alpha": trained.alpha,
                "intercept": float(trained.model.intercept_),
                "n_factors": len(trained.factor_cols),
                "train_rows": trained.train_rows,
                "valid_rows": trained.valid_rows,
                "test_rows": trained.test_rows,
                "train_start": trained.train_dates[0],
                "train_end": trained.train_dates[-1],
                "valid_start": trained.valid_dates[0],
                "valid_end": trained.valid_dates[-1],
                "test_start": trained.test_dates[0],
                "test_end": trained.test_dates[-1],
            }
        ]
    )
    meta.to_csv(output_dir / "model_meta.csv", index=False)
