"""LightGBM training with early stopping and optional Optuna hyperparameter search."""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

_MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from data import to_xy  # noqa: E402
from label_processor import attach_training_label  # noqa: E402


@dataclass
class LGBParams:
    learning_rate: float = 0.025
    num_leaves: int = 95
    max_depth: int = -1
    min_child_samples: int = 650
    subsample: float = 0.8
    colsample_bytree: float = 0.65
    reg_alpha: float = 0.1
    reg_lambda: float = 3.0
    n_estimators: int = 2000
    early_stopping_rounds: int = 100


@dataclass
class TrainResult:
    model: lgb.LGBMRegressor
    best_iteration: int
    factor_cols: list[str]
    training_label_col: str
    params: LGBParams = field(default_factory=LGBParams)


def _lgb_regressor(params: LGBParams, n_estimators: int | None = None) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        max_depth=params.max_depth,
        min_child_samples=params.min_child_samples,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        n_estimators=n_estimators or params.n_estimators,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def train_lgbm(
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    params: LGBParams | None = None,
    *,
    winsorize_q: float = 0.01,
) -> TrainResult:
    """Train with early stopping on validation; label winsorized by minute (exp1)."""
    params = params or LGBParams()
    train_prep, train_label = attach_training_label(train_panel, label_col, winsorize_q=winsorize_q)
    valid_prep, _ = attach_training_label(valid_panel, label_col, winsorize_q=winsorize_q)

    x_train, y_train, _ = to_xy(train_prep, factor_cols, train_label)
    x_valid, y_valid, _ = to_xy(valid_prep, factor_cols, train_label)

    if len(y_train) == 0 or len(y_valid) == 0:
        raise ValueError("Empty train or validation set after label processing.")

    model = _lgb_regressor(params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(params.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = int(model.best_iteration_ or model.n_estimators)
    return TrainResult(model, best_iter, factor_cols, train_label, params)


def train_lgbm_full(
    panel: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    params: LGBParams,
    best_iteration: int,
    *,
    winsorize_q: float = 0.01,
) -> TrainResult:
    """Retrain on full data with fixed tree count (no validation split)."""
    prep, train_label = attach_training_label(panel, label_col, winsorize_q=winsorize_q)
    x, y, _ = to_xy(prep, factor_cols, train_label)
    model = _lgb_regressor(params, n_estimators=best_iteration)
    model.fit(x, y)
    return TrainResult(model, best_iteration, factor_cols, train_label, params)


def predict_panel(
    model: lgb.LGBMRegressor,
    panel: pd.DataFrame,
    factor_cols: list[str],
) -> np.ndarray:
    x = panel[factor_cols].to_numpy(dtype=np.float32)
    valid = np.all(np.isfinite(x), axis=1)
    pred = np.full(len(panel), np.nan, dtype=np.float64)
    if valid.any():
        pred[valid] = model.predict(x[valid])
    return pred


def save_model(result: TrainResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "best_iteration": result.best_iteration,
            "factor_cols": result.factor_cols,
            "training_label_col": result.training_label_col,
            "params": result.params,
        },
        path,
    )


def load_model(path: Path) -> TrainResult:
    data = joblib.load(path)
    return TrainResult(**data)


def tune_hyperparameters_optuna(
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    *,
    n_trials: int = 20,
    winsorize_q: float = 0.01,
) -> tuple[LGBParams, dict[str, float]]:
    """Optuna search on validation Rank IC; falls back to defaults if optuna unavailable."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        warnings.warn("Optuna not installed; using default LGBParams.", stacklevel=2)
        return LGBParams(), {}

    from evaluator import evaluate_predictions  # noqa: WPS433 — avoid circular at module level

    def objective(trial: "optuna.Trial") -> float:
        params = LGBParams(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.03),
            num_leaves=trial.suggest_int("num_leaves", 63, 127),
            min_child_samples=trial.suggest_int("min_child_samples", 500, 800, step=50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.7),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 5.0, log=True),
        )
        result = train_lgbm(
            train_panel, valid_panel, factor_cols, label_col, params, winsorize_q=winsorize_q,
        )
        pred = predict_panel(result.model, valid_panel, factor_cols)
        metrics = evaluate_predictions(valid_panel, pred, label_col, min_obs=10)
        mean_rank_ic = metrics["daily_metrics"]["rankic_mean"].mean()
        if not np.isfinite(mean_rank_ic):
            return -1.0
        return float(mean_rank_ic)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best_params = LGBParams(
        learning_rate=best["learning_rate"],
        num_leaves=best["num_leaves"],
        min_child_samples=best["min_child_samples"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        reg_lambda=best["reg_lambda"],
    )
    meta = {"best_val_rank_ic": float(study.best_value), **best}
    return best_params, meta
