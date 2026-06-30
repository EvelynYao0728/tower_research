"""Quantitative evaluation metrics for cross-sectional return forecasts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_RESEARCH = Path(__file__).resolve().parents[2]
if str(_RESEARCH / "backtest") not in sys.path:
    sys.path.insert(0, str(_RESEARCH / "backtest"))

from single_factor_bt.metrics import cross_section_day, summarize  # noqa: E402

KEYS = ("date", "sym_root", "sym_suffix", "minute")

def _rowwise_pearson(pred: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    p = pred[mask]
    t = y[mask]
    if p.std(ddof=1) < 1e-12 or t.std(ddof=1) < 1e-12:
        return np.nan
    return float(np.corrcoef(p, t)[0, 1])


def _rowwise_spearman(pred: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    corr, _ = spearmanr(pred[mask], y[mask], nan_policy="omit")
    return float(corr)


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(y)
    p = pred[mask]
    t = y[mask]
    if p.size == 0:
        return {
            "mse": np.nan, "rmse": np.nan, "r2": np.nan,
            "mae": np.nan, "hit_rate": np.nan,
        }
    resid = p - t
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "mse": float(np.mean(resid ** 2)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": r2,
        "mae": float(np.mean(np.abs(resid))),
        "hit_rate": float(np.mean(np.sign(p) == np.sign(t))),
    }


def cross_section_metrics(
    panel: pd.DataFrame,
    pred_col: str,
    label_col: str,
    n_deciles: int = 10,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute per-(date, minute) IC / RankIC and decile portfolio returns."""
    rows: list[pd.DataFrame] = []
    grouped = panel.groupby(["date", "minute"], sort=False)
    iterator = tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Cross-section metrics",
        unit="slice",
        disable=not show_progress or grouped.ngroups == 0,
    )
    for (date, minute), grp in iterator:
        pred = grp[pred_col].to_numpy(dtype=np.float64)
        y = grp[label_col].to_numpy(dtype=np.float64)
        ic = _rowwise_pearson(pred, y)
        rankic = _rowwise_spearman(pred, y)
        n_obs = int((np.isfinite(pred) & np.isfinite(y)).sum())

        tickers = grp["ticker"].astype(str).to_numpy()
        pred_mat = pred.reshape(1, -1)
        y_mat = y.reshape(1, -1)
        minute_arr = np.array([int(minute)], dtype=np.int32)
        day_metrics = cross_section_day(
            minute_arr,
            pred_mat.astype(np.float32),
            y_mat.astype(np.float32),
            n_groups=n_deciles,
        )
        day_metrics.insert(0, "date", str(date))
        day_metrics["ic"] = ic
        day_metrics["rankic"] = rankic
        day_metrics["n_obs"] = n_obs
        rows.append(day_metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def daily_aggregate(minute_metrics: pd.DataFrame) -> pd.DataFrame:
    if minute_metrics.empty:
        return pd.DataFrame()
    g = minute_metrics.groupby("date", sort=True)
    out = pd.DataFrame(
        {
            "ic_mean": g["ic"].mean(),
            "ic_std": g["ic"].std(ddof=1),
            "rankic_mean": g["rankic"].mean(),
            "rankic_std": g["rankic"].std(ddof=1),
            "long_short_ret_mean": g["long_short_ret"].mean(),
            "n_minutes": g.size(),
        }
    ).reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["icir"] = out["ic_mean"] / out["ic_std"]
        out["rankicir"] = out["rankic_mean"] / out["rankic_std"]
    return out


def evaluate_split(
    panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
    split_name: str,
    n_deciles: int,
    annualization_days: int,
    show_progress: bool = True,
) -> dict[str, object]:
    eval_panel = panel[list(KEYS) + ["ticker", label_col]].copy()
    eval_panel["prediction"] = pred.astype(np.float32)
    minute_metrics = cross_section_metrics(
        eval_panel,
        "prediction",
        label_col,
        n_deciles=n_deciles,
        show_progress=show_progress,
    )
    daily_metrics = daily_aggregate(minute_metrics)
    reg = regression_metrics(pred, eval_panel[label_col].to_numpy())
    quant_summary = summarize(minute_metrics, annualization_days=annualization_days)

    summary = {
        "split": split_name,
        "n_rows": int(len(pred)),
        "n_minutes": int(len(minute_metrics)),
        "n_days": int(minute_metrics["date"].nunique()) if not minute_metrics.empty else 0,
        **reg,
        **{k: float(quant_summary.iloc[0][k]) for k in quant_summary.columns},
    }
    return {
        "summary": summary,
        "minute_metrics": minute_metrics,
        "daily_metrics": daily_metrics,
    }


def build_evaluation_report(
    panels: dict[str, pd.DataFrame],
    model,
    factor_cols: Sequence[str],
    label_col: str,
    n_deciles: int,
    annualization_days: int,
    show_progress: bool = True,
) -> dict[str, object]:
    report: dict[str, object] = {"splits": {}, "combined_summary": []}
    split_iter = tqdm(
        panels.items(),
        desc="Evaluate splits",
        unit="split",
        disable=not show_progress,
    )
    for split_name, panel in split_iter:
        split_iter.set_postfix(split=split_name)
        x, y, meta = _to_xy_with_meta(panel, factor_cols, label_col)
        pred = model.predict(x)
        del x
        eval_panel = meta.copy()
        eval_panel[label_col] = y
        split_report = evaluate_split(
            eval_panel,
            pred,
            label_col,
            split_name,
            n_deciles,
            annualization_days,
            show_progress=show_progress,
        )
        report["splits"][split_name] = split_report
        report["combined_summary"].append(split_report["summary"])

    report["combined_summary"] = pd.DataFrame(report["combined_summary"])
    return report


def _to_xy_with_meta(panel, factor_cols, label_col):
    from data import to_xy
    return to_xy(panel, factor_cols, label_col)


def save_evaluation_outputs(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df: pd.DataFrame = report["combined_summary"]
    summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)

    for split_name, split_report in report["splits"].items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_report["minute_metrics"].to_csv(
            split_dir / "minute_metrics.csv", index=False,
        )
        split_report["daily_metrics"].to_csv(
            split_dir / "daily_metrics.csv", index=False,
        )
