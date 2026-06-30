"""IC / ICIR / quantile returns / long-short evaluation metrics."""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Pearson IC = corr(y_pred, y_true)."""
    mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    t = y_true[mask]
    p = y_pred[mask]
    if t.std(ddof=1) < 1e-12 or p.std(ddof=1) < 1e-12:
        return np.nan
    return float(np.corrcoef(p, t)[0, 1])


def compute_rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Rank IC = Spearman corr(y_pred, y_true)."""
    mask = y_true.notna() & y_pred.notna() & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    corr, _ = spearmanr(y_pred[mask], y_true[mask], nan_policy="omit")
    return float(corr)


def compute_icir(ic_series: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC)."""
    s = ic_series.dropna()
    if len(s) < 2 or s.std(ddof=1) < 1e-12:
        return np.nan
    return float(s.mean() / s.std(ddof=1))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    t, p = y_true[mask], y_pred[mask]
    if t.size == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    resid = p - t
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "mse": float(np.mean(resid ** 2)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": r2,
    }


def cross_section_metrics(
    panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
    *,
    min_obs: int = 10,
    n_quantiles: int = 10,
    long_pct: float = 0.2,
    short_pct: float = 0.2,
) -> pd.DataFrame:
    """Per-(date, minute) IC, rank IC, quantile returns, and long-short return."""
    eval_df = panel[["date", "minute", "ticker", label_col]].copy()
    eval_df["y_pred"] = pred
    eval_df["y_true"] = eval_df[label_col]
    rows: list[dict] = []

    for (date, minute), grp in eval_df.groupby(["date", "minute"], sort=False):
        n = grp["y_pred"].notna().sum()
        if n < min_obs:
            continue
        ic = compute_ic(grp["y_true"], grp["y_pred"])
        rank_ic = compute_rank_ic(grp["y_true"], grp["y_pred"])

        grp = grp.sort_values("y_pred")
        n_stocks = len(grp)
        n_long = max(1, int(n_stocks * long_pct))
        n_short = max(1, int(n_stocks * short_pct))
        long_ret = grp["y_true"].iloc[-n_long:].mean()
        short_ret = grp["y_true"].iloc[:n_short].mean()
        ls_ret = long_ret - short_ret

        try:
            grp = grp.copy()
            grp["quantile"] = pd.qcut(
                grp["y_pred"].rank(method="first"),
                q=n_quantiles,
                labels=False,
                duplicates="drop",
            )
            q_rets = grp.groupby("quantile")["y_true"].mean()
            q_dict = {f"q{int(q)+1}_ret": float(v) for q, v in q_rets.items()}
        except ValueError:
            q_dict = {}

        rows.append({
            "date": str(date),
            "minute": int(minute),
            "ic": ic,
            "rankic": rank_ic,
            "n_obs": int(n),
            "long_short_ret": ls_ret,
            "benchmark_ret": grp["y_true"].mean(),
            **q_dict,
        })

    if not rows:
        warnings.warn("No cross-sections passed min_obs filter.", stacklevel=2)
        return pd.DataFrame()
    return pd.DataFrame(rows)


def daily_aggregate(minute_metrics: pd.DataFrame) -> pd.DataFrame:
    if minute_metrics.empty:
        return pd.DataFrame()
    g = minute_metrics.groupby("date", sort=True)
    out = pd.DataFrame({
        "ic_mean": g["ic"].mean(),
        "ic_std": g["ic"].std(ddof=1),
        "rankic_mean": g["rankic"].mean(),
        "rankic_std": g["rankic"].std(ddof=1),
        "long_short_ret_mean": g["long_short_ret"].mean(),
        "benchmark_ret_mean": g["benchmark_ret"].mean(),
        "n_minutes": g.size(),
    }).reset_index()
    out["icir"] = out["ic_mean"] / out["ic_std"]
    out["rankicir"] = out["rankic_mean"] / out["rankic_std"]
    return out


def evaluate_predictions(
    panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
    *,
    normalized_label: np.ndarray | None = None,
    min_obs: int = 10,
    annualization_periods: int = 39 * 252,
) -> dict:
    """Full evaluation bundle: minute metrics, daily metrics, regression, quantiles."""
    minute_metrics = cross_section_metrics(panel, pred, label_col, min_obs=min_obs)
    daily_metrics = daily_aggregate(minute_metrics)

    reg = regression_metrics(
        normalized_label if normalized_label is not None else panel[label_col].to_numpy(),
        pred,
    )

    quantile_cols = [c for c in minute_metrics.columns if c.startswith("q") and c.endswith("_ret")]
    quantile_returns = pd.DataFrame()
    if quantile_cols:
        quantile_returns = minute_metrics[quantile_cols].mean().to_frame("mean_ret").reset_index()
        quantile_returns.columns = ["quantile", "mean_ret"]
        quantile_returns["annualized_ret"] = quantile_returns["mean_ret"] * annualization_periods

    summary = {
        "mean_ic": float(daily_metrics["ic_mean"].mean()) if not daily_metrics.empty else np.nan,
        "icir": compute_icir(daily_metrics["ic_mean"]) if not daily_metrics.empty else np.nan,
        "mean_rank_ic": float(daily_metrics["rankic_mean"].mean()) if not daily_metrics.empty else np.nan,
        "rankicir": compute_icir(daily_metrics["rankic_mean"]) if not daily_metrics.empty else np.nan,
        **reg,
    }
    return {
        "summary": summary,
        "minute_metrics": minute_metrics,
        "daily_metrics": daily_metrics,
        "quantile_returns": quantile_returns,
    }


def build_predictions_df(
    panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
) -> pd.DataFrame:
    meta = panel[["date", "minute", "ticker", "stock_code"]].copy()
    meta["datetime"] = panel["datetime"] if "datetime" in panel.columns else pd.NaT
    meta["y_pred"] = pred
    meta["y_true"] = panel[label_col].to_numpy()
    return meta
