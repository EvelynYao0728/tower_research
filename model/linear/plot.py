"""Publication-style figures for linear model evaluation (English labels)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm

# Muted academic palette
COLORS = {
    "primary": "#2C3E50",
    "accent": "#C0392B",
    "secondary": "#7F8C8D",
    "highlight": "#2980B9",
    "positive": "#27AE60",
    "negative": "#E74C3C",
    "grid": "#BDC3C7",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["primary"],
            "axes.labelcolor": COLORS["primary"],
            "xtick.color": COLORS["primary"],
            "ytick.color": COLORS["primary"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.35,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_ic_distribution(
    minute_metrics: pd.DataFrame,
    out_path: Path,
    split_name: str = "test",
) -> None:
    _apply_style()
    ic = minute_metrics["ic"].dropna().to_numpy()
    ric = minute_metrics["rankic"].dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, data, title, color in [
        (axes[0], ic, "Pearson IC Distribution", COLORS["highlight"]),
        (axes[1], ric, "Spearman Rank IC Distribution", COLORS["accent"]),
    ]:
        if data.size == 0:
            ax.set_visible(False)
            continue
        bins = min(40, max(10, int(np.sqrt(data.size))))
        ax.hist(data, bins=bins, color=color, alpha=0.72, edgecolor="white", linewidth=0.6)
        mu = np.nanmean(data)
        ax.axvline(mu, color=COLORS["primary"], linestyle="--", linewidth=1.2, label=f"Mean = {mu:.4f}")
        ax.axvline(0.0, color=COLORS["secondary"], linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Frequency")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y")
    fig.suptitle(f"Cross-Sectional Information Coefficient — {split_name.capitalize()} Set", y=1.02)
    _save(fig, out_path)


def plot_cumulative_ic(
    daily_metrics: pd.DataFrame,
    out_path: Path,
    split_name: str = "test",
) -> None:
    _apply_style()
    df = daily_metrics.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"].astype(str))
    cum_ic = df["ic_mean"].cumsum()
    cum_ric = df["rankic_mean"].cumsum()

    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    ax.plot(dates, cum_ic, color=COLORS["highlight"], linewidth=1.4, label="Cumulative IC")
    ax.plot(dates, cum_ric, color=COLORS["accent"], linewidth=1.4, label="Cumulative Rank IC")
    ax.axhline(0.0, color=COLORS["secondary"], linestyle=":", linewidth=0.9)
    ax.set_title(f"Cumulative Information Coefficient — {split_name.capitalize()} Set")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Sum")
    ax.legend(loc="best")
    ax.grid(True)
    fig.autofmt_xdate(rotation=30, ha="right")
    _save(fig, out_path)


def plot_daily_ic_timeseries(
    daily_metrics: pd.DataFrame,
    out_path: Path,
    split_name: str = "test",
    window: int = 5,
) -> None:
    _apply_style()
    df = daily_metrics.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"].astype(str))
    ic = df["ic_mean"].to_numpy()
    roll = pd.Series(ic).rolling(window, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    ax.bar(dates, ic, width=0.8, color=COLORS["highlight"], alpha=0.35, label="Daily IC")
    ax.plot(dates, roll, color=COLORS["primary"], linewidth=1.6, label=f"{window}-Day Moving Avg")
    ax.axhline(0.0, color=COLORS["secondary"], linestyle=":", linewidth=0.9)
    ax.set_title(f"Daily Cross-Sectional IC — {split_name.capitalize()} Set")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.legend(loc="best")
    ax.grid(True, axis="y")
    fig.autofmt_xdate(rotation=30, ha="right")
    _save(fig, out_path)


def plot_decile_returns(
    minute_metrics: pd.DataFrame,
    out_path: Path,
    split_name: str = "test",
    n_deciles: int = 10,
) -> None:
    _apply_style()
    decile_cols = [f"decile_{k}_ret" for k in range(1, n_deciles + 1)]
    means = [minute_metrics[c].mean() for c in decile_cols if c in minute_metrics.columns]
    if not means:
        return
    x = np.arange(1, len(means) + 1)
    colors = [
        COLORS["negative"] if v < 0 else COLORS["positive"] for v in means
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.bar(x, means, color=colors, alpha=0.82, edgecolor="white", linewidth=0.6)
    ax.axhline(0.0, color=COLORS["secondary"], linestyle=":", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{k}" for k in x])
    ax.set_title(f"Mean Return by Prediction Decile — {split_name.capitalize()} Set")
    ax.set_xlabel("Decile (D1 = lowest prediction, D10 = highest)")
    ax.set_ylabel("Mean Forward Return")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(True, axis="y")
    _save(fig, out_path)


def plot_coefficients(
    coef_df: pd.DataFrame,
    out_path: Path,
    top_n: int = 20,
) -> None:
    _apply_style()
    df = coef_df.copy().sort_values("coefficient", key=np.abs, ascending=True).tail(top_n)
    colors = [
        COLORS["positive"] if v >= 0 else COLORS["negative"] for v in df["coefficient"]
    ]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.28 * len(df))), constrained_layout=True)
    y = np.arange(len(df))
    ax.barh(y, df["coefficient"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["factor"])
    ax.axvline(0.0, color=COLORS["secondary"], linestyle=":", linewidth=0.9)
    ax.set_title("Ridge Regression Coefficients (Standardized Features)")
    ax.set_xlabel("Coefficient")
    ax.grid(True, axis="x")
    _save(fig, out_path)


def plot_pred_vs_actual(
    panel: pd.DataFrame,
    pred: np.ndarray,
    label_col: str,
    out_path: Path,
    split_name: str = "test",
    n_bins: int = 20,
) -> None:
    _apply_style()
    y = panel[label_col].to_numpy(dtype=np.float64)
    p = pred.astype(np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size < n_bins * 5:
        return

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return
    bin_idx = np.digitize(p, edges[1:-1], right=True)
    bin_pred, bin_actual, bin_se = [], [], []
    for b in range(edges.size - 1):
        m = bin_idx == b
        if m.sum() < 5:
            continue
        bin_pred.append(p[m].mean())
        bin_actual.append(y[m].mean())
        bin_se.append(y[m].std(ddof=1) / np.sqrt(m.sum()))

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    x = np.arange(len(bin_pred))
    ax.errorbar(
        x, bin_actual, yerr=bin_se, fmt="o-", color=COLORS["highlight"],
        ecolor=COLORS["secondary"], elinewidth=1.0, capsize=3, markersize=5,
        label="Binned mean actual return ± SE",
    )
    ax2 = ax.twinx()
    ax2.plot(x, bin_pred, "s--", color=COLORS["accent"], markersize=4, alpha=0.85, label="Binned mean prediction")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{k+1}" for k in x], rotation=45, ha="right")
    ax.set_title(f"Prediction Calibration — {split_name.capitalize()} Set")
    ax.set_xlabel("Prediction Quantile Bin")
    ax.set_ylabel("Mean Actual Return")
    ax2.set_ylabel("Mean Prediction")
    ax.grid(True, axis="y")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    _save(fig, out_path)


def plot_residual_analysis(
    pred: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    split_name: str = "test",
) -> None:
    _apply_style()
    mask = np.isfinite(pred) & np.isfinite(y)
    resid = pred[mask] - y[mask]
    if resid.size < 20:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].hist(resid, bins=40, color=COLORS["highlight"], alpha=0.72, edgecolor="white", linewidth=0.5)
    axes[0].axvline(0.0, color=COLORS["secondary"], linestyle=":", linewidth=0.9)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Prediction − Actual")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, axis="y")

    sorted_resid = np.sort(resid)
    n = sorted_resid.size
    theoretical = np.linspace(-3, 3, n)
    axes[1].scatter(theoretical, sorted_resid, s=8, alpha=0.45, color=COLORS["primary"], edgecolors="none")
    lim = max(np.abs(theoretical).max(), np.abs(sorted_resid).max())
    axes[1].plot([-lim, lim], [-lim, lim], color=COLORS["accent"], linestyle="--", linewidth=1.0)
    axes[1].set_title("Residual Q–Q Plot (Normal Reference)")
    axes[1].set_xlabel("Theoretical Quantile")
    axes[1].set_ylabel("Sample Quantile")
    axes[1].grid(True)
    fig.suptitle(f"Residual Diagnostics — {split_name.capitalize()} Set", y=1.02)
    _save(fig, out_path)


def plot_metrics_summary_card(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    _apply_style()
    test_row = summary_df.loc[summary_df["split"] == "test"]
    if test_row.empty:
        test_row = summary_df.tail(1)
    row = test_row.iloc[0]

    metrics = [
        ("IC", row.get("IC", np.nan)),
        ("ICIR", row.get("ICIR", np.nan)),
        ("Rank IC", row.get("RankIC", np.nan)),
        ("Rank ICIR", row.get("RankICIR", np.nan)),
        ("R²", row.get("r2", np.nan)),
        ("RMSE", row.get("rmse", np.nan)),
        ("Hit Rate", row.get("hit_rate", np.nan)),
        ("L/S Return (ann.)", row.get("long_short_ret", np.nan)),
        ("Sharpe (ann.)", row.get("sharpe", np.nan)),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.axis("off")
    y = 0.92
    ax.text(0.5, 0.98, "Linear Model — Test Set Performance Summary", ha="center", va="top",
            fontsize=14, fontweight="bold", color=COLORS["primary"])
    for name, val in metrics:
        txt = f"{name}: {val:.4f}" if np.isfinite(val) else f"{name}: N/A"
        ax.text(0.08, y, txt, fontsize=12, color=COLORS["primary"])
        y -= 0.09
    _save(fig, out_path)


def generate_all_plots(
    report: dict,
    coef_df: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    model,
    factor_cols: list[str],
    label_col: str,
    output_dir: Path,
    n_deciles: int = 10,
    show_progress: bool = True,
) -> None:
    fig_dir = output_dir / "figures"
    plot_tasks = [
        ("coefficients", lambda: plot_coefficients(coef_df, fig_dir / "coefficients.png")),
        (
            "metrics summary",
            lambda: plot_metrics_summary_card(
                report["combined_summary"], fig_dir / "metrics_summary.png",
            ),
        ),
    ]
    for split_name in ("valid", "test"):
        if split_name not in report["splits"]:
            continue
        split = report["splits"][split_name]
        minute_metrics = split["minute_metrics"]
        daily_metrics = split["daily_metrics"]
        prefix = fig_dir / split_name
        panel = panels[split_name]
        x, y, meta = _predict_xy(panel, factor_cols, label_col)
        pred = model.predict(x)
        eval_panel = meta.copy()
        eval_panel[label_col] = y
        plot_tasks.extend(
            [
                (
                    f"{split_name} ic distribution",
                    lambda p=prefix, m=minute_metrics, s=split_name: plot_ic_distribution(
                        m, p / "ic_distribution.png", s,
                    ),
                ),
                (
                    f"{split_name} cumulative ic",
                    lambda p=prefix, d=daily_metrics, s=split_name: plot_cumulative_ic(
                        d, p / "cumulative_ic.png", s,
                    ),
                ),
                (
                    f"{split_name} daily ic",
                    lambda p=prefix, d=daily_metrics, s=split_name: plot_daily_ic_timeseries(
                        d, p / "daily_ic.png", s,
                    ),
                ),
                (
                    f"{split_name} decile returns",
                    lambda p=prefix, m=minute_metrics, s=split_name: plot_decile_returns(
                        m, p / "decile_returns.png", s, n_deciles,
                    ),
                ),
                (
                    f"{split_name} calibration",
                    lambda p=prefix, ep=eval_panel, pr=pred, s=split_name: plot_pred_vs_actual(
                        ep, pr, label_col, p / "calibration.png", s,
                    ),
                ),
                (
                    f"{split_name} residuals",
                    lambda p=prefix, pr=pred, yy=y, s=split_name: plot_residual_analysis(
                        pr, yy, p / "residuals.png", s,
                    ),
                ),
            ]
        )

    for name, fn in tqdm(
        plot_tasks,
        desc="Generate figures",
        unit="fig",
        disable=not show_progress,
    ):
        tqdm.write(f"  plotting: {name}")
        fn()


def _predict_xy(panel, factor_cols, label_col):
    from data import to_xy
    return to_xy(panel, factor_cols, label_col)
