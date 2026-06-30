"""Publication-style figures for LGBM strategy evaluation (English labels)."""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Academic palette (aligned with model/linear/plot.py)
COLORS = {
    "primary": "#2C3E50",
    "accent": "#C0392B",
    "secondary": "#7F8C8D",
    "highlight": "#2980B9",
    "positive": "#27AE60",
    "negative": "#E74C3C",
    "grid": "#BDC3C7",
    "missing": "#ECF0F1",
}
CATEGORY_COLORS = ["#2980B9", "#C0392B", "#27AE60", "#E67E22", "#8E44AD", "#16A085", "#7F8C8D"]
DPI = 300

FACTOR_CATEGORY_MAP: dict[str, str] = {
    "B4": "Trade Flow", "B5": "Trade Flow",
    "C3": "Quote", "C5": "Quote", "C8": "Quote",
    "D3": "NBBO", "D5": "NBBO",
    "E6": "Microstructure",
    "F4": "Micro", "F5": "Micro", "F11": "Micro", "F12": "Micro",
    "N1": "New Alpha", "N7": "New Alpha", "N11": "New Alpha",
    "clv": "Momentum", "clv_x_imb": "Momentum", "ret_1m": "Momentum",
    "ret_5m": "Momentum", "ret_10m_past": "Momentum",
    "dist_from_5m_high": "Momentum", "dist_from_5m_low": "Momentum",
    "range_pos_5m": "Momentum", "open_mean_dev": "Momentum",
    "ofi_1m": "Order Flow", "ofi_5m_avg": "Order Flow", "count_ofi": "Order Flow",
    "avg_trade_size": "Order Flow", "vwap_mid_dev": "Order Flow",
    "imbalance": "Imbalance", "imb_": "Imbalance",
    "spread": "Spread/Liq", "liq_": "Spread/Liq", "vol_adj": "Spread/Liq",
}


@dataclass
class TestCoverage:
    """Metadata for annotating partial test-period coverage on figures."""
    expected_start: str = "20260101"
    expected_end: str = "20260430"
    actual_start: str = ""
    actual_end: str = ""
    n_days: int = 0
    bottleneck_factors: list[str] | None = None


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
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _to_dt(date_str: str) -> pd.Timestamp:
    s = str(date_str).replace("-", "")
    return pd.to_datetime(s, format="%Y%m%d")


def _coverage_subtitle(coverage: TestCoverage | None) -> str:
    if coverage is None or not coverage.actual_start:
        return ""
    exp = f"{_to_dt(coverage.expected_start):%Y-%m-%d} to {_to_dt(coverage.expected_end):%Y-%m-%d}"
    act = f"{_to_dt(coverage.actual_start):%Y-%m-%d} to {_to_dt(coverage.actual_end):%Y-%m-%d}"
    if coverage.actual_end >= coverage.expected_end.replace("-", ""):
        return f"Test period: {act} ({coverage.n_days} trading days)"
    return (
        f"Actual data: {act} ({coverage.n_days} days) | "
        f"Target window: {exp} — partial coverage"
    )


def _format_date_axis(ax: plt.Axes, start: str, end: str) -> None:
    ax.set_xlim(_to_dt(start), _to_dt(end))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def _shade_missing_data(ax: plt.Axes, coverage: TestCoverage | None) -> None:
    """Shade the target-but-missing portion of the test window."""
    if coverage is None:
        return
    actual_end = _to_dt(coverage.actual_end)
    expected_end = _to_dt(coverage.expected_end)
    if actual_end >= expected_end:
        return
    ax.axvspan(actual_end, expected_end, color=COLORS["missing"], alpha=0.55, zorder=0)
    mid = actual_end + (expected_end - actual_end) / 2
    ax.text(
        mid, 0.02, "Missing data",
        transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=9, color=COLORS["secondary"], style="italic",
    )


def _add_title_block(fig: plt.Figure, title: str, coverage: TestCoverage | None = None) -> None:
    subtitle = _coverage_subtitle(coverage)
    if subtitle:
        fig.suptitle(title, fontsize=14, fontweight="bold", color=COLORS["primary"], y=0.98)
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10, color=COLORS["secondary"])
    else:
        fig.suptitle(title, fontsize=14, fontweight="bold", color=COLORS["primary"], y=0.97)


def _factor_category(name: str) -> str:
    for prefix, cat in FACTOR_CATEGORY_MAP.items():
        if name.startswith(prefix) or prefix in name:
            return cat
    return "Other"


def plot_daily_ic_timeseries(
    daily_metrics: pd.DataFrame,
    out_path: Path,
    *,
    icir: float | None = None,
    ic_col: str = "rankic_mean",
    coverage: TestCoverage | None = None,
) -> None:
    _apply_style()
    df = daily_metrics.sort_values("date").copy()
    df["date_dt"] = pd.to_datetime(df["date"].astype(str))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.97)

    _shade_missing_data(ax, coverage)
    ax.plot(
        df["date_dt"], df[ic_col],
        color=COLORS["highlight"], alpha=0.35, linewidth=0.9, label="Daily Rank IC",
    )
    roll = df[ic_col].rolling(20, min_periods=3).mean()
    ax.plot(
        df["date_dt"], roll,
        color=COLORS["accent"], linewidth=2.0, label="20-day moving average",
    )
    ax.axhline(0, color=COLORS["primary"], linestyle="--", linewidth=0.9, alpha=0.7)

    title = "Daily Rank IC — Out-of-Sample Test"
    if icir is not None and np.isfinite(icir):
        title += f"  (Rank ICIR = {icir:.3f})"
    _add_title_block(fig, title, coverage)

    if coverage and coverage.expected_start:
        _format_date_axis(ax, coverage.expected_start, coverage.expected_end)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Rank IC")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y")
    _save(fig, out_path)


def plot_ic_distribution(
    train_daily: pd.DataFrame,
    test_daily: pd.DataFrame,
    out_path: Path,
    ic_col: str = "rankic_mean",
    coverage: TestCoverage | None = None,
) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.0))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.10, right=0.97)

    test_ic = test_daily[ic_col].dropna()
    n_bins = min(20, max(8, len(test_ic) // 2))
    sns.histplot(
        test_ic, kde=True, ax=ax, color=COLORS["highlight"],
        alpha=0.65, bins=n_bins, edgecolor="white", linewidth=0.4,
    )
    train_mean = train_daily[ic_col].mean()
    test_mean = test_ic.mean()
    test_std = test_ic.std(ddof=1)
    pos_ratio = (test_ic > 0).mean()
    ax.axvline(train_mean, color=COLORS["highlight"], linestyle="--", linewidth=1.4,
               label=f"CV validation mean = {train_mean:.4f}")
    ax.axvline(test_mean, color=COLORS["accent"], linestyle="-", linewidth=1.6,
               label=f"Test mean = {test_mean:.4f}")
    ax.axvline(0, color=COLORS["secondary"], linestyle=":", linewidth=0.9)

    stats = f"μ={test_mean:.4f}  σ={test_std:.4f}  P(IC>0)={pos_ratio:.1%}  N={len(test_ic)} days"
    _add_title_block(fig, f"Daily Rank IC Distribution — Test Set\n{stats}", coverage)
    ax.set_xlabel("Daily Rank IC")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, out_path)


def plot_quantile_returns(quantile_returns: pd.DataFrame, out_path: Path) -> None:
    _apply_style()
    df = quantile_returns.copy()
    if df.empty:
        return
    if "quantile" in df.columns:
        df["q_num"] = df["quantile"].astype(str).str.extract(r"q(\d+)_ret", expand=False).astype(float)
        df = df.sort_values("q_num")
        df["q_label"] = "Q" + df["q_num"].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ycol = "annualized_ret" if "annualized_ret" in df.columns else "mean_ret"
    bars = ax.bar(df["q_label"], df[ycol], color=COLORS["highlight"], alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, df[ycol]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2%}" if abs(val) < 1 else f"{val:.3f}",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=8,
        )
    ax.axhline(0, color=COLORS["primary"], linestyle="--", linewidth=0.9)
    ax.set_title("Decile Portfolio Returns (Annualized, Test Set)", fontweight="bold", pad=12)
    ax.set_xlabel("Prediction Decile (Q1 = lowest score, Q10 = highest)")
    ax.set_ylabel("Annualized 10-min Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}" if abs(y) < 1 else f"{y:.2f}"))
    _save(fig, out_path)


def plot_longshort_cumulative(
    daily_metrics: pd.DataFrame,
    out_path: Path,
    coverage: TestCoverage | None = None,
) -> None:
    _apply_style()
    df = daily_metrics.sort_values("date").copy()
    df["date_dt"] = pd.to_datetime(df["date"].astype(str))
    df["ls_cum"] = df["long_short_ret_mean"].cumsum()
    df["bench_cum"] = df["benchmark_ret_mean"].cumsum()

    fig, ax = plt.subplots(figsize=(11, 5.0))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.97)

    _shade_missing_data(ax, coverage)
    ax.plot(df["date_dt"], df["ls_cum"], color=COLORS["accent"], linewidth=2.2,
            label="Long-Short (Top 20% − Bottom 20%, equal-weight)")
    ax.plot(df["date_dt"], df["bench_cum"], color=COLORS["positive"], linewidth=1.6,
            linestyle="--", label="Equal-weight benchmark")
    ax.axhline(0, color=COLORS["secondary"], linestyle=":", linewidth=0.9)
    _add_title_block(fig, "Cumulative Returns — Out-of-Sample Test", coverage)
    if coverage and coverage.expected_start:
        _format_date_axis(ax, coverage.expected_start, coverage.expected_end)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2%}"))
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, out_path)


def plot_feature_importance(
    factor_cols: list[str],
    importance: np.ndarray,
    out_path: Path,
    top_n: int = 30,
) -> None:
    _apply_style()
    df = pd.DataFrame({"factor": factor_cols, "gain": importance})
    df["category"] = df["factor"].map(_factor_category)
    df = df.sort_values("gain", ascending=True).tail(top_n)
    cats = sorted(df["category"].unique())
    cat_colors = {c: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, c in enumerate(cats)}

    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.32 * len(df))), constrained_layout=True)
    colors = [cat_colors[c] for c in df["category"]]
    ax.barh(df["factor"], df["gain"], color=colors, alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.set_title(f"Feature Importance by Gain (Top {top_n})", fontweight="bold", pad=12)
    ax.set_xlabel("Gain")
    ax.grid(True, axis="x")
    handles = [Patch(facecolor=cat_colors[c], label=c) for c in cats]
    ax.legend(handles=handles, title="Factor Category", loc="lower right", fontsize=9)
    _save(fig, out_path)


def plot_monthly_ic_heatmap(
    daily_metrics: pd.DataFrame,
    out_path: Path,
    ic_col: str = "rankic_mean",
    coverage: TestCoverage | None = None,
) -> None:
    _apply_style()
    df = daily_metrics.copy()
    df["date_dt"] = pd.to_datetime(df["date"].astype(str))
    df["year"] = df["date_dt"].dt.year
    df["month"] = df["date_dt"].dt.month

    # Build full grid for 2025 (train CV) + 2026 test months 1-4
    observed = df.groupby(["month", "year"])[ic_col].mean()
    years = sorted(df["year"].unique())
    if coverage:
        years = sorted(set(years) | {int(coverage.expected_end[:4])})
    months = list(range(1, 13))
    if coverage and int(coverage.expected_end[:4]) in years:
        months = list(range(1, 5))  # Jan–Apr for test year focus when single year

    idx = pd.MultiIndex.from_product([months, years], names=["month", "year"])
    pivot = observed.reindex(idx).unstack("year")

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(pivot.columns)), 6), constrained_layout=True)
    mask = pivot.isna()
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Mean Daily Rank IC", "shrink": 0.85},
        mask=mask,
    )
    # Mark missing cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if mask.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, "N/A", ha="center", va="center",
                        fontsize=8, color=COLORS["secondary"])

    ax.set_title("Monthly Mean Rank IC", fontweight="bold", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    ax.set_yticklabels([calendar.month_abbr[int(m)] for m in pivot.index], rotation=0)

    if coverage and int(coverage.expected_end[:4]) in pivot.columns:
        col_idx = list(pivot.columns).index(int(coverage.expected_end[:4]))
        ax.add_patch(plt.Rectangle(
            (col_idx, 0), 1, len(pivot),
            fill=False, edgecolor=COLORS["accent"], linewidth=2, linestyle="--",
        ))
    _save(fig, out_path)


def plot_lag_decay(decay_df: pd.DataFrame, out_path: Path) -> None:
    _apply_style()
    fig, ax1 = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax2 = ax1.twinx()

    x = decay_df["lag"].to_numpy()
    ax1.plot(x, decay_df["mean_ic"], color=COLORS["highlight"], marker="o",
             markersize=7, linewidth=2.2, label="Mean Rank IC")
    ax2.bar(x, decay_df["icir"], color=COLORS["accent"], alpha=0.45, width=0.35, label="ICIR")

    for _, row in decay_df.iterrows():
        ax1.annotate(
            f"{row['ic_decay_ratio']:.0%}",
            (row["lag"], row["mean_ic"]),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9,
            color=COLORS["primary"],
        )

    ax1.set_xlabel("Execution Lag (minutes)")
    ax1.set_ylabel("Mean Rank IC", color=COLORS["highlight"])
    ax2.set_ylabel("ICIR", color=COLORS["accent"])
    ax1.set_title("Information Decay under Execution Lag", fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.axhline(0, color=COLORS["secondary"], linestyle=":", linewidth=0.8)
    ax1.grid(True, axis="y", alpha=0.35)
    _save(fig, out_path)


def plot_fold_stability(
    fold_daily_ics: dict[int, pd.DataFrame],
    out_path: Path,
    ic_col: str = "rankic_mean",
) -> None:
    _apply_style()
    records = []
    for fold_id, daily in fold_daily_ics.items():
        for v in daily[ic_col].dropna():
            records.append({"fold": f"Fold {fold_id}", "rank_ic": v})
    df = pd.DataFrame(records)
    order = [f"Fold {i}" for i in sorted(fold_daily_ics.keys())]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    sns.boxplot(
        data=df, x="fold", y="rank_ic", order=order, ax=ax,
        color=COLORS["highlight"], width=0.45, fliersize=3,
        boxprops={"alpha": 0.55}, whiskerprops={"color": COLORS["primary"]},
    )
    sns.stripplot(
        data=df, x="fold", y="rank_ic", order=order, ax=ax,
        color=COLORS["primary"], alpha=0.25, size=2.5, jitter=0.25,
    )
    ax.axhline(0, color=COLORS["secondary"], linestyle="--", linewidth=0.9)
    ax.set_title("Cross-Validation Stability — Daily Rank IC by Fold", fontweight="bold", pad=12)
    ax.set_xlabel("Expanding Window Fold")
    ax.set_ylabel("Daily Rank IC")
    _save(fig, out_path)


def generate_all_figures(
    *,
    test_daily: pd.DataFrame,
    train_daily: pd.DataFrame | None,
    quantile_returns: pd.DataFrame,
    decay_df: pd.DataFrame,
    fold_daily_ics: dict[int, pd.DataFrame],
    factor_cols: list[str],
    importance: np.ndarray,
    figures_dir: Path,
    test_icir: float | None = None,
    coverage: TestCoverage | None = None,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_daily_ic_timeseries(
        test_daily, figures_dir / "fig1_ic_timeseries.png",
        icir=test_icir, coverage=coverage,
    )
    plot_ic_distribution(
        train_daily if train_daily is not None else test_daily,
        test_daily,
        figures_dir / "fig2_ic_distribution.png",
        coverage=coverage,
    )
    if not quantile_returns.empty:
        plot_quantile_returns(quantile_returns, figures_dir / "fig3_quantile_returns.png")
    plot_longshort_cumulative(test_daily, figures_dir / "fig4_longshort_returns.png", coverage=coverage)
    plot_feature_importance(factor_cols, importance, figures_dir / "fig5_feature_importance.png")
    plot_monthly_ic_heatmap(test_daily, figures_dir / "fig6_monthly_ic_heatmap.png", coverage=coverage)
    if not decay_df.empty:
        plot_lag_decay(decay_df, figures_dir / "fig7_lag_decay.png")
    if fold_daily_ics:
        plot_fold_stability(fold_daily_ics, figures_dir / "fig8_fold_stability.png")
