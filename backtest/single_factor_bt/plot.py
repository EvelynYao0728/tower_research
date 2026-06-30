"""Matplotlib plots for backtest output."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_datetime(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        df["date"].astype(str)
        + df["minute"].astype(int).astype(str).str.zfill(4),
        format="%Y-%m-%d%H%M",
        errors="coerce",
    )


def _hhmm_to_str(m: int) -> str:
    m = int(m)
    return f"{m // 100:02d}:{m % 100:02d}"


# --------------------------------------------------------------------------- #
# Cross-section IC profile within a trading day                               #
# --------------------------------------------------------------------------- #
def intraday_ic_profile(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average per-minute IC / RankIC across all dates.

    Returns a DataFrame indexed by minute (HHMM int) with columns:
        ic_mean, ic_std, ic_n, ic_t,
        rankic_mean, rankic_std, rankic_n, rankic_t

    `ic_t` is the t-stat of the cross-day mean (mean / (std/√n)).
    """
    if "minute" not in metrics.columns:
        raise ValueError("metrics needs a 'minute' column")

    def _agg(col: str) -> pd.DataFrame:
        g = metrics.groupby("minute")[col]
        n = g.count().rename(f"{col}_n")
        mu = g.mean().rename(f"{col}_mean")
        sd = g.std(ddof=1).rename(f"{col}_std")
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (mu / (sd / np.sqrt(n.replace(0, np.nan)))).rename(f"{col}_t")
        return pd.concat([mu, sd, n, t], axis=1)

    out = pd.concat(
        [_agg("ic"), _agg("rankic")], axis=1
    ).sort_index()
    out.index.name = "minute"
    return out.reset_index()


def plot_intraday_ic(
    profile: pd.DataFrame, factor_name: str, out_path: Path,
) -> None:
    """Plot intraday IC / RankIC averaged across days, with ±1σ band."""
    df = profile.sort_values("minute").reset_index(drop=True)
    minutes = df["minute"].astype(int).to_numpy()
    x = np.arange(len(df))
    labels = [_hhmm_to_str(m) for m in minutes]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             constrained_layout=True)

    for ax, prefix, color, title in [
        (axes[0], "ic", "#1f77b4", "Cross-day Mean IC by Minute"),
        (axes[1], "rankic", "#ff7f0e", "Cross-day Mean RankIC by Minute"),
    ]:
        mu = df[f"{prefix}_mean"].to_numpy()
        sd = df[f"{prefix}_std"].to_numpy()
        n = df[f"{prefix}_n"].to_numpy()
        se = sd / np.sqrt(np.maximum(n, 1))

        ax.plot(x, mu, color=color, linewidth=1.4, label=f"{prefix.upper()} mean")
        ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.18,
                        label="±1 SE")
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(np.nanmean(mu), color=color, linestyle=":", linewidth=1.0,
                   label=f"daily-avg mean = {np.nanmean(mu):.4f}")
        ax.set_ylabel(prefix.upper())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(title)

    step = max(1, len(df) // 14)
    tick_idx = np.arange(0, len(df), step)
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([labels[i] for i in tick_idx], rotation=45, ha="right")
    axes[1].set_xlabel("Time of day")

    fig.suptitle(
        f"Intraday IC profile  —  {factor_name}  "
        f"(averaged over {int(df['ic_n'].max() if 'ic_n' in df else 0)} dates)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# existing plots                                                              #
# --------------------------------------------------------------------------- #
def _xaxis_continuous(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """Return continuous integer x-axis + tick labels.

    Using a true datetime axis would leave huge overnight gaps (16:00 ->
    next 09:30 = 17.5h of whitespace). Sequential index keeps the curve
    continuous, which is what users expect.
    """
    n = len(df)
    x = np.arange(n)
    if n == 0:
        return x, []
    dates = df["date"].astype(str).to_numpy()
    minutes = df["minute"].astype(int).to_numpy()
    # show one tick per day at the open
    is_new_day = np.r_[True, dates[1:] != dates[:-1]]
    tick_idx = np.where(is_new_day)[0]
    tick_labels = [
        f"{dates[i]} {_hhmm_to_str(minutes[i])}" for i in tick_idx
    ]
    return x, list(zip(tick_idx.tolist(), tick_labels))


def plot_inner_decile_ls(
    metrics: pd.DataFrame, n_groups: int, out_path: Path
) -> None:
    df = metrics.sort_values(["date", "minute"]).reset_index(drop=True)
    x, ticks = _xaxis_continuous(df)
    fig, ax = plt.subplots(figsize=(15, 6))
    cmap = plt.get_cmap("tab10")
    for k in range(1, n_groups + 1):
        col = f"decile_{k}_inner_long_short"
        if col not in df.columns:
            continue
        cum = df[col].fillna(0.0).cumsum()
        color = cmap((k - 1) % 10)
        ax.plot(x, cum, color=color, linewidth=1.2, label=f"Decile {k}")
        ax.text(x[-1], cum.iloc[-1], f" D{k}", color=color, fontsize=8,
                va="center", ha="left", fontweight="bold")
    ax.set_title(f"Within-decile Long-Short Cumulative Return ({n_groups} groups)")
    ax.set_xlabel("Trading session")
    ax.set_ylabel("Cumulative L-S return")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=5, fontsize=9)
    if ticks:
        idxs, labs = zip(*ticks)
        # thin out if too many days
        step = max(1, len(idxs) // 14)
        idxs = idxs[::step]
        labs = labs[::step]
        ax.set_xticks(idxs)
        ax.set_xticklabels(labs, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overall_ls(metrics: pd.DataFrame, out_path: Path) -> None:
    df = metrics.sort_values(["date", "minute"]).reset_index(drop=True)
    x, ticks = _xaxis_continuous(df)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x, df["long_ret"].fillna(0).cumsum(), color="red", label="Long (D10)")
    ax.plot(x, df["short_ret"].fillna(0).cumsum(), color="green", label="Short (D1)")
    ax.plot(x, df["long_short_ret"].fillna(0).cumsum(), color="black",
            linewidth=2, label="Long-Short (avg)")
    ax.set_title("Overall Long / Short / Long-Short Cumulative Return")
    ax.set_xlabel("Trading session")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if ticks:
        idxs, labs = zip(*ticks)
        step = max(1, len(idxs) // 14)
        idxs = idxs[::step]
        labs = labs[::step]
        ax.set_xticks(idxs)
        ax.set_xticklabels(labs, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
