"""Vectorized cross-section metrics. Pure numpy. No python loops over stocks."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

# Exported in summary.csv and QuantaAlpha trajectory metrics (no decile / monotonicity).
CORE_SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "IC",
    "ICIR",
    "RankIC",
    "RankICIR",
    "long_ret",
    "short_ret",
    "long_short_ret",
    "volatility",
    "sharpe",
    "turnover",
)


def _row_pearson(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Pearson correlation per row of (T, N), masking invalid cells.

    All inputs shape (T, N). NaN-safe via ``valid`` mask. Returns shape (T,).
    """
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    cnt = valid.sum(axis=1).astype(np.float64)
    sx = xv.sum(axis=1)
    sy = yv.sum(axis=1)
    sxx = (xv * xv).sum(axis=1)
    syy = (yv * yv).sum(axis=1)
    sxy = (xv * yv).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = sx / cnt
        my = sy / cnt
        cov = sxy / cnt - mx * my
        vx = sxx / cnt - mx * mx
        vy = syy / cnt - my * my
        out = cov / np.sqrt(vx * vy)
    out[cnt < 2] = np.nan
    return out


def _row_rank(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Average rank per row over (T, N), masking invalid as NaN.

    Implementation: pandas DataFrame.rank along axis=1, vectorized in C.
    """
    masked = np.where(valid, arr, np.nan)
    ranks = pd.DataFrame(masked).rank(axis=1, method="average").to_numpy()
    return ranks


def _decile_assign(
    f_rank_asc: np.ndarray, valid: np.ndarray, n_groups: int
) -> np.ndarray:
    """Assign decile index per cell. D1 = smallest factor, DK = largest.

    Returns shape (T, N), values in {0..K}, 0=invalid.
    """
    n_valid = valid.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore"):
        bucket = np.ceil(f_rank_asc * n_groups / np.maximum(n_valid, 1.0))
    bucket = np.where(valid, bucket, 0.0)
    bucket = np.clip(bucket, 1, n_groups).astype(np.int32)
    bucket[~valid] = 0
    return bucket


def _group_mean(
    values: np.ndarray, group: np.ndarray, n_groups: int
) -> np.ndarray:
    """Row-wise group mean. Returns (T, n_groups). NaN where group empty.

    values: (T, N). group: (T, N) int in {0..K}, 0=skip.
    Vectorized via flat bincount with composite key = t * (K+1) + g.
    """
    T, N = values.shape
    valid = (group > 0) & np.isfinite(values)
    flat_v = values.ravel()
    flat_g = group.ravel()
    flat_t = np.repeat(np.arange(T, dtype=np.int64), N)
    K1 = n_groups + 1
    key = flat_t * K1 + flat_g
    m = valid.ravel()
    sums = np.bincount(key[m], weights=flat_v[m], minlength=T * K1)
    cnts = np.bincount(key[m], minlength=T * K1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / cnts
    means[cnts == 0] = np.nan
    means = means.reshape(T, K1)
    return means[:, 1:]  # drop group=0 column


def _inner_long_short(
    factor: np.ndarray, ret: np.ndarray, decile: np.ndarray, n_groups: int, q: float
) -> np.ndarray:
    """Within-decile top-q long, bottom-q short, return (long+short)/2.

    factor/ret: (T, N). decile: (T, N) int in {0..K}, 0=skip.
    Returns (T, n_groups). For each decile k, picks top-q*size_k highest factor
    stocks (long) and bottom-q*size_k lowest factor stocks (short) within row.
    """
    T, N = factor.shape
    out = np.full((T, n_groups), np.nan, dtype=np.float64)

    for k in range(1, n_groups + 1):
        mask_k = decile == k                                         # (T, N)
        # masked factor: NaN where not in this decile, so they sort to the end
        f_k = np.where(mask_k, factor, np.nan)
        # rank descending: larger factor => smaller rank (1 = top)
        # use pandas rank axis=1 of (-f_k); NaN preserved
        ranks_desc = pd.DataFrame(-f_k).rank(axis=1, method="first").to_numpy()
        size_k = mask_k.sum(axis=1, keepdims=True).astype(np.int64)  # (T, 1)
        n_pick = np.maximum(1, np.floor(size_k * q).astype(np.int64))

        long_mask = mask_k & (ranks_desc <= n_pick)
        short_mask = mask_k & (ranks_desc > size_k - n_pick) & (size_k.astype(bool))

        # mean ret long & short row-wise, NaN safe
        def _row_mean(mask: np.ndarray) -> np.ndarray:
            r = np.where(mask, ret, 0.0)
            c = (mask & np.isfinite(ret)).sum(axis=1).astype(np.float64)
            s = np.where(mask & np.isfinite(ret), ret, 0.0).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = s / c
            m[c == 0] = np.nan
            return m

        long_v = _row_mean(long_mask)
        short_v = -_row_mean(short_mask)
        out[:, k - 1] = (long_v + short_v) / 2.0
    return out


def _monotonicity_score(decile_rets: Sequence[float], n_groups: int) -> float:
    """Fraction of adjacent decile pairs with decile_{k+1}_ret > decile_k_ret."""
    ok = 0
    pairs = 0
    for k in range(n_groups - 1):
        a, b = decile_rets[k], decile_rets[k + 1]
        if np.isfinite(a) and np.isfinite(b):
            pairs += 1
            if b > a:
                ok += 1
    return ok / pairs if pairs else np.nan


def _portfolio_turnover(
    decile: np.ndarray, n_groups: int, hold: int = 10
) -> np.ndarray:
    """One-sided per-minute turnover of the dollar-neutral 50/50 portfolio.

    Methodology
    -----------
    Each minute we generate a signal that is 50% long (decile 1, equal-weight
    among its members) and 50% short (decile K, equal-weight). We hold each
    signal for ``hold`` minutes, so the actually-held portfolio at minute t is
    the average of the most recent ``hold`` signals (overlapping book).

    Signal weight at minute t, stock i:
        +1/(2*|D1_t|)   if decile_t,i == 1     (long  bucket, 50% gross)
        -1/(2*|DK_t|)   if decile_t,i == K     (short bucket, 50% gross)
         0              otherwise
    Held portfolio:
        w_t = (1/hold) * sum_{k=0..hold-1} signal_{t-k}
    Per-minute one-sided turnover (telescoping → only entering/exiting tranche):
        TO_t = 0.5 * sum_i |w_t,i - w_{t-1},i|
             = 1/(2*hold) * L1( signal_t - signal_{t-hold} )

    Returns (T,) one-sided per-minute turnover; first ``hold`` rows are NaN
    (warm-up). Annualization: mean(TO_t) * minutes_per_year.
    """
    T, N = decile.shape
    out = np.full(T, np.nan, dtype=np.float64)
    if T <= hold:
        return out

    long_member = (decile == n_groups).astype(np.float64)
    short_member = (decile == 1).astype(np.float64)
    nL = long_member.sum(axis=1, keepdims=True)
    nS = short_member.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        wL = np.where(nL > 0, long_member / (2.0 * nL), 0.0)
        wS = np.where(nS > 0, -short_member / (2.0 * nS), 0.0)
    sig = wL + wS  # (T, N), gross 1.0 row-wise

    diff = sig[hold:] - sig[:-hold]
    l1 = np.abs(diff).sum(axis=1)
    out[hold:] = l1 / (2.0 * hold)
    return out


def cross_section_day(
    minutes: np.ndarray,           # (T,)
    factor_mat: np.ndarray,        # (T, N) float32
    ret_mat: np.ndarray,           # (T, N) float32
    n_groups: int = 10,
    inner_q: float = 0.20,
    hold_minutes: int = 10,
) -> pd.DataFrame:
    """Compute all metrics for one day given pre-pivoted matrices.

    Returns one row per minute with columns:
        minute, n_obs,
        ic, rankic,
        long_ret, short_ret, long_short_ret,
        turnover,
        decile_1_ret .. decile_K_ret,
        decile_1_inner_long_short .. decile_K_inner_long_short
    """
    F = factor_mat.astype(np.float64, copy=False)
    R = ret_mat.astype(np.float64, copy=False)
    valid_f = np.isfinite(F)
    valid_r = np.isfinite(R)
    valid = valid_f & valid_r

    n_obs = valid.sum(axis=1).astype(np.int64)

    ic = _row_pearson(F, R, valid)

    # Spearman RankIC: rank factor and label on the SAME sample (the
    # intersection where both are finite), then Pearson-correlate the ranks.
    # Ranking each side on its own validity (valid_f vs valid_r) and then
    # correlating on the intersection puts the two rank vectors on different
    # denominators and is not the textbook Spearman correlation.
    f_rank_xs = _row_rank(F, valid)
    r_rank_xs = _row_rank(R, valid)
    rankic = _row_pearson(f_rank_xs, r_rank_xs, valid)

    # Decile bucket comes from a factor-only rank (factor needs to exist to
    # be assigned a bucket); cells with missing label are still bucketed but
    # contribute NaN to the per-bucket return mean.
    f_rank_all = _row_rank(F, valid_f)
    decile = _decile_assign(f_rank_all, valid_f, n_groups)
    # only decile cells that also have valid ret count for returns
    decile_for_ret = np.where(valid_r, decile, 0)
    dec_means = _group_mean(R, decile_for_ret, n_groups)        # (T, K)
    long_ret = dec_means[:, n_groups - 1]
    short_ret = dec_means[:, 0]
    long_short_ret = (long_ret - short_ret) / 2.0

    # one-sided per-minute turnover of the full 50/50 long-short book with
    # overlapping ``hold_minutes`` holdings
    turnover = _portfolio_turnover(decile, n_groups, hold=hold_minutes)

    inner_ls = _inner_long_short(F, R, decile_for_ret, n_groups, inner_q)

    out = {
        "minute": minutes.astype(np.int32),
        "n_obs": n_obs,
        "ic": ic,
        "rankic": rankic,
        "long_ret": long_ret,
        "short_ret": short_ret,
        "long_short_ret": long_short_ret,
        "turnover": turnover,
    }
    for k in range(1, n_groups + 1):
        out[f"decile_{k}_ret"] = dec_means[:, k - 1]
    for k in range(1, n_groups + 1):
        out[f"decile_{k}_inner_long_short"] = inner_ls[:, k - 1]
    return pd.DataFrame(out)


def _business_days_in_span(metrics: pd.DataFrame) -> int:
    """Count business days in the inclusive range [date_min, date_max].

    Uses numpy.busday_count which excludes weekends (Sat/Sun) by default.
    Returns 0 if the metrics frame has no usable 'date' column.
    """
    if "date" not in metrics.columns or len(metrics) == 0:
        return 0
    dates = pd.to_datetime(metrics["date"].astype(str), errors="coerce").dropna()
    if dates.empty:
        return 0
    d_min = dates.min().date()
    d_max = dates.max().date()
    end_exclusive = d_max + pd.Timedelta(days=1)
    return int(np.busday_count(d_min, end_exclusive.date() if hasattr(end_exclusive, "date") else end_exclusive))


def summarize(metrics: pd.DataFrame, annualization_days: int = 252) -> pd.DataFrame:
    """Aggregate per-cross-section metrics into a 1-row summary.

    All return / risk figures are reported **annualized** by default; column
    names omit any '_ann' / '_mean' suffix. Rounding to 4 dp is done in
    engine._round_summary.

    Annualization (calendar-span based, no hard-coded period count)
    ---------------------------------------------------------------
    The annualization coefficient is derived from the **actual calendar
    span** of the data, NOT a hard-coded constant like 9828 or
    obs_per_day × 252:

        biz_days_in_span = numpy.busday_count(date_min, date_max + 1 day)
        years            = biz_days_in_span / annualization_days
        obs_per_year     = n_samples / years

        annual_ret  = mean(per-obs ret) * obs_per_year
        annual_vol  = std(per-obs ret)  * sqrt(obs_per_year)
        sharpe      = annual_ret / annual_vol
                    = (mean / std) * sqrt(obs_per_year)

    Why this is more honest than `obs_per_day × annualization_days`
    --------------------------------------------------------------
    * Contiguous full-year data with no missing days: identical result.
    * Sparse data (some trading days skipped, e.g. data outage, partial
      universe): the calendar-span method correctly yields a SMALLER
      obs_per_year, because the strategy effectively "didn't trade" on
      missing days. The old formula would invent samples by extrapolating
      obs_per_day × 252.
    * Partial-year data (e.g. 6 months): years ≈ 0.5, so obs_per_year is
      half of a full-year run with the same intraday density.

    The only finance-convention constant is `annualization_days` (default
    252 trading days per year). Pass annualization_days=250 to align with
    tower_research's `run_full_backtest_report.py`.

    Caveat for overlapping holds (e.g. 10-minute forward label sampled every
    minute): per-minute returns are highly autocorrelated, so std is biased
    down and Sharpe is biased up by ~sqrt(H). If you need a "true" Sharpe,
    sample every H minutes (--sample-every-n-minutes H) so observations are
    non-overlapping; obs_per_year will shrink accordingly.

    Turnover stays the **per-minute one-sided rate** (NOT annualized):
    overlapping-hold telescoping makes this exactly the real trading need.
    Range [0, 0.1]; 0.1 = full-book turn every 10 minutes.
    """

    def m(c):
        v = metrics[c].to_numpy()
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.nan

    def s(c):
        v = metrics[c].to_numpy()
        v = v[np.isfinite(v)]
        return float(v.std(ddof=1)) if v.size > 1 else np.nan

    ic_m, ic_s = m("ic"), s("ic")
    ri_m, ri_s = m("rankic"), s("rankic")

    # ── Calendar-span annualization ──────────────────────────────────────
    # n_samples       : actually observed cross-sections (after session
    #                   filter, sample_every, NaN drop).
    # n_days          : distinct trading dates that produced data.
    # biz_days_in_span: business days between min(date) and max(date)
    #                   inclusive; this is the calendar denominator.
    # years           : biz_days_in_span / annualization_days.
    # obs_per_year    : n_samples / years -- the only correct rate for
    #                   "if this strategy ran continuously, how many
    #                    observations would it accumulate per year?"
    n_samples = len(metrics)
    if "date" in metrics.columns and n_samples:
        n_days = max(int(metrics["date"].nunique()), 1)
    else:
        n_days = 1
    biz_days_in_span = _business_days_in_span(metrics)
    if biz_days_in_span >= 1 and annualization_days > 0:
        years = biz_days_in_span / annualization_days
        PERIODS_PER_YEAR = (n_samples / years) if years > 0 else 0.0
    elif n_samples:
        # single-day fallback: treat the day as 1/annualization_days of a year
        PERIODS_PER_YEAR = float(n_samples) * annualization_days
        biz_days_in_span = max(biz_days_in_span, 1)
    else:
        PERIODS_PER_YEAR = 0.0

    long_m = m("long_ret")
    short_m = m("short_ret")
    ls_m = m("long_short_ret")
    ls_s = s("long_short_ret")

    has_periods = PERIODS_PER_YEAR > 0
    long_ann = long_m * PERIODS_PER_YEAR if has_periods and np.isfinite(long_m) else np.nan
    short_ann = short_m * PERIODS_PER_YEAR if has_periods and np.isfinite(short_m) else np.nan
    ls_ann = ls_m * PERIODS_PER_YEAR if has_periods and np.isfinite(ls_m) else np.nan
    ls_vol_ann = (
        ls_s * np.sqrt(PERIODS_PER_YEAR)
        if has_periods and ls_s and np.isfinite(ls_s) else np.nan
    )
    sharpe = (
        ls_ann / ls_vol_ann
        if ls_vol_ann and np.isfinite(ls_vol_ann)
        else np.nan
    )
    # Real turnover rate per minute (overlapping-hold aware).
    turnover_rate = m("turnover") if "turnover" in metrics.columns else np.nan

    decile_cols = [
        c for c in metrics.columns if c.startswith("decile_") and c.endswith("_ret")
    ]
    n_groups = len(decile_cols)
    decile_raw = [m(c) for c in sorted(decile_cols, key=lambda x: int(x.split("_")[1]))]
    row: dict = {
        "IC": ic_m,
        "ICIR": ic_m / ic_s if ic_s and np.isfinite(ic_s) else np.nan,
        "RankIC": ri_m,
        "RankICIR": ri_m / ri_s if ri_s and np.isfinite(ri_s) else np.nan,
        "long_ret": long_ann,
        "short_ret": short_ann,
        "long_short_ret": ls_ann,
        "volatility": ls_vol_ann,
        "sharpe": sharpe,
        "turnover": turnover_rate,
    }

    return pd.DataFrame([row], columns=list(CORE_SUMMARY_METRIC_COLUMNS))
