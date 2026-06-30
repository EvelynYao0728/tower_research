"""Tiny synthetic-data sanity test for vectorized engine."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from single_factor_bt.metrics import (
    CORE_SUMMARY_METRIC_COLUMNS,
    _business_days_in_span,
    cross_section_day,
    summarize,
)
from single_factor_bt.universe import (
    apply_pit_membership_filter,
    load_pit_membership_table,
)


def test_perfect_rank_factor():
    # 2 minutes, 10 stocks. Factor = ret exactly => IC == RankIC == 1.0
    np.random.seed(0)
    T, N, K = 2, 10, 10
    ret = np.random.randn(T, N).astype(np.float32)
    factor = ret.copy()
    minutes = np.array([930, 940], dtype=np.int32)

    df = cross_section_day(minutes, factor, ret, n_groups=K, inner_q=0.2)
    assert np.allclose(df["ic"], 1.0, atol=1e-6)
    assert np.allclose(df["rankic"], 1.0, atol=1e-6)
    # decile_1 = lowest factor; decile_K = highest
    assert (df["decile_1_ret"] < df[f"decile_{K}_ret"]).all()
    assert (df["long_short_ret"] > 0).all()


def test_summary_runs():
    minutes = np.arange(930, 940, dtype=np.int32)
    T, N, K = 10, 30, 10
    rng = np.random.default_rng(1)
    f = rng.standard_normal((T, N)).astype(np.float32)
    r = rng.standard_normal((T, N)).astype(np.float32)
    df = cross_section_day(minutes, f, r, n_groups=K, inner_q=0.2)
    s = summarize(df)
    assert list(s.columns) == list(CORE_SUMMARY_METRIC_COLUMNS)


def _make_metrics(start_date: str, n_business_days: int, n_minutes: int,
                  seed: int = 0) -> pd.DataFrame:
    """Build a synthetic per-cross-section metrics frame spanning real dates."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_business_days)
    rows = []
    for d in dates:
        for mm in range(n_minutes):
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "minute": 930 + mm,
                "ic": rng.normal(0.05, 0.5),
                "rankic": rng.normal(0.05, 0.5),
                "long_ret": rng.normal(1e-4, 1e-3),
                "short_ret": rng.normal(1e-4, 1e-3),
                "long_short_ret": rng.normal(1e-4, 1e-3),
                "turnover": 0.05,
            })
    return pd.DataFrame(rows)


def _periods_per_year(metrics: pd.DataFrame, annualization_days: int = 252) -> float:
    n_samples = len(metrics)
    biz = _business_days_in_span(metrics)
    if biz >= 1 and annualization_days > 0:
        years = biz / annualization_days
        return (n_samples / years) if years > 0 else 0.0
    if n_samples:
        return float(n_samples) * annualization_days
    return 0.0


def test_summary_annualization_uses_calendar_span():
    """Annualized long_short_ret scales with n_samples / (biz_days_in_span / 252)."""
    m = _make_metrics("2025-01-02", n_business_days=5, n_minutes=100)
    s = summarize(m)
    raw_ls = m["long_short_ret"].mean()
    ppy = _periods_per_year(m, 252)
    assert math.isclose(float(s["long_short_ret"].iloc[0]), raw_ls * ppy, rel_tol=1e-9)

    m_sparse = _make_metrics("2025-03-03", n_business_days=5, n_minutes=10)
    obs_dates = sorted(m_sparse["date"].unique())
    m_sparse = m_sparse[m_sparse["date"].isin({obs_dates[0], obs_dates[-1]})]
    s_sparse = summarize(m_sparse.reset_index(drop=True))
    ppy_sparse = _periods_per_year(m_sparse.reset_index(drop=True), 252)
    raw_sparse = m_sparse["long_short_ret"].mean()
    assert math.isclose(
        float(s_sparse["long_short_ret"].iloc[0]), raw_sparse * ppy_sparse, rel_tol=1e-9
    )


def test_summary_sharpe_scales_with_sqrt_obs_per_year():
    metrics = _make_metrics("2025-04-01", n_business_days=3, n_minutes=80, seed=7)
    s = summarize(metrics)

    ls = metrics["long_short_ret"].to_numpy()
    raw_sharpe = ls.mean() / ls.std(ddof=1)
    ppy = _periods_per_year(metrics, 252)
    expected_sharpe = raw_sharpe * math.sqrt(ppy)

    assert math.isclose(float(s["sharpe"].iloc[0]), expected_sharpe, rel_tol=1e-6)


def test_summary_annualization_days_param():
    """annualization_days override (e.g. 250) scales annualized returns linearly."""
    metrics = _make_metrics("2025-05-05", n_business_days=4, n_minutes=50, seed=3)
    s252 = summarize(metrics, annualization_days=252)
    s250 = summarize(metrics, annualization_days=250)
    ratio = float(s252["long_short_ret"].iloc[0]) / float(s250["long_short_ret"].iloc[0])
    assert math.isclose(ratio, 252 / 250, rel_tol=1e-9)


def test_rankic_matches_textbook_spearman_with_nan():
    """RankIC must equal scipy.spearmanr on the (factor∩label) intersection.

    Regression test for the previous bug where factor was ranked on
    valid_f and label on valid_r (different denominators) before
    correlating on the intersection.
    """
    rng = np.random.default_rng(12345)
    T, N, K = 5, 40, 10
    F = rng.standard_normal((T, N)).astype(np.float32)
    R = rng.standard_normal((T, N)).astype(np.float32)
    # Drill holes: some cells factor-only NaN, some label-only NaN, some both.
    nan_f = rng.random((T, N)) < 0.10
    nan_r = rng.random((T, N)) < 0.10
    F[nan_f] = np.nan
    R[nan_r] = np.nan
    minutes = np.arange(930, 930 + T, dtype=np.int32)

    df = cross_section_day(minutes, F, R, n_groups=K, inner_q=0.2)

    for t in range(T):
        f_row, r_row = F[t], R[t]
        m = np.isfinite(f_row) & np.isfinite(r_row)
        if m.sum() < 2:
            assert np.isnan(df["rankic"].iloc[t])
            continue
        # scipy spearmanr ranks both inputs on the SAME sample (f_row[m], r_row[m]).
        rho, _ = spearmanr(f_row[m], r_row[m])
        assert math.isclose(float(df["rankic"].iloc[t]), float(rho), rel_tol=1e-5,
                            abs_tol=1e-7), (t, df["rankic"].iloc[t], rho)


# ── PIT universe filter ─────────────────────────────────────────────────────
def _make_universe_json(path: Path):
    payload = {
        "metadata": {"source": "test"},
        "universe": [
            # always active (full year)
            {"ticker": "AAA", "wrds_sym_root": "AAA", "wrds_sym_suffix": None,
             "in_date": "2025-01-02", "out_date": None},
            # active until mid-year (delisted on 2025-06-30)
            {"ticker": "BBB", "wrds_sym_root": "BBB", "wrds_sym_suffix": None,
             "in_date": "2025-01-02", "out_date": "2025-06-30"},
            # share class with explicit suffix
            {"ticker": "CCC.B", "wrds_sym_root": "CCC", "wrds_sym_suffix": "B",
             "in_date": "2025-01-02", "out_date": None},
            # added mid-year
            {"ticker": "DDD", "wrds_sym_root": "DDD", "wrds_sym_suffix": None,
             "in_date": "2025-07-01", "out_date": None},
        ],
    }
    path.write_text(json.dumps(payload))


def _make_bars():
    """Bars across 4 dates, mixing universe + non-universe + PIT-edge cases."""
    return pd.DataFrame({
        "date":       ["2025-01-15"] * 5 + ["2025-06-29"] * 5 + ["2025-07-15"] * 5,
        "sym_root":   ["AAA", "BBB", "CCC", "DDD", "ZZZ"] * 3,
        # mix all three "no suffix" representations to exercise normalization
        "sym_suffix": [None, "None", "B", "", None] * 3,
        "minute":     [930] * 15,
        "factor":     list(range(15)),
    })


def test_pit_universe_filter_membership_and_pit_window():
    with tempfile.TemporaryDirectory() as tmp:
        upath = Path(tmp) / "uni.json"
        _make_universe_json(upath)
        utbl = load_pit_membership_table(upath)
        # Universe table itself: 4 rows, sym_suffix canonicalized
        assert set(utbl["sym_root"].astype(str)) == {"AAA", "BBB", "CCC", "DDD"}
        assert (utbl.loc[utbl["sym_root"].astype(str) == "AAA", "sym_suffix"]
                .astype(str).iloc[0] == "")
        assert (utbl.loc[utbl["sym_root"].astype(str) == "CCC", "sym_suffix"]
                .astype(str).iloc[0] == "B")

        bars = _make_bars()
        out = apply_pit_membership_filter(bars, utbl).reset_index(drop=True)

        # ZZZ is never in the universe -> dropped on every date
        assert (out["sym_root"].astype(str) != "ZZZ").all()

        # On 2025-01-15: AAA, BBB, CCC.B in; DDD not yet (in_date=2025-07-01)
        d1 = out[out["date"].astype(str) == "2025-01-15"]
        assert set(d1["sym_root"].astype(str)) == {"AAA", "BBB", "CCC"}

        # On 2025-06-29: BBB still active (out_date=2025-06-30, exclusive)
        d2 = out[out["date"].astype(str) == "2025-06-29"]
        assert "BBB" in set(d2["sym_root"].astype(str))

        # On 2025-07-15: BBB delisted (06-30), DDD now active (in_date 07-01)
        d3 = out[out["date"].astype(str) == "2025-07-15"]
        d3_roots = set(d3["sym_root"].astype(str))
        assert "BBB" not in d3_roots
        assert "DDD" in d3_roots
        assert "AAA" in d3_roots and "CCC" in d3_roots


def test_pit_universe_out_date_is_exclusive_upper_bound():
    """Boundary: a ticker with out_date=D must be filtered OUT on date D."""
    with tempfile.TemporaryDirectory() as tmp:
        upath = Path(tmp) / "uni.json"
        _make_universe_json(upath)
        utbl = load_pit_membership_table(upath)

        # BBB has out_date=2025-06-30; bar on exactly 2025-06-30 must be dropped.
        bars = pd.DataFrame({
            "date":       ["2025-06-30"],
            "sym_root":   ["BBB"],
            "sym_suffix": [None],
            "minute":     [930],
            "factor":     [1.0],
        })
        out = apply_pit_membership_filter(bars, utbl)
        assert out.empty


if __name__ == "__main__":
    test_perfect_rank_factor()
    test_summary_runs()
    test_summary_annualization_uses_calendar_span()
    test_summary_sharpe_scales_with_sqrt_obs_per_year()
    test_summary_annualization_days_param()
    test_rankic_matches_textbook_spearman_with_nan()
    test_pit_universe_filter_membership_and_pit_window()
    test_pit_universe_out_date_is_exclusive_upper_bound()
    print("OK")
