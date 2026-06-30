"""Multi-process per-day backtest engine with multi-factor support.

Per-day worker reads factor parquet ONCE with column projection for *all*
requested factor columns, joins label ONCE, then pivots+computes per factor.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import io_utils
from .metrics import cross_section_day, summarize


# --------------------------------------------------------------------------- #
# data classes                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class DayJob:
    factor_path: Path
    label_path: Path
    factor_cols: Tuple[str, ...]
    label_col: str
    n_groups: int
    inner_q: float
    session_start: int
    session_end: int
    sample_every_n_minutes: int
    cache_root: Optional[Path]   # output_root  (per-factor subdirs underneath)
    universe_file: Optional[Path] = None  # optional PIT membership JSON


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _minute_filter(minutes: np.ndarray, start: int, end: int, every: int) -> np.ndarray:
    in_session = (minutes >= start) & (minutes <= end)
    if every <= 1:
        return in_session
    s = (minutes // 100) * 60 + (minutes % 100)
    s0 = (start // 100) * 60 + (start % 100)
    return in_session & (((s - s0) % every) == 0) & (s >= s0)


def _cache_path(
    cache_root: Optional[Path],
    factor_col: str,
    day_stem: str,
    universe_file: Optional[Path] = None,
) -> Optional[Path]:
    """Per-(factor, day) parquet cache path.

    When `universe_file` is set, the cache is namespaced under
    `_cache_<universe_stem>/` so runs with different universes don't share
    stale results. None universe lives under plain `_cache/`.
    """
    if cache_root is None:
        return None
    bucket = "_cache" if universe_file is None else f"_cache_{Path(universe_file).stem}"
    return cache_root / factor_col / bucket / f"{day_stem}.parquet"


# --------------------------------------------------------------------------- #
# per-day worker                                                              #
# --------------------------------------------------------------------------- #
def _process_long_pair(job: DayJob) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    cache_paths = {
        fc: _cache_path(job.cache_root, fc, job.factor_path.stem, job.universe_file)
        for fc in job.factor_cols
    }

    # cache hit?
    for fc in job.factor_cols:
        cp = cache_paths[fc]
        if cp is not None and cp.exists():
            try:
                out[fc] = pd.read_parquet(cp)
            except Exception:
                pass
    missing = tuple(fc for fc in job.factor_cols if fc not in out)
    if not missing:
        return out

    f = io_utils.read_long_day(job.factor_path, list(missing))
    l = io_utils.read_long_day(job.label_path, [job.label_col])
    if f.empty or l.empty:
        return out

    # PIT universe filter (mirrors tower_research). Apply BEFORE add_ticker
    # because the join key is (sym_root, sym_suffix) + date window. Filtering
    # `f` is sufficient — the subsequent inner-join with `l` on (minute,
    # ticker) drops any non-member that survives in `l`.
    if job.universe_file is not None:
        from .universe import apply_pit_membership_filter, load_pit_membership_table
        utbl = load_pit_membership_table(job.universe_file)
        f = apply_pit_membership_filter(f, utbl, date_col="date")
        if f.empty:
            return out

    io_utils.add_ticker_inplace(f)
    io_utils.add_ticker_inplace(l)

    keep_f = ["date", "minute", "ticker"] + list(missing)
    keep_l = ["minute", "ticker", job.label_col]
    f = f[keep_f]
    l = l[keep_l]

    merged = f.merge(l, on=["minute", "ticker"], how="inner", copy=False)
    if merged.empty:
        return out

    minute_arr = merged["minute"].to_numpy()
    keep = _minute_filter(
        minute_arr, job.session_start, job.session_end, job.sample_every_n_minutes
    )
    if not keep.any():
        return out
    merged = merged.loc[keep]

    R = merged.pivot_table(
        index="minute", columns="ticker", values=job.label_col,
        aggfunc="last", observed=True,
    )
    date_str = str(merged["date"].iloc[0])

    for fc in missing:
        F = merged.pivot_table(
            index="minute", columns="ticker", values=fc,
            aggfunc="last", observed=True,
        )
        F2, R2 = F.align(R, join="inner", axis=0)
        F2, R2 = F2.align(R2, join="inner", axis=1)
        minutes = F2.index.to_numpy(dtype=np.int32)
        m = cross_section_day(
            minutes,
            F2.to_numpy(dtype=np.float32, na_value=np.nan),
            R2.to_numpy(dtype=np.float32, na_value=np.nan),
            n_groups=job.n_groups,
            inner_q=job.inner_q,
        )
        m.insert(0, "date", date_str)
        out[fc] = m

        cp = cache_paths[fc]
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            try:
                m.to_parquet(cp, index=False)
            except Exception:
                pass
    return out


def _worker(job: DayJob) -> Dict[str, pd.DataFrame]:
    return _process_long_pair(job)


# --------------------------------------------------------------------------- #
# detection                                                                   #
# --------------------------------------------------------------------------- #
def _detect_factor_cols(
    src: io_utils.DataSource, override: Optional[str]
) -> List[str]:
    """If user provides --factor-col, use it; otherwise return ALL factor cols."""
    if override is not None:
        return [override]
    if src.is_long:
        cands = [
            c for c in src.columns
            if c not in io_utils.LONG_KEYS + ("Datetime",)
        ]
        if not cands:
            raise ValueError(f"No factor columns found in {src.path}")
        return cands
    # wide single-file: only one factor whose name = file stem
    return [src.path.stem]


def _detect_label_col(
    src: io_utils.DataSource, override: Optional[str]
) -> str:
    if override is not None and override in src.columns:
        return override
    candidates = ("ex_log_ret_10m", "log_ret_10m", "ex_ret_10m", "ret_10m")
    for c in candidates:
        if c in src.columns:
            return c
    if override is not None:
        return override
    raise ValueError(
        f"Cannot find a 10-minute return column; pass --label-col. "
        f"Available: {src.columns}"
    )


# --------------------------------------------------------------------------- #
# rounding for output                                                         #
# --------------------------------------------------------------------------- #
_IC_LIKE_COLS = {
    "ic", "rankic",
    "IC", "ICIR", "RankIC", "RankICIR",
}


def _round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Round IC-like cols to 4 dp; return/turnover cols to 6 dp."""
    df = df.copy()
    for c in df.columns:
        if c in _IC_LIKE_COLS:
            df[c] = df[c].astype(float).round(4)
        elif c in {
            "long_ret", "short_ret", "long_short_ret", "turnover",
        } or c.startswith("decile_"):
            df[c] = df[c].astype(float).round(6)
    return df


def _round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary all to 4 dp."""
    df = df.copy()
    int_cols = {"factor"}
    for c in df.columns:
        if c in int_cols:
            continue
        df[c] = df[c].astype(float).round(4)
    return df


# --------------------------------------------------------------------------- #
# orchestration                                                               #
# --------------------------------------------------------------------------- #
def _run_one_factor_outputs(
    factor_name: str, metrics: pd.DataFrame, n_groups: int, out_dir: Path,
) -> pd.DataFrame:
    """Write per-factor CSVs + plots; return the (un-rounded) summary row."""
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics_per_minute.csv"
    inner_path = out_dir / "decile_inner_long_short.csv"
    intraday_path = out_dir / "intraday_ic_profile.csv"

    inner_cols = ["date", "minute"] + [
        f"decile_{k}_inner_long_short" for k in range(1, n_groups + 1)
    ]

    _round_metrics(metrics).to_csv(metrics_path, index=False)
    _round_metrics(metrics[inner_cols]).to_csv(inner_path, index=False)

    from .plot import (
        plot_inner_decile_ls,
        plot_overall_ls,
        intraday_ic_profile,
        plot_intraday_ic,
    )

    profile = intraday_ic_profile(metrics)
    profile_round = profile.copy()
    for c in profile_round.columns:
        if c == "minute" or c.endswith("_n"):
            continue
        profile_round[c] = profile_round[c].astype(float).round(4)
    profile_round.to_csv(intraday_path, index=False)

    plot_inner_decile_ls(metrics, n_groups, out_dir / "decile_inner_long_short.png")
    plot_overall_ls(metrics, out_dir / "long_short_overall.png")
    plot_intraday_ic(profile, factor_name, out_dir / "intraday_ic_profile.png")

    return summarize(metrics).assign(factor=factor_name)


def _merge_summary(
    output_root: Path, new_rows: pd.DataFrame
) -> pd.DataFrame:
    """Merge new factor summary rows into output_root/summary.csv (cumulative)."""
    summary_path = output_root / "summary.csv"
    if summary_path.exists():
        try:
            old = pd.read_csv(summary_path)
            if "factor" in old.columns:
                old = old[~old["factor"].isin(new_rows["factor"])]
                merged = pd.concat([old, new_rows], ignore_index=True)
            else:
                merged = new_rows
        except Exception:
            merged = new_rows
    else:
        merged = new_rows

    cols = ["factor"] + [c for c in merged.columns if c != "factor"]
    merged = merged[cols].sort_values("factor").reset_index(drop=True)
    _round_summary(merged).to_csv(summary_path, index=False)
    return merged


def run_backtest(
    factor_path: Path,
    label_path: Path,
    output_dir: Path,
    factor_col: Optional[str] = None,
    label_col: Optional[str] = None,
    n_groups: int = 10,
    inner_q: float = 0.20,
    sample_every_n_minutes: int = 1,
    session_start: int = 930,
    session_end: int = 1550,
    workers: Optional[int] = None,
    use_cache: bool = True,
    trade_date_csv: Optional[Path] = None,
    universe_file: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path]:
    """Run a (multi-)factor backtest.

    Returns (combined_summary_df, output_root).
    """
    factor_src = io_utils.detect(factor_path)
    label_src = io_utils.detect(label_path)

    f_cols = _detect_factor_cols(factor_src, factor_col)
    l_col = _detect_label_col(label_src, label_col)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root if use_cache else None

    print(f"[info] factors  : {f_cols}")
    print(f"[info] label    : {l_col}")
    print(f"[info] output   : {output_root}")
    if universe_file is not None:
        print(f"[info] universe : {universe_file}")

    # --- branch 1: long-format directory --------------------------------- #
    if factor_src.is_long and label_src.is_long and factor_src.is_dir and label_src.is_dir:
        pairs = io_utils.pair_long_days(factor_src, label_src)
        if trade_date_csv is not None and Path(trade_date_csv).exists():
            td = pd.read_csv(trade_date_csv)
            keys = {pd.to_datetime(str(x)).strftime("%Y%m%d") for x in td.iloc[:, 0]}
            pairs = [(f, l) for f, l in pairs if f.stem.replace("-", "") in keys]
        if not pairs:
            raise RuntimeError("No matched (factor, label) day pairs found.")

        jobs = [
            DayJob(
                factor_path=fp,
                label_path=lp,
                factor_cols=tuple(f_cols),
                label_col=l_col,
                n_groups=n_groups,
                inner_q=inner_q,
                session_start=session_start,
                session_end=session_end,
                sample_every_n_minutes=sample_every_n_minutes,
                cache_root=cache_root,
                universe_file=Path(universe_file) if universe_file else None,
            )
            for fp, lp in pairs
        ]

        n_workers = workers or min(8, max(1, (os.cpu_count() or 2) - 1))
        print(f"[info] days     : {len(jobs)}   workers: {n_workers}")
        per_factor: Dict[str, List[pd.DataFrame]] = {fc: [] for fc in f_cols}

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, j): j for j in jobs}
            desc = f"backtest[{','.join(f_cols) if len(f_cols) <= 3 else f'{len(f_cols)} factors'}]"
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"[warn] worker failed: {e}")
                    continue
                for fc, df in res.items():
                    if df is not None and not df.empty:
                        per_factor[fc].append(df)

    else:
        # --- branch 2: single-file long OR wide; sequential is fine ----- #
        per_factor = {fc: [] for fc in f_cols}
        if factor_src.is_long:
            f_long = pd.concat(
                [io_utils.read_long_day(p, list(f_cols)) for p in factor_src.files],
                ignore_index=True,
            )
            io_utils.add_ticker_inplace(f_long)
        else:
            f_long = io_utils.read_wide(factor_src, "factor")  # only 1 factor
        if label_src.is_long:
            l_long = pd.concat(
                [io_utils.read_long_day(p, [l_col]) for p in label_src.files],
                ignore_index=True,
            )
            io_utils.add_ticker_inplace(l_long)
            l_long = l_long.rename(columns={l_col: "ret"})
        else:
            l_long = io_utils.read_wide(label_src, "ret")

        for fc in f_cols:
            if factor_src.is_long:
                fdf = f_long[["date", "minute", "ticker", fc]].rename(columns={fc: "factor"})
            else:
                fdf = f_long  # already 'factor'
            merged = fdf.merge(l_long[["date", "minute", "ticker", "ret"]],
                               on=["date", "minute", "ticker"], how="inner")
            keep = _minute_filter(
                merged["minute"].to_numpy(),
                session_start, session_end, sample_every_n_minutes,
            )
            merged = merged.loc[keep]
            if merged.empty:
                continue
            for date, sub in tqdm(merged.groupby("date", sort=True),
                                  desc=f"backtest[{fc}]"):
                F = sub.pivot_table(index="minute", columns="ticker",
                                    values="factor", aggfunc="last", observed=True)
                R = sub.pivot_table(index="minute", columns="ticker",
                                    values="ret", aggfunc="last", observed=True)
                F, R = F.align(R, join="inner", axis=0)
                F, R = F.align(R, join="inner", axis=1)
                md = cross_section_day(
                    F.index.to_numpy(dtype=np.int32),
                    F.to_numpy(dtype=np.float32, na_value=np.nan),
                    R.to_numpy(dtype=np.float32, na_value=np.nan),
                    n_groups=n_groups, inner_q=inner_q,
                )
                md.insert(0, "date", str(date))
                per_factor[fc].append(md)

    # --- write per-factor outputs and aggregate summary ----------------- #
    summaries: List[pd.DataFrame] = []
    for fc in f_cols:
        chunks = per_factor.get(fc, [])
        if not chunks:
            print(f"[warn] {fc}: no metrics produced")
            continue
        metrics = (pd.concat(chunks, ignore_index=True)
                     .sort_values(["date", "minute"]).reset_index(drop=True))
        out_dir = output_root / fc
        s = _run_one_factor_outputs(fc, metrics, n_groups, out_dir)
        summaries.append(s)

    if not summaries:
        raise RuntimeError("No factor produced any metrics.")
    new_rows = pd.concat(summaries, ignore_index=True)
    combined = _merge_summary(output_root, new_rows)
    return combined, output_root
