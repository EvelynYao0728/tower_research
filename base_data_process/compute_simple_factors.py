"""Compute minute simple_factors from TAQ quote + trade universe."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from base_data_process.config import (
    OUTPUT_COLS,
    QUOTE_COLS,
    SESSION_MINUTES,
    TRADE_UNIVERSE_COLS,
)
from base_data_process.time_utils import (
    add_minute_from_ts,
    add_ts_ns_quote,
    filter_session,
    minute_start_ns,
    normalize_suffix,
)

SESSION_MINUTES_ARR = np.array(SESSION_MINUTES, dtype=np.int32)
_NUMBA_WARMED = False


@njit(cache=True)
def _snapshot_std(values: np.ndarray, cnt: int, mean: float) -> float:
    if cnt <= 1:
        return np.nan
    var = 0.0
    for j in range(cnt):
        d = values[j] - mean
        var += d * d
    return math.sqrt(var / (cnt - 1))


@njit(cache=True)
def _stats_one_minute(
    ts: np.ndarray,
    spread: np.ndarray,
    mid: np.ndarray,
    imb: np.ndarray,
    n: int,
    minute: int,
) -> tuple[bool, float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """Aggregate one minute from in-minute quote snapshots (unweighted mean/std)."""
    ms = ((minute // 100) * 3600 + (minute % 100) * 60) * 1_000_000_000
    me = ms + 60_000_000_000

    snap_spread = np.empty(n, dtype=np.float64)
    snap_mid = np.empty(n, dtype=np.float64)
    snap_imb = np.empty(n, dtype=np.float64)
    snap_cnt = 0
    for i in range(n):
        if ts[i] >= ms and ts[i] < me:
            snap_spread[snap_cnt] = spread[i]
            snap_mid[snap_cnt] = mid[i]
            snap_imb[snap_cnt] = imb[i]
            snap_cnt += 1
    if snap_cnt == 0:
        return False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

    sp_first = snap_spread[0]
    sp_last = snap_spread[snap_cnt - 1]
    sp_max = snap_spread[0]
    sp_min = snap_spread[0]
    mid_first = snap_mid[0]
    mid_last = snap_mid[snap_cnt - 1]
    mid_max = snap_mid[0]
    mid_min = snap_mid[0]

    sum_sp = sp_first
    sum_mid = mid_first
    sum_imb = snap_imb[0]
    for j in range(1, snap_cnt):
        v = snap_spread[j]
        sum_sp += v
        if v > sp_max:
            sp_max = v
        if v < sp_min:
            sp_min = v
        v = snap_mid[j]
        sum_mid += v
        if v > mid_max:
            mid_max = v
        if v < mid_min:
            mid_min = v
        sum_imb += snap_imb[j]

    mean_sp = sum_sp / snap_cnt
    mean_mid = sum_mid / snap_cnt
    mean_imb = sum_imb / snap_cnt
    std_sp = _snapshot_std(snap_spread, snap_cnt, mean_sp)
    std_mid = _snapshot_std(snap_mid, snap_cnt, mean_mid)
    std_imb = _snapshot_std(snap_imb, snap_cnt, mean_imb)

    return (
        True,
        sp_first,
        sp_last,
        sp_max,
        sp_min,
        mean_sp,
        std_sp,
        mid_first,
        mid_last,
        mid_max,
        mid_min,
        mean_mid,
        std_mid,
        mean_imb,
        std_imb,
    )


@njit(cache=True)
def _compute_ticker(
    ts: np.ndarray,
    spread: np.ndarray,
    mid: np.ndarray,
    imb: np.ndarray,
    minutes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = ts.shape[0]
    max_out = minutes.shape[0]
    out_m = np.empty(max_out, dtype=np.int32)
    out = np.empty((max_out, 14), dtype=np.float64)
    n_out = 0
    for k in range(minutes.shape[0]):
        minute = minutes[k]
        ok, sp_first, sp_last, sp_max, sp_min, mean_sp, std_sp, mid_first, mid_last, mid_max, mid_min, mean_mid, std_mid, mean_imb, std_imb = _stats_one_minute(
            ts, spread, mid, imb, n, minute
        )
        if not ok:
            continue
        out_m[n_out] = minute
        out[n_out, 0] = sp_first
        out[n_out, 1] = sp_last
        out[n_out, 2] = sp_max
        out[n_out, 3] = sp_min
        out[n_out, 4] = mean_sp
        out[n_out, 5] = std_sp
        out[n_out, 6] = mid_first
        out[n_out, 7] = mid_last
        out[n_out, 8] = mid_max
        out[n_out, 9] = mid_min
        out[n_out, 10] = mean_mid
        out[n_out, 11] = std_mid
        out[n_out, 12] = mean_imb
        out[n_out, 13] = std_imb
        n_out += 1
    return (
        out_m[:n_out],
        out[:n_out, 0],
        out[:n_out, 1],
        out[:n_out, 2],
        out[:n_out, 3],
        out[:n_out, 4],
        out[:n_out, 5],
        out[:n_out, 6],
        out[:n_out, 7],
        out[:n_out, 8],
        out[:n_out, 9],
        out[:n_out, 10],
        out[:n_out, 11],
        out[:n_out, 12],
        out[:n_out, 13],
    )


def _warmup_numba() -> None:
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return
    ts = np.array([minute_start_ns(930), minute_start_ns(930) + 1_000_000_000], dtype=np.int64)
    spread = np.array([0.01, 0.02], dtype=np.float64)
    mid = np.array([100.0, 100.01], dtype=np.float64)
    imb = np.array([0.1, -0.1], dtype=np.float64)
    _compute_ticker(ts, spread, mid, imb, SESSION_MINUTES_ARR)
    _NUMBA_WARMED = True


def _suffix_pairs(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(["sym_root", "sym_suffix"]).unique()


def load_universe(trade_path: Path, quote_path: Path) -> pl.DataFrame:
    """Tickers present in both trade and quote for the day."""
    trade_u = _suffix_pairs(pl.read_parquet(trade_path, columns=TRADE_UNIVERSE_COLS))
    quote_u = _suffix_pairs(pl.read_parquet(quote_path, columns=TRADE_UNIVERSE_COLS))
    trade_u = normalize_suffix(trade_u)
    quote_u = normalize_suffix(quote_u)
    return trade_u.join(quote_u, on=["sym_root", "sym_suffix"], how="inner")


def load_quotes(quote_path: Path, universe: pl.DataFrame) -> pl.DataFrame:
    q = pl.read_parquet(quote_path, columns=QUOTE_COLS)
    q = add_ts_ns_quote(q)
    q = add_minute_from_ts(q)
    q = filter_session(q)
    q = normalize_suffix(q)
    q = q.join(universe, on=["sym_root", "sym_suffix"], how="inner")
    q = q.filter(
        (pl.col("best_bid") > 0)
        & (pl.col("best_ask") > 0)
        & (pl.col("best_bid") <= pl.col("best_ask"))
    )
    q = q.with_columns(
        pl.col("best_bid").cast(pl.Float32),
        pl.col("best_ask").cast(pl.Float32),
        pl.col("best_bidsiz").cast(pl.Float32),
        pl.col("best_asksiz").cast(pl.Float32),
    )
    q = q.with_columns(
        (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
        ((pl.col("best_bid") + pl.col("best_ask")) * 0.5).alias("mid"),
        (
            (pl.col("best_bidsiz") - pl.col("best_asksiz"))
            / (pl.col("best_bidsiz") + pl.col("best_asksiz"))
        ).alias("imb"),
        pl.col("date").cast(pl.Utf8).alias("date"),
    )
    return q.sort(["sym_root", "sym_suffix", "time_m", "time_m_nano"])


def _compute_ticker_frame_from_arrays(
    date_str: str,
    sym_root: str,
    sym_suffix: str,
    ts: np.ndarray,
    spread: np.ndarray,
    mid: np.ndarray,
    imb: np.ndarray,
) -> pl.DataFrame | None:
    if ts.shape[0] == 0:
        return None
    out = _compute_ticker(ts, spread, mid, imb, SESSION_MINUTES_ARR)
    if out[0].shape[0] == 0:
        return None

    suffix_out = None if sym_suffix == "None" else sym_suffix
    n_rows = out[0].shape[0]
    return pl.DataFrame(
        {
            "date": [date_str] * n_rows,
            "sym_root": [sym_root] * n_rows,
            "sym_suffix": pl.Series([suffix_out] * n_rows, dtype=pl.Utf8),
            "minute": out[0],
            "spread_first": out[1].astype(np.float32),
            "spread_last": out[2].astype(np.float32),
            "spread_max": out[3].astype(np.float32),
            "spread_min": out[4].astype(np.float32),
            "spread_mean": out[5].astype(np.float32),
            "spread_std": out[6].astype(np.float32),
            "mid_price_first": out[7].astype(np.float32),
            "mid_price_last": out[8].astype(np.float32),
            "mid_price_max": out[9].astype(np.float32),
            "mid_price_min": out[10].astype(np.float32),
            "mid_price_mean": out[11].astype(np.float32),
            "mid_price_std": out[12].astype(np.float32),
            "imbalance_mean": out[13].astype(np.float32),
            "imbalance_std": out[14].astype(np.float32),
        }
    )


def _ticker_worker(args: tuple) -> pl.DataFrame | None:
    date_str, sym_root, sym_suffix, ts, spread, mid, imb = args
    return _compute_ticker_frame_from_arrays(date_str, sym_root, sym_suffix, ts, spread, mid, imb)


def _run_ticker_pool(
    tasks: list[tuple],
    max_workers: int,
) -> list[pl.DataFrame]:
    if max_workers <= 1 or len(tasks) <= 1:
        chunks: list[pl.DataFrame] = []
        for args in tasks:
            frame = _ticker_worker(args)
            if frame is not None:
                chunks.append(frame)
        return chunks

    chunks = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for frame in pool.map(_ticker_worker, tasks, chunksize=max(1, len(tasks) // (max_workers * 4))):
            if frame is not None:
                chunks.append(frame)
    return chunks


def compute_one_day(
    quote_path: Path,
    trade_path: Path,
    tick_workers: int = 1,
) -> pl.DataFrame:
    """Compute simple_factors for one trading day."""
    _warmup_numba()
    universe = load_universe(trade_path, quote_path)
    quotes = load_quotes(quote_path, universe)
    if quotes.is_empty():
        return pl.DataFrame(schema={c: pl.Utf8 for c in OUTPUT_COLS})

    date_str = str(quotes["date"][0])
    groups = quotes.partition_by(["sym_root", "sym_suffix"], as_dict=True)
    tasks: list[tuple] = []
    for (sym_root, sym_suffix), sub in groups.items():
        tasks.append(
            (
                date_str,
                sym_root,
                sym_suffix,
                sub["ts_ns"].to_numpy().astype(np.int64),
                sub["spread"].to_numpy().astype(np.float32).astype(np.float64),
                sub["mid"].to_numpy().astype(np.float32).astype(np.float64),
                sub["imb"].to_numpy().astype(np.float32).astype(np.float64),
            )
        )

    chunks = _run_ticker_pool(tasks, tick_workers)
    if not chunks:
        return pl.DataFrame(schema={c: pl.Utf8 for c in OUTPUT_COLS})

    out = pl.concat(chunks).select(OUTPUT_COLS)
    return out.sort(["sym_root", "sym_suffix", "minute"])


def write_day(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
