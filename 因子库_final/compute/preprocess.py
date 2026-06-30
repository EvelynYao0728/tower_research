"""Load and preprocess TAQ data for O-family factors (full 630-minute grid)."""
from __future__ import annotations

import polars as pl

from compute.config import SESSION_END, SESSION_START
from compute.time_utils import (
    add_minute_from_ts,
    add_ts_ns_quote,
    filter_session_range,
    minute_grid,
    normalize_suffix,
)

QUOTE_COLS = [
    "date",
    "time_m",
    "time_m_nano",
    "sym_root",
    "sym_suffix",
    "best_bid",
    "best_ask",
    "best_bidsiz",
    "best_asksiz",
]

TRADE_COLS = [
    "date",
    "time_m",
    "time_m_nano",
    "sym_root",
    "sym_suffix",
    "price",
    "size",
    "ex",
]


def add_ts_ns_trade(df: pl.DataFrame) -> pl.DataFrame:
    h = pl.col("time_m").dt.hour().cast(pl.Int64)
    m = pl.col("time_m").dt.minute().cast(pl.Int64)
    s = pl.col("time_m").dt.second().cast(pl.Int64)
    sub = pl.col("time_m").dt.nanosecond().cast(pl.Int64)
    base = (h * 3600 + m * 60 + s) * 1_000_000_000 + sub
    return df.with_columns((base + pl.col("time_m_nano").fill_null(0)).alias("ts_ns"))


def load_quotes(day_path: str) -> pl.DataFrame:
    q = pl.read_parquet(day_path, columns=QUOTE_COLS)
    q = add_ts_ns_quote(q)
    q = add_minute_from_ts(q)
    q = filter_session_range(q)
    q = normalize_suffix(q)
    q = q.filter(
        (pl.col("best_bid") > 0)
        & (pl.col("best_ask") > 0)
        & (pl.col("best_bid") < pl.col("best_ask"))
    )
    q = q.with_columns(
        ((pl.col("best_bid") + pl.col("best_ask")) * 0.5).alias("mid"),
        pl.col("date").cast(pl.Utf8).alias("date"),
    )
    return q.sort(["sym_root", "sym_suffix", "ts_ns"])


def load_trades(day_path: str) -> pl.DataFrame:
    t = pl.read_parquet(day_path, columns=TRADE_COLS)
    t = add_ts_ns_trade(t)
    t = add_minute_from_ts(t)
    t = filter_session_range(t)
    t = normalize_suffix(t)
    t = t.filter((pl.col("price") > 0) & (pl.col("size") > 0))
    t = t.with_columns(
        pl.col("date").cast(pl.Utf8).alias("date"),
        (pl.col("price") * pl.col("size")).alias("dollar_vol"),
    )
    return t.sort(["sym_root", "sym_suffix", "ts_ns"])


def nbbo_mid_series(quotes: pl.DataFrame) -> pl.DataFrame:
    keys = ["sym_root", "sym_suffix", "ts_ns"]
    changed = (
        (pl.col("best_bid") != pl.col("best_bid").shift(1).over(["sym_root", "sym_suffix"]))
        | (pl.col("best_bidsiz") != pl.col("best_bidsiz").shift(1).over(["sym_root", "sym_suffix"]))
        | (pl.col("best_ask") != pl.col("best_ask").shift(1).over(["sym_root", "sym_suffix"]))
        | (pl.col("best_asksiz") != pl.col("best_asksiz").shift(1).over(["sym_root", "sym_suffix"]))
    )
    first = pl.arange(0, pl.len()).over(["sym_root", "sym_suffix"]) == 0
    return quotes.filter(changed | first).select(*keys, "minute", "date", "mid", "best_bid", "best_ask")


def enrich_trades(trades: pl.DataFrame, quotes: pl.DataFrame) -> pl.DataFrame:
    q_mid = quotes.select("sym_root", "sym_suffix", "ts_ns", "mid")
    t = trades.join_asof(
        q_mid,
        on="ts_ns",
        by=["sym_root", "sym_suffix"],
        strategy="backward",
    )
    t = t.with_columns(
        pl.when(pl.col("price") > pl.col("mid"))
        .then(pl.lit(1))
        .when(pl.col("price") < pl.col("mid"))
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("_lr0")
    )
    t = t.with_columns(
        pl.col("price").shift(1).over(["sym_root", "sym_suffix"]).alias("_prev_price")
    )
    sign = (
        pl.when(pl.col("_lr0") != 0)
        .then(pl.col("_lr0"))
        .when(pl.col("price") > pl.col("_prev_price"))
        .then(pl.lit(1))
        .when(pl.col("price") < pl.col("_prev_price"))
        .then(pl.lit(-1))
        .otherwise(pl.lit(1))
    )
    return t.with_columns(sign.alias("sign").cast(pl.Int8)).drop("_lr0", "_prev_price")


def ticker_universe(trades: pl.DataFrame, quotes: pl.DataFrame) -> pl.DataFrame:
    u1 = trades.select("sym_root", "sym_suffix").unique()
    u2 = quotes.select("sym_root", "sym_suffix").unique()
    return pl.concat([u1, u2]).unique()


def expand_to_grid(
    df: pl.DataFrame,
    universe: pl.DataFrame,
    date_str: str,
    value_cols: list[str],
) -> pl.DataFrame:
    grid = universe.join(minute_grid(), how="cross").with_columns(pl.lit(date_str).alias("date"))
    keys = ["date", "sym_root", "sym_suffix", "minute"]
    out = grid.join(df.select(keys + value_cols), on=keys, how="left")
    for c in value_cols:
        out = out.with_columns(pl.col(c).cast(pl.Float32))
    return out
