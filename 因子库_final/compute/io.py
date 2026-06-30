"""Shared TAQ load, universe, and grid helpers for L4 factors."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from compute.config import SESSION_MINUTES, TRADE_UNIVERSE_COLS
from compute.time_utils import (
    add_minute_from_ts,
    add_ts_ns_quote,
    filter_session,
    normalize_suffix,
)

QUOTE_COLS = [
    "date",
    "time_m",
    "time_m_nano",
    "sym_root",
    "sym_suffix",
    "bid",
    "ask",
    "bidsiz",
    "asksiz",
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

TRADE_COLS_ALPHA = [
    "date",
    "time_m",
    "time_m_nano",
    "sym_root",
    "sym_suffix",
    "price",
    "size",
    "tr_scond",
    "tr_corr",
    "tr_seqnum",
]

QUOTE_COLS_ALPHA = [
    "date",
    "time_m",
    "time_m_nano",
    "sym_root",
    "sym_suffix",
    "bid",
    "ask",
    "bidsiz",
    "asksiz",
    "best_bid",
    "best_ask",
    "best_bidsiz",
    "best_asksiz",
    "qu_cond",
    "qu_cancel",
    "secstat_ind",
    "luld_nbbo_indicator",
]

QUOTE_MINUTES: tuple[int, ...] = tuple(m for m in SESSION_MINUTES if m <= 1550)
TRADE_MINUTES: tuple[int, ...] = SESSION_MINUTES

EPS_T = 1e-9
EPS_M = 1e-8


def _suffix_none_str(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("sym_suffix").is_null() | (pl.col("sym_suffix") == ""))
        .then(pl.lit("None"))
        .otherwise(pl.col("sym_suffix").cast(pl.Utf8))
        .alias("sym_suffix")
    )


def _suffix_empty_str(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("sym_suffix").is_null() | (pl.col("sym_suffix") == "None"))
        .then(pl.lit(""))
        .otherwise(pl.col("sym_suffix").cast(pl.Utf8))
        .alias("sym_suffix")
    )


def load_universe(trade_path: Path, quote_path: Path) -> pl.DataFrame:
    cols = list(TRADE_UNIVERSE_COLS)
    trade_u = pl.scan_parquet(trade_path).select(cols).unique().collect()
    quote_u = pl.scan_parquet(quote_path).select(cols).unique().collect()
    trade_u = normalize_suffix(trade_u)
    quote_u = normalize_suffix(quote_u)
    return trade_u.join(quote_u, on=["sym_root", "sym_suffix"], how="inner")


def add_ts_ns_trade(df: pl.DataFrame) -> pl.DataFrame:
    h = pl.col("time_m").dt.hour().cast(pl.Int64)
    m = pl.col("time_m").dt.minute().cast(pl.Int64)
    s = pl.col("time_m").dt.second().cast(pl.Int64)
    sub = pl.col("time_m").dt.nanosecond().cast(pl.Int64)
    base = (h * 3600 + m * 60 + s) * 1_000_000_000 + sub
    return df.with_columns((base + pl.col("time_m_nano").fill_null(0)).alias("ts_ns"))


def load_quotes(quote_path: Path, universe: pl.DataFrame) -> pl.DataFrame:
    q = pl.read_parquet(quote_path, columns=QUOTE_COLS)
    return _finalize_quotes(q, universe)


def load_quotes_alpha(quote_path: Path, universe: pl.DataFrame) -> pl.DataFrame:
    """alpha_generate quote filters."""
    q = (
        pl.scan_parquet(quote_path)
        .select(QUOTE_COLS_ALPHA)
        .filter(
            (pl.col("qu_cancel") != "B")
            & (pl.col("best_bid") > 0)
            & (pl.col("best_ask") > 0)
            & (
                pl.col("secstat_ind").is_null()
                | pl.col("secstat_ind").cast(pl.Utf8).is_in(["", "None", "nan"])
            )
            & (
                pl.col("luld_nbbo_indicator").is_null()
                | pl.col("luld_nbbo_indicator").cast(pl.Utf8).is_in(["A", "None", "nan", ""])
            )
            & ~pl.col("qu_cond").cast(pl.Utf8).is_in(["N", "H", "Z"])
        )
        .collect()
    )
    return _finalize_quotes(q, universe)


def _finalize_quotes(q: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
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
        pl.col("bid").cast(pl.Float32),
        pl.col("ask").cast(pl.Float32),
        pl.col("bidsiz").cast(pl.Float32),
        pl.col("asksiz").cast(pl.Float32),
        pl.col("date").cast(pl.Utf8),
    )
    return q.sort(["sym_root", "sym_suffix", "ts_ns"])


def load_trades(trade_path: Path, universe: pl.DataFrame) -> pl.DataFrame:
    t = pl.read_parquet(trade_path, columns=TRADE_COLS)
    return _finalize_trades(t, universe)


def load_trades_alpha(trade_path: Path, universe: pl.DataFrame) -> pl.DataFrame:
    """alpha_generate trade filters: tr_corr==00, exclude T/U sale condition."""
    t = (
        pl.scan_parquet(trade_path)
        .select(TRADE_COLS_ALPHA)
        .filter(
            (pl.col("tr_corr") == "00")
            & ~pl.col("tr_scond").cast(pl.Utf8).str.contains("T|U")
            & (pl.col("price") > 0)
            & (pl.col("size") > 0)
        )
        .collect()
    )
    return _finalize_trades(t, universe)


def _finalize_trades(t: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    t = add_ts_ns_trade(t)
    t = add_minute_from_ts(t)
    t = filter_session(t)
    t = normalize_suffix(t)
    t = t.join(universe, on=["sym_root", "sym_suffix"], how="inner")
    t = t.filter((pl.col("price") > 0) & (pl.col("size") > 0))
    t = t.with_columns(
        pl.col("price").cast(pl.Float32),
        pl.col("size").cast(pl.Float32),
        pl.col("date").cast(pl.Utf8),
    )
    return t.sort(["sym_root", "sym_suffix", "ts_ns"])


def expand_grid(
    universe: pl.DataFrame,
    date_str: str,
    minutes: tuple[int, ...],
    suffix_mode: str = "none",
) -> pl.DataFrame:
    u = _suffix_empty_str(universe) if suffix_mode == "empty" else _suffix_none_str(universe)
    return (
        u.join(pl.DataFrame({"minute": list(minutes)}), how="cross")
        .with_columns(pl.lit(date_str).alias("date"))
    )


def activity_grid(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    max_minute: int | None = None,
) -> pl.DataFrame:
    """Minute keys: union of trade and quote activity (matches ofi_1m grid)."""
    t = trades.select("date", "sym_root", "sym_suffix", "minute")
    q = quotes.select("date", "sym_root", "sym_suffix", "minute")
    g = pl.concat([t, q]).unique()
    if max_minute is not None:
        g = g.filter(pl.col("minute") <= max_minute)
    return g


def quote_factor_grid(trades: pl.DataFrame) -> pl.DataFrame:
    """C/B/D/E/N quote-class grid: trade minutes capped at 15:50 (matches 因子库_final)."""
    g = trades.select("date", "sym_root", "sym_suffix", "minute").unique()
    g = g.filter(pl.col("minute") <= 1550)
    # Reference excludes this single trade-only minute (data quirk on 20250102).
    g = g.filter(~((pl.col("sym_root") == "DAY") & (pl.col("minute") == 939)))
    return g


def attach_activity_grid(
    sparse: pl.DataFrame,
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    value_cols: list[str],
    max_minute: int | None = None,
    suffix_mode: str = "none",
) -> pl.DataFrame:
    grid = activity_grid(trades, quotes, max_minute=max_minute)
    return _join_grid(sparse, grid, value_cols, suffix_mode)


def attach_trade_grid(
    sparse: pl.DataFrame,
    trades: pl.DataFrame,
    value_cols: list[str],
    max_minute: int | None = None,
    suffix_mode: str = "none",
) -> pl.DataFrame:
    if max_minute == 1550:
        grid = quote_factor_grid(trades)
    else:
        grid = trades.select("date", "sym_root", "sym_suffix", "minute").unique()
        if max_minute is not None:
            grid = grid.filter(pl.col("minute") <= max_minute)
    return _join_grid(sparse, grid, value_cols, suffix_mode)


def _join_grid(
    sparse: pl.DataFrame,
    grid: pl.DataFrame,
    value_cols: list[str],
    suffix_mode: str,
) -> pl.DataFrame:
    if suffix_mode == "empty":
        grid = grid.with_columns(
            pl.when(pl.col("sym_suffix") == "None")
            .then(pl.lit(""))
            .otherwise(pl.col("sym_suffix"))
            .alias("sym_suffix")
        )
    keys = ["date", "sym_root", "sym_suffix", "minute"]
    out = grid.join(sparse.select(keys + value_cols), on=keys, how="left")
    for c in value_cols:
        if sparse.schema.get(c) == pl.Float64:
            out = out.with_columns(pl.col(c).cast(pl.Float64))
        else:
            out = out.with_columns(pl.col(c).cast(pl.Float32))
    return out


def attach_grid(
    sparse: pl.DataFrame,
    universe: pl.DataFrame,
    date_str: str,
    minutes: tuple[int, ...],
    value_cols: list[str],
    suffix_mode: str = "none",
) -> pl.DataFrame:
    grid = expand_grid(universe, date_str, minutes, suffix_mode=suffix_mode)
    return _join_grid(sparse, grid, value_cols, suffix_mode)


def lee_ready_sign(trades: pl.DataFrame, quotes: pl.DataFrame) -> pl.DataFrame:
    q_mid = quotes.select("sym_root", "sym_suffix", "ts_ns", "mid")
    t = trades.join_asof(
        q_mid,
        on="ts_ns",
        by=["sym_root", "sym_suffix"],
        strategy="backward",
    )
    t = t.with_columns(
        pl.when(pl.col("price").cast(pl.Float32) > pl.col("mid"))
        .then(1)
        .when(pl.col("price").cast(pl.Float32) < pl.col("mid"))
        .then(-1)
        .otherwise(0)
        .alias("_lr0")
    )
    t = t.with_columns(pl.col("price").shift(1).over(["sym_root", "sym_suffix"]).alias("_pp"))
    sign = (
        pl.when(pl.col("_lr0") != 0)
        .then(pl.col("_lr0"))
        .when(pl.col("price") > pl.col("_pp"))
        .then(1)
        .when(pl.col("price") < pl.col("_pp"))
        .then(-1)
        .otherwise(1)
    )
    return t.with_columns(sign.cast(pl.Int8).alias("sign")).drop("_lr0", "_pp")


def prepare_quotes_mid(quotes: pl.DataFrame) -> pl.DataFrame:
    return quotes.with_columns(
        (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
        ((pl.col("best_bid") + pl.col("best_ask")) * 0.5).alias("mid"),
        (
            (pl.col("best_bidsiz") - pl.col("best_asksiz")
            ) / (pl.col("best_bidsiz") + pl.col("best_asksiz"))
        ).cast(pl.Float32).alias("imb"),
        (
            (pl.col("best_bid") * pl.col("best_asksiz") + pl.col("best_ask") * pl.col("best_bidsiz"))
            / (pl.col("best_bidsiz") + pl.col("best_asksiz"))
        ).alias("microprice"),
    )
