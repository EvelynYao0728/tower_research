"""Timestamp helpers for TAQ quote/trade streams."""
from __future__ import annotations

import polars as pl

from compute.config import SESSION_END, SESSION_MINUTES, SESSION_START


def add_ts_ns_quote(df: pl.DataFrame) -> pl.DataFrame:
    """Parse quote time_m string (with fractional seconds) to ts_ns."""
    base = (
        pl.col("time_m").str.slice(0, 2).cast(pl.Int64, strict=False) * 3600
        + pl.col("time_m").str.slice(3, 2).cast(pl.Int64, strict=False) * 60
        + pl.col("time_m").str.slice(6, 2).cast(pl.Int64, strict=False)
    ) * 1_000_000_000
    frac_ns = (
        pl.when(pl.col("time_m").str.len_chars() > 8)
        .then(pl.col("time_m").str.slice(8).str.replace(r"^\.?", "").str.pad_end(9, "0"))
        .otherwise(pl.lit("0"))
        .cast(pl.Int64, strict=False)
    )
    return df.with_columns((base + frac_ns).alias("ts_ns"))


def ts_ns_to_minute(ts_ns: pl.Expr) -> pl.Expr:
    total_sec = (ts_ns // 1_000_000_000).cast(pl.Int64)
    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    return (hh * 100 + mm).cast(pl.Int32)


def add_minute_from_ts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(ts_ns_to_minute(pl.col("ts_ns")).alias("minute"))


def filter_session(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("minute").is_in(SESSION_MINUTES))


def filter_session_range(df: pl.DataFrame) -> pl.DataFrame:
    """Keep regular session minutes [SESSION_START, SESSION_END] (O 族全网格)."""
    return df.filter(
        (pl.col("minute") >= SESSION_START) & (pl.col("minute") <= SESSION_END)
    )


def normalize_suffix(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("sym_suffix").is_null() | (pl.col("sym_suffix") == ""))
        .then(pl.lit("None"))
        .otherwise(pl.col("sym_suffix").cast(pl.Utf8))
        .alias("sym_suffix")
    )


def minute_grid() -> pl.DataFrame:
    return pl.DataFrame({"minute": list(range(SESSION_START, SESSION_END + 1))})
