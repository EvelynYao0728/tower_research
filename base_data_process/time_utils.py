"""Timestamp helpers for TAQ quote streams."""
from __future__ import annotations

import polars as pl

from base_data_process.config import SESSION_END, SESSION_MINUTES, SESSION_START


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
    """Keep quotes in valid RTH minutes (excludes 960-999 etc.)."""
    return df.filter(pl.col("minute").is_in(SESSION_MINUTES))


def normalize_suffix(df: pl.DataFrame) -> pl.DataFrame:
    """Internal key uses string 'None'; output converts back to null."""
    return df.with_columns(
        pl.when(pl.col("sym_suffix").is_null() | (pl.col("sym_suffix") == ""))
        .then(pl.lit("None"))
        .otherwise(pl.col("sym_suffix").cast(pl.Utf8))
        .alias("sym_suffix")
    )


def minute_start_ns(minute: int) -> int:
    hh, mm = divmod(int(minute), 100)
    return (hh * 3600 + mm * 60) * 1_000_000_000


def minute_end_ns(minute: int) -> int:
    return minute_start_ns(minute) + 60_000_000_000


def session_end_ns() -> int:
    return minute_end_ns(SESSION_END)
