"""Compute minute labels from simple_factors mid_price_last (10-row mid-to-mid)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from base_data_process.config import LABEL_COLS, LABEL_HORIZON


def _cs_mean_f32(series: pd.Series) -> np.float32:
    return np.float32(np.mean(series.astype(np.float64)))


def compute_labels_from_frame(sf: pd.DataFrame) -> pd.DataFrame:
    """Build one-day label frame from simple_factors rows."""
    need = {"date", "sym_root", "sym_suffix", "minute", "mid_price_last"}
    missing = need - set(sf.columns)
    if missing:
        raise ValueError(f"simple_factors missing columns: {sorted(missing)}")

    base = (
        pl.from_pandas(sf[list(need)])
        .with_columns(
            pl.col("date").cast(pl.Utf8),
            pl.col("minute").cast(pl.Int32),
            pl.col("mid_price_last").cast(pl.Float32).alias("mid"),
        )
        .sort(["sym_root", "sym_suffix", "minute"])
        .with_columns(
            pl.col("mid").shift(-LABEL_HORIZON).over(["sym_root", "sym_suffix"]).alias("mid_fwd"),
        )
        .with_columns(
            (pl.col("mid_fwd") / pl.col("mid")).alias("ratio"),
        )
        .with_columns(
            (pl.col("ratio") - 1.0).cast(pl.Float32).alias("ret_10m"),
            pl.col("ratio").log().cast(pl.Float32).alias("log_ret_10m"),
        )
        .select(["date", "sym_root", "sym_suffix", "minute", "ret_10m", "log_ret_10m"])
    )

    pdf = base.to_pandas()
    pdf["ex_ret_10m"] = pdf.groupby("minute", sort=False)["ret_10m"].transform(
        lambda s: s.astype(np.float32) - _cs_mean_f32(s)
    ).astype(np.float32)
    pdf["ex_log_ret_10m"] = pdf.groupby("minute", sort=False)["log_ret_10m"].transform(
        lambda s: s.astype(np.float32) - _cs_mean_f32(s)
    ).astype(np.float32)
    pdf["sym_suffix"] = pdf["sym_suffix"].apply(
        lambda x: "None" if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" or x == "None" else str(x)
    )
    pdf["minute"] = pdf["minute"].astype(np.int32)

    return pdf[LABEL_COLS].sort_values(
        ["sym_root", "sym_suffix", "minute"], kind="mergesort"
    ).reset_index(drop=True)


def compute_one_day(factors_path: Path) -> pd.DataFrame:
    sf = pd.read_parquet(factors_path)
    return compute_labels_from_frame(sf)


def write_day(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pl.from_pandas(
        df,
        schema_overrides={
            "minute": pl.Int32,
            "ret_10m": pl.Float32,
            "ex_ret_10m": pl.Float32,
            "log_ret_10m": pl.Float32,
            "ex_log_ret_10m": pl.Float32,
        },
    )
    table.write_parquet(out_path, compression="zstd")


def compare_label_frames(ref: pd.DataFrame, gen: pd.DataFrame) -> list[str]:
    """Return list of mismatch descriptions; empty means identical."""
    errors: list[str] = []
    if list(ref.columns) != list(gen.columns):
        errors.append(f"columns differ: ref={list(ref.columns)} gen={list(gen.columns)}")
        return errors
    if len(ref) != len(gen):
        errors.append(f"row count differ: ref={len(ref)} gen={len(gen)}")

    keys = ["date", "sym_root", "sym_suffix", "minute"]
    ref_s = ref.sort_values(keys, kind="mergesort").reset_index(drop=True)
    gen_s = gen.sort_values(keys, kind="mergesort").reset_index(drop=True)

    for col in keys:
        if not ref_s[col].equals(gen_s[col]):
            errors.append(f"column {col} differs")
            break

    for col in LABEL_COLS[4:]:
        a = ref_s[col].to_numpy(dtype=np.float32)
        b = gen_s[col].to_numpy(dtype=np.float32)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            n = (~same).sum()
            max_diff = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))
            errors.append(f"{col}: {n} rows differ, max_abs_diff={max_diff}")
    return errors
