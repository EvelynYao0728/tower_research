"""IO with auto-detect: parquet/feather, long/wide, file/directory.

Public datasets in this project are *long* parquet, partitioned by date:
    columns: date, sym_root, sym_suffix, minute, <factor or label cols>

The design spec also describes *wide* feather:
    columns: Datetime, A, AAPL, ..., Z

Both layouts are accepted transparently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.feather as pf

PARQUET_EXTS = (".parquet", ".pq")
FEATHER_EXTS = (".fea", ".feather")
DATA_EXTS = PARQUET_EXTS + FEATHER_EXTS

LONG_KEYS = ("date", "sym_root", "sym_suffix", "minute")
WIDE_TIME_CANDS = ("Datetime", "datetime", "DATETIME", "time", "Time")


# --------------------------------------------------------------------------- #
# format detection                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataSource:
    path: Path
    is_dir: bool
    is_long: bool
    files: Tuple[Path, ...]
    columns: Tuple[str, ...]
    file_kind: str          # "parquet" or "feather"
    time_col: Optional[str] # for wide format


def _ext_kind(p: Path) -> str:
    s = p.suffix.lower()
    if s in PARQUET_EXTS:
        return "parquet"
    if s in FEATHER_EXTS:
        return "feather"
    raise ValueError(f"Unsupported extension: {p}")


def _peek_columns(p: Path) -> Tuple[str, ...]:
    if _ext_kind(p) == "parquet":
        return tuple(pq.ParquetFile(p).schema_arrow.names)
    return tuple(pf.read_table(p, columns=None).schema.names)


def detect(path: Path) -> DataSource:
    """Inspect a file or directory and return a typed DataSource descriptor."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        files: List[Path] = []
        for ext in DATA_EXTS:
            files.extend(p.glob(f"*{ext}"))
        if not files:
            raise FileNotFoundError(f"No parquet/feather files under {p}")
        files.sort()
        sample = files[0]
        cols = _peek_columns(sample)
        kind = _ext_kind(sample)
        is_long = all(c in cols for c in LONG_KEYS)
        time_col = next((c for c in WIDE_TIME_CANDS if c in cols), None)
        return DataSource(p, True, is_long, tuple(files), cols, kind, time_col)

    cols = _peek_columns(p)
    kind = _ext_kind(p)
    is_long = all(c in cols for c in LONG_KEYS)
    time_col = next((c for c in WIDE_TIME_CANDS if c in cols), None)
    return DataSource(p, False, is_long, (p,), cols, kind, time_col)


# --------------------------------------------------------------------------- #
# load helpers                                                                #
# --------------------------------------------------------------------------- #
def _read_table(p: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if _ext_kind(p) == "parquet":
        return pq.read_table(p, columns=list(columns) if columns else None).to_pandas(
            self_destruct=True, types_mapper=pd.ArrowDtype
        ).astype(
            {c: "float32" for c in (columns or []) if c not in LONG_KEYS},
            errors="ignore",
        )
    tbl = pf.read_table(p, columns=list(columns) if columns else None)
    return tbl.to_pandas(self_destruct=True)


def read_long_day(
    p: Path,
    factor_or_label_cols: Sequence[str],
) -> pd.DataFrame:
    """Read a single long-format file with column projection (parquet or feather).

    Returns a DataFrame with columns: date, sym_root, sym_suffix, minute, <cols...>
    """
    needed = list(LONG_KEYS) + [c for c in factor_or_label_cols if c not in LONG_KEYS]
    if _ext_kind(p) == "parquet":
        tbl = pq.read_table(p, columns=needed)
    else:
        tbl = pf.read_table(p, columns=needed)
    df = tbl.to_pandas()
    # downcast numeric value cols to float32 to save RAM/CPU
    for c in factor_or_label_cols:
        if c in df.columns and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32", copy=False)
    return df


def add_ticker_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical ticker = sym_root[.sym_suffix] in long-format frame."""
    if "ticker" in df.columns:
        return df
    suf = df["sym_suffix"]
    suf_str = suf.astype("string").fillna("")
    suf_str = suf_str.where(~suf_str.eq("None"), "")
    root = df["sym_root"].astype("string")
    df["ticker"] = np.where(
        suf_str.eq(""), root.to_numpy(), (root + "." + suf_str).to_numpy()
    )
    df["ticker"] = df["ticker"].astype("category")
    return df


def list_long_days(src: DataSource) -> List[Path]:
    """For a directory of per-day long files, return list of files (sorted)."""
    if not src.is_dir or not src.is_long:
        raise ValueError("list_long_days expects a long-format directory source")
    return list(src.files)


def read_wide(
    src: DataSource, value_name: str
) -> pd.DataFrame:
    """Load a wide-format single file and convert to long.

    Returns columns: date, minute, ticker, <value_name>.
    """
    if src.is_dir:
        raise ValueError("Wide format from a directory is not supported.")
    p = src.files[0]
    if src.time_col is None:
        # use first column
        first = src.columns[0]
    else:
        first = src.time_col
    df = _read_table(p)
    df = df.rename(columns={first: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    long = df.melt(id_vars=["Datetime"], var_name="ticker", value_name=value_name)
    long["date"] = long["Datetime"].dt.strftime("%Y-%m-%d")
    long["minute"] = (
        long["Datetime"].dt.hour.astype("int32") * 100
        + long["Datetime"].dt.minute.astype("int32")
    )
    long["ticker"] = long["ticker"].astype("category")
    long[value_name] = long[value_name].astype("float32", copy=False)
    return long[["date", "minute", "ticker", value_name]]


# --------------------------------------------------------------------------- #
# pairing days between factor dir and label dir                               #
# --------------------------------------------------------------------------- #
def _day_stem(p: Path) -> str:
    return p.stem.replace("-", "")


def pair_long_days(
    factor_src: DataSource, label_src: DataSource
) -> List[Tuple[Path, Path]]:
    """Inner-join factor-dir and label-dir by date stem."""
    f_map = {_day_stem(p): p for p in factor_src.files}
    l_map = {_day_stem(p): p for p in label_src.files}
    common = sorted(set(f_map) & set(l_map))
    return [(f_map[d], l_map[d]) for d in common]
