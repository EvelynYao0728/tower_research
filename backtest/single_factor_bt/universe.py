"""Point-in-time S&P 500 membership filter.

Universe JSON schema (e.g. /home/hchen.25/public/2025_universe.json):

    {
        "metadata": {...},
        "universe": [
            {"ticker": "A", "wrds_sym_root": "A", "wrds_sym_suffix": null,
             "in_date": "2025-01-02", "out_date": null, ...},
            ...
        ]
    }

A ticker is active on date d iff:
    in_date <= d < (out_date or "9999-12-31")

This mirrors tower_research's apply_pit_membership.py so the two frameworks
can be aligned on stock universe.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd


def _canonical_suffix(s: pd.Series) -> pd.Series:
    """Collapse {None, 'None', '', null} representations of "no suffix" to ''."""
    out = s.astype("string").fillna("")
    out = out.where(~out.eq("None"), "")
    return out


@lru_cache(maxsize=8)
def load_pit_membership_table(universe_file: str | Path) -> pd.DataFrame:
    """Load universe JSON into [sym_root, sym_suffix, in_date, out_date].

    Cached so repeated calls in the same worker process re-parse the JSON
    only once. sym_suffix is canonicalized to '' for missing values, matching
    add_ticker_inplace's normalization.
    """
    p = Path(universe_file)
    raw = json.loads(p.read_text())
    rows = raw["universe"]
    df = pd.DataFrame(rows)
    df["sym_suffix"] = _canonical_suffix(df["wrds_sym_suffix"])
    df["sym_root"] = df["wrds_sym_root"].astype("string")
    df["in_date"] = df["in_date"].astype("string")
    df["out_date"] = df["out_date"].astype("string")  # NaN string for nulls
    return df[["sym_root", "sym_suffix", "in_date", "out_date"]].copy()


def apply_pit_membership_filter(
    bars: pd.DataFrame,
    universe_table: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Inner-join `bars` with `universe_table` on (sym_root, sym_suffix); keep
    rows where in_date <= bars[date] < (out_date or +inf).

    Both sides have sym_suffix normalized to '' for the missing case so the
    join works regardless of how the source files encode "no suffix".
    """
    if date_col not in bars.columns:
        return bars
    if "sym_root" not in bars.columns or "sym_suffix" not in bars.columns:
        raise ValueError(
            "PIT filter requires sym_root + sym_suffix columns in `bars`"
        )

    b = bars.copy()
    b["sym_root"] = b["sym_root"].astype("string")
    b["sym_suffix"] = _canonical_suffix(b["sym_suffix"])
    b[date_col] = b[date_col].astype("string")

    joined = b.merge(
        universe_table, on=["sym_root", "sym_suffix"], how="inner", copy=False,
    )
    keep = (joined[date_col] >= joined["in_date"]) & (
        joined["out_date"].isna() | (joined[date_col] < joined["out_date"])
    )
    return joined.loc[keep].drop(columns=["in_date", "out_date"])
