"""Feature engineering: intraday lag shift for execution-delay simulation."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def create_lagged_features(
    df: pd.DataFrame,
    lag: int,
    factor_cols: list[str],
    *,
    stock_col: str = "stock_code",
) -> pd.DataFrame:
    """
    Shift factor columns by ``lag`` minutes within each (stock, date) group.
    Label column is not shifted. Rows with NaN from shift are dropped.
    """
    if lag == 0:
        return df.copy()

    required = {stock_col, "date", "minute", *factor_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for lag shift: {missing}")

    out = df.sort_values([stock_col, "date", "minute"]).copy()
    shifted_parts: list[pd.DataFrame] = []

    for (_, _), grp in out.groupby([stock_col, "date"], sort=False):
        g = grp.sort_values("minute").copy()
        for col in factor_cols:
            g[col] = g[col].shift(lag)
        shifted_parts.append(g)

    result = pd.concat(shifted_parts, ignore_index=True)
    before = len(result)
    result = result.dropna(subset=factor_cols)
    dropped = before - len(result)
    if dropped > 0:
        warnings.warn(
            f"Lag={lag}: dropped {dropped:,} rows ({dropped / max(before, 1):.1%}) after intraday shift.",
            stacklevel=2,
        )
    return result.reset_index(drop=True)


def get_factor_columns(panel: pd.DataFrame, label_col: str) -> list[str]:
    """Infer factor column names by excluding metadata and label columns."""
    exclude = {
        "date", "sym_root", "sym_suffix", "minute", "ticker", "stock_code",
        "datetime", label_col, "_training_label", "_relevance_label",
        "ret_10m", "ex_ret_10m", "log_ret_10m", "ex_log_ret_10m",
    }
    numeric = [
        c for c in panel.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(panel[c])
    ]
    return sorted(numeric)
