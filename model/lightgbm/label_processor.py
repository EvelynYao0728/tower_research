"""Label winsorize-by-minute only (exp1 / lgb_exp1_winsorize style)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from data import (  # noqa: E402
    TRAINING_LABEL_COL,
    _winsorize_label_by_minute,
    apply_training_label,
)

TRAINING_LABEL_COL_ALIAS = TRAINING_LABEL_COL


def process_label_winsorize(
    panel: pd.DataFrame,
    label_col: str,
    *,
    winsorize_q: float = 0.01,
    output_col: str = TRAINING_LABEL_COL_ALIAS,
) -> pd.DataFrame:
    """Minute-level winsorize (1%-99% quantile per minute across all dates)."""
    out = panel.copy()
    out[output_col] = _winsorize_label_by_minute(out, label_col, winsorize_q).astype("float32")
    return out


def attach_training_label(
    panel: pd.DataFrame,
    label_col: str,
    *,
    winsorize_q: float = 0.01,
) -> tuple[pd.DataFrame, str]:
    """Wrapper around shared apply_training_label for winsorize-only mode (exp1)."""
    out = panel.copy()
    training_col = apply_training_label(
        out,
        label_col,
        label_mode="winsorize",
        label_winsorize_q=winsorize_q,
    )
    return out, training_col
