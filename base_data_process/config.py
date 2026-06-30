"""Paths and session constants for simple_factors generation."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent

QUOTE_DIR_2026 = RESEARCH / "public" / "Quote_2026"
TRADE_DIR_2026 = RESEARCH / "public" / "Trade_2026"
OUTPUT_DIR = RESEARCH / "base_data_process" / "simple_factors"

SIMPLE_FACTORS_DIR_2025 = RESEARCH / "data" / "simple_factors"
SIMPLE_FACTORS_DIR_2026 = OUTPUT_DIR
LABEL_OUTPUT_DIR = RESEARCH / "base_data_process" / "label"
LABEL_REF_DIR = RESEARCH / "public" / "label"

LABEL_HORIZON = 10
LABEL_COLS = [
    "date",
    "sym_root",
    "sym_suffix",
    "minute",
    "ret_10m",
    "ex_ret_10m",
    "log_ret_10m",
    "ex_log_ret_10m",
]

SESSION_START = 930
SESSION_END = 1559


def build_session_minutes(start: int = SESSION_START, end: int = SESSION_END) -> tuple[int, ...]:
    """Valid RTH minute ids: 930-959, 1000-1059, ..., 1500-1559 (skip 960-999)."""
    out: list[int] = []
    start_h, start_m = divmod(start, 100)
    end_h, end_m = divmod(end, 100)
    for h in range(start_h, end_h + 1):
        m_begin = start_m if h == start_h else 0
        m_end = end_m if h == end_h else 59
        for m in range(m_begin, m_end + 1):
            out.append(h * 100 + m)
    return tuple(out)


SESSION_MINUTES = build_session_minutes()
MINUTES_PER_SESSION = len(SESSION_MINUTES)

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

TRADE_UNIVERSE_COLS = ["sym_root", "sym_suffix"]

OUTPUT_COLS = [
    "date",
    "sym_root",
    "sym_suffix",
    "minute",
    "spread_first",
    "spread_last",
    "spread_max",
    "spread_min",
    "spread_mean",
    "spread_std",
    "mid_price_first",
    "mid_price_last",
    "mid_price_max",
    "mid_price_min",
    "mid_price_mean",
    "mid_price_std",
    "imbalance_mean",
    "imbalance_std",
]
