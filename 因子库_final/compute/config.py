"""Paths, session constants, and factor lists for 因子库_final (28 factors)."""
from __future__ import annotations

from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parent.parent
RESEARCH = LIBRARY_ROOT.parent

QUOTE_DIR = RESEARCH / "public" / "Quote_2026"
TRADE_DIR = RESEARCH / "public" / "Trade_2026"
SIMPLE_FACTORS_DIR = RESEARCH / "base_data_process" / "simple_factors"
TRADE_DATE_CSV = RESEARCH / "data" / "trade_date.csv"

SESSION_START = 930
SESSION_END = 1559
ALPHA_END = 1550
EPS = 1e-8

# Lee-Mykland jump detection (O 族 quote 因子)
LM_K = 270
LM_BETA = 4.6
ODD_LOT_THRESHOLD = 100
TICK_SIZE = 0.01

TRADE_UNIVERSE_COLS = ["sym_root", "sym_suffix"]


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
QUOTE_MINUTES: tuple[int, ...] = tuple(m for m in SESSION_MINUTES if m <= ALPHA_END)

# B/C/D/E 族（8）
BCDE_FACTORS: tuple[str, ...] = (
    "B4_cross_dup_price_spread_bps",
    "B5_trade_dup_vwap_premium",
    "C3_oir",
    "C5_layer_stacking_x_flow_signed_1m",
    "C8_spread_asymm_vol",
    "D3_nbbo_size_imb",
    "D5_spread_diff_mean",
    "E6_trade_position_skew",
)

# MB + S 族（15）
MB_FACTORS: tuple[str, ...] = (
    "imbalance_mean",
    "imb_current",
    "imb_trend",
    "clv",
    "clv_x_imb",
    "spread_x_imb",
    "vol_adj_imb",
    "open_mean_dev",
    "ret_1m",
    "ret_5m",
    "ret_10m_past",
    "liq_adj_ret",
    "range_pos_5m",
    "dist_from_5m_high",
    "dist_from_5m_low",
)

# O 族（5）
O_FACTORS: tuple[str, ...] = (
    "odd_lot_ofi",
    "mid_trade_share",
    "mid_trade_vol_share",
    "signed_jump_var",
    "realized_skew_tick",
)

ALL_FACTORS: tuple[str, ...] = BCDE_FACTORS + MB_FACTORS + O_FACTORS

FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "bcde": BCDE_FACTORS,
    "mb": MB_FACTORS,
    "o": O_FACTORS,
}

KEYS = ("date", "sym_root", "sym_suffix", "minute")
