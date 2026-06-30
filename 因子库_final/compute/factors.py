"""All 28 factor formulas: B/C/D/E, MB/S, and O families in one module."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from compute.config import (
    ALPHA_END,
    BCDE_FACTORS,
    EPS,
    KEYS,
    LM_BETA,
    LM_K,
    MB_FACTORS,
    ODD_LOT_THRESHOLD,
    O_FACTORS,
    TICK_SIZE,
)
from compute.io import (
    load_quotes_alpha,
    load_trades_alpha,
    load_universe,
    quote_factor_grid,
)
from compute.preprocess import (
    enrich_trades,
    expand_to_grid,
    load_quotes,
    load_trades,
    nbbo_mid_series,
    ticker_universe,
)
from compute.tick_stats import process_ticker_ticks

_EPS = 1e-12
_MINUTE_KEYS = list(KEYS)


# ---------------------------------------------------------------------------
# MB + S 族（15）— 由 simple_factors 二次衍生
# ---------------------------------------------------------------------------

def _suffix_for_output(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.is_null() | (expr.cast(pl.Utf8) == "") | (expr.cast(pl.Utf8) == "None"))
        .then(pl.lit("None"))
        .otherwise(expr.cast(pl.Utf8))
    )


def compute_mb_factors(sf: pl.DataFrame) -> pl.DataFrame:
    """Return long frame with all MB/S factor columns."""
    keys = ["sym_root", "sym_suffix"]
    base = sf.sort(keys + ["minute"]).with_columns(pl.col("date").cast(pl.Utf8))

    base = base.with_columns([
        pl.col("imbalance_mean").cast(pl.Float32),
        ((pl.col("mid_price_last") - pl.col("mid_price_min"))
         / (pl.col("mid_price_max") - pl.col("mid_price_min") + EPS)).cast(pl.Float32).alias("clv"),
        ((pl.col("mid_price_first") - pl.col("mid_price_mean"))
         / (pl.col("mid_price_mean") + EPS)).cast(pl.Float32).alias("open_mean_dev"),
        (pl.col("spread_mean") * pl.col("imbalance_mean")).cast(pl.Float32).alias("spread_x_imb"),
        (pl.col("imbalance_mean") / (pl.col("mid_price_std") / pl.col("mid_price_mean") + EPS))
        .cast(pl.Float32).alias("vol_adj_imb"),
        (pl.col("mid_price_last").cast(pl.Float32) / pl.col("mid_price_first").cast(pl.Float32))
        .log().cast(pl.Float32).alias("ret_1m"),
        (pl.col("mid_price_last").cast(pl.Float32)
         / pl.col("mid_price_last").shift(5).over(keys).cast(pl.Float32))
        .log().cast(pl.Float32).alias("ret_5m"),
        (pl.col("mid_price_last").cast(pl.Float32)
         / pl.col("mid_price_last").shift(10).over(keys).cast(pl.Float32))
        .log().cast(pl.Float32).alias("ret_10m_past"),
    ])

    base = base.with_columns([
        pl.col("imbalance_mean").alias("imb_current"),
        (pl.col("imbalance_mean")
         - pl.col("imbalance_mean").rolling_mean(5, min_samples=1).over(keys))
        .cast(pl.Float32).alias("imb_trend"),
        (pl.col("ret_1m") / (pl.col("spread_mean") / pl.col("mid_price_mean") + EPS))
        .cast(pl.Float32).alias("liq_adj_ret"),
        (pl.col("clv") * pl.col("imbalance_mean")).cast(pl.Float32).alias("clv_x_imb"),
    ])

    base = base.with_columns([
        pl.col("mid_price_max").rolling_max(5, min_samples=1).over(keys).alias("h5"),
        pl.col("mid_price_min").rolling_min(5, min_samples=1).over(keys).alias("l5"),
    ])

    base = base.with_columns([
        ((pl.col("mid_price_last") - pl.col("l5")) / (pl.col("h5") - pl.col("l5") + EPS))
        .cast(pl.Float32).alias("range_pos_5m"),
        ((pl.col("h5") - pl.col("mid_price_last")) / (pl.col("mid_price_mean") + EPS))
        .cast(pl.Float32).alias("dist_from_5m_high"),
        ((pl.col("mid_price_last") - pl.col("l5")) / (pl.col("mid_price_mean") + EPS))
        .cast(pl.Float32).alias("dist_from_5m_low"),
        _suffix_for_output(pl.col("sym_suffix")).alias("sym_suffix"),
    ])

    return base.select(["date", "sym_root", "sym_suffix", "minute", *MB_FACTORS])


# ---------------------------------------------------------------------------
# B/C/D/E 族（8）— TAQ quote/trade
# ---------------------------------------------------------------------------

def _quote_minute_features(quotes: pl.DataFrame) -> pl.DataFrame:
    keys = _MINUTE_KEYS
    q = quotes.sort(["sym_root", "sym_suffix", "ts_ns"]).with_columns([
        (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
        (pl.col("ask").cast(pl.Float32) - pl.col("bid").cast(pl.Float32)).alias("_local_spread"),
        (
            (pl.col("best_bidsiz") - pl.col("best_asksiz"))
            / (pl.col("best_bidsiz") + pl.col("best_asksiz") + _EPS)
        ).cast(pl.Float32).alias("nbbo_imb"),
        pl.col("best_bid").diff().over(["sym_root", "sym_suffix"]).alias("bid_chg"),
        pl.col("best_ask").diff().over(["sym_root", "sym_suffix"]).alias("ask_chg"),
        pl.col("bidsiz").diff().over(["sym_root", "sym_suffix"]).alias("dbs"),
        pl.col("ask").shift(1).over(["sym_root", "sym_suffix"]).alias("_ask_p"),
        pl.col("asksiz").shift(1).over(["sym_root", "sym_suffix"]).alias("_asks_p"),
    ])
    q = q.with_columns(
        (
            (pl.col("dbs") > 0)
            & (pl.col("ask") == pl.col("_ask_p"))
            & (pl.col("asksiz") == pl.col("_asks_p"))
        )
        .cast(pl.Float32)
        .alias("is_layer")
    )
    qm = q.group_by(keys).agg(
        pl.col("nbbo_imb").mean().cast(pl.Float32).alias("nbbo_imb_mean"),
        (pl.col("_local_spread") - pl.col("spread")).mean().cast(pl.Float32).alias("spread_diff_mean"),
        pl.col("is_layer").mean().cast(pl.Float32).alias("layer_stacking"),
        ((pl.col("best_bid") + pl.col("best_ask")) * 0.5).last().cast(pl.Float32).alias("mid_last"),
        pl.col("bid_chg").std().fill_null(0.0).alias("bid_vol"),
        pl.col("ask_chg").std().fill_null(0.0).alias("ask_vol"),
    )
    quote_flow = (
        pl.when(pl.col("nbbo_imb_mean") > 0)
        .then(1.0)
        .when(pl.col("nbbo_imb_mean") < 0)
        .then(-1.0)
        .otherwise(0.0)
    )
    return qm.with_columns([
        pl.col("nbbo_imb_mean").alias("C3_oir"),
        pl.col("nbbo_imb_mean").alias("D3_nbbo_size_imb"),
        pl.col("spread_diff_mean").alias("D5_spread_diff_mean"),
        (pl.col("bid_vol") - pl.col("ask_vol")).cast(pl.Float32).alias("C8_spread_asymm_vol"),
        (pl.col("layer_stacking") * quote_flow).cast(pl.Float32).alias("C5_layer_stacking_x_flow_signed_1m"),
    ])


def _dup_trade_factors(trades: pl.DataFrame, qm: pl.DataFrame) -> pl.DataFrame:
    keys = _MINUTE_KEYS
    t = trades.filter(pl.col("minute") <= ALPHA_END)
    dup_min = (
        t.group_by(["sym_root", "sym_suffix", "minute", "date", "time_m", "time_m_nano"])
        .agg(
            pl.len().alias("_n"),
            (
                (pl.col("price").cast(pl.Float64) * pl.col("size").cast(pl.Float64)).sum()
                / pl.col("size").cast(pl.Float64).sum()
            ).cast(pl.Float32).alias("dup_vwap"),
        )
        .filter(pl.col("_n") > 1)
        .group_by(keys)
        .agg(pl.col("dup_vwap").mean().cast(pl.Float32))
    )
    if dup_min.is_empty():
        schema = {**{k: pl.Utf8 for k in keys}, "B4_cross_dup_price_spread_bps": pl.Float32, "B5_trade_dup_vwap_premium": pl.Float32}
        return pl.DataFrame(schema=schema)

    minute_vwap = t.group_by(keys).agg(
        (
            (pl.col("price").cast(pl.Float64) * pl.col("size").cast(pl.Float64)).sum()
            / pl.col("size").cast(pl.Float64).sum()
        ).cast(pl.Float32).alias("trade_vwap"),
    )
    return (
        dup_min.join(qm.select(keys + ["mid_last"]), on=keys, how="left")
        .join(minute_vwap, on=keys, how="left")
        .with_columns([
            (((pl.col("dup_vwap") - pl.col("mid_last")) / (pl.col("mid_last") + _EPS)) * 10_000.0)
            .cast(pl.Float32)
            .alias("B4_cross_dup_price_spread_bps"),
            (pl.col("dup_vwap") / (pl.col("trade_vwap") + _EPS)).cast(pl.Float32).alias("B5_trade_dup_vwap_premium"),
        ])
        .select(keys + ["B4_cross_dup_price_spread_bps", "B5_trade_dup_vwap_premium"])
    )


def _e6_factor(trades: pl.DataFrame, quotes: pl.DataFrame) -> pl.DataFrame:
    keys = _MINUTE_KEYS
    ba = quotes.filter(pl.col("minute") <= ALPHA_END).group_by(keys).agg(
        pl.col("best_bid").last().cast(pl.Float32).alias("best_bid"),
        pl.col("best_ask").last().cast(pl.Float32).alias("best_ask"),
    )
    return (
        trades.filter(pl.col("minute") <= ALPHA_END)
        .join(ba, on=keys, how="left")
        .filter(pl.col("best_bid").is_not_null())
        .with_columns(
            (
                (pl.col("price").cast(pl.Float32) - pl.col("best_bid"))
                / (pl.col("best_ask") - pl.col("best_bid") + _EPS)
                - 0.5
            )
            .cast(pl.Float32)
            .alias("E6_trade_position_skew")
        )
        .group_by(keys)
        .agg(pl.col("E6_trade_position_skew").mean().cast(pl.Float32))
    )


def compute_bcde_factors(
    trade_path: Path,
    quote_path: Path,
    *,
    ref_root: Path | None = None,
) -> pl.DataFrame:
    """Return wide frame: KEYS + 8 B/C/D/E factor columns."""
    universe = load_universe(trade_path, quote_path)
    trades = load_trades_alpha(trade_path, universe).filter(pl.col("minute") <= ALPHA_END)
    quotes = load_quotes_alpha(quote_path, universe).filter(pl.col("minute") <= ALPHA_END)

    qm = _quote_minute_features(quotes)
    if ref_root is not None:
        template = ref_root / "C3_oir" / f"{trade_path.stem}.parquet"
        grid = pl.read_parquet(template, columns=list(KEYS)) if template.is_file() else quote_factor_grid(trades)
    else:
        grid = quote_factor_grid(trades)

    return (
        grid.join(qm.select(list(KEYS) + [
            "C3_oir",
            "C5_layer_stacking_x_flow_signed_1m",
            "C8_spread_asymm_vol",
            "D3_nbbo_size_imb",
            "D5_spread_diff_mean",
        ]), on=list(KEYS), how="left")
        .join(_dup_trade_factors(trades, qm), on=list(KEYS), how="left")
        .join(_e6_factor(trades, quotes), on=list(KEYS), how="left")
        .select(list(KEYS) + list(BCDE_FACTORS))
    )


# ---------------------------------------------------------------------------
# O 族（5）— 正交 TAQ 因子（630 分钟全网格）
# ---------------------------------------------------------------------------

def _odd_lot_ofi(enriched: pl.DataFrame) -> pl.DataFrame:
    keys = _MINUTE_KEYS
    t = enriched.with_columns((pl.col("size") < ODD_LOT_THRESHOLD).alias("is_odd"))
    odd = t.filter(pl.col("is_odd"))
    buy_vol = odd.filter(pl.col("sign") > 0).group_by(keys).agg(pl.col("size").sum().alias("v_buy"))
    sell_vol = odd.filter(pl.col("sign") < 0).group_by(keys).agg(pl.col("size").sum().alias("v_sell"))
    return (
        buy_vol.join(sell_vol, on=keys, how="full", coalesce=True)
        .with_columns(
            (
                (pl.col("v_buy").fill_null(0) - pl.col("v_sell").fill_null(0))
                / (pl.col("v_buy").fill_null(0) + pl.col("v_sell").fill_null(0) + 1e-9)
            ).alias("odd_lot_ofi")
        )
        .select(*keys, "odd_lot_ofi")
    )


def _mid_trade_factors(enriched: pl.DataFrame) -> pl.DataFrame:
    keys = _MINUTE_KEYS
    half_tick = TICK_SIZE * 0.5
    t = enriched.filter(pl.col("mid").is_not_null()).with_columns(
        ((pl.col("price") - pl.col("mid")).abs() < half_tick).alias("is_mid")
    )
    mid = t.filter(pl.col("is_mid"))
    return (
        t.group_by(keys)
        .agg(pl.len().alias("n_all"), pl.col("size").sum().alias("v_all"))
        .join(
            mid.group_by(keys).agg(
                pl.len().alias("n_mid"),
                pl.col("size").sum().alias("v_mid"),
            ),
            on=keys,
            how="left",
        )
        .with_columns(
            (pl.col("n_mid").fill_null(0) / pl.col("n_all").clip(lower_bound=1)).alias("mid_trade_share"),
            (pl.col("v_mid").fill_null(0) / pl.col("v_all").clip(lower_bound=1)).alias("mid_trade_vol_share"),
        )
        .select(*keys, "mid_trade_share", "mid_trade_vol_share")
    )


def _tick_worker(args: tuple) -> dict | None:
    sym_root, sym_suffix, date_str, mid, minutes, k, beta = args
    m, _jc, _sj, sjv, _sjr, rsk = process_ticker_ticks(mid, minutes, k, beta)
    if m.shape[0] == 0:
        return None
    return {
        "date": date_str,
        "sym_root": sym_root,
        "sym_suffix": sym_suffix,
        "minute": m.astype(np.int32),
        "signed_jump_var": sjv.astype(np.float32),
        "realized_skew_tick": rsk.astype(np.float32),
    }


def _quote_tick_factors(
    nbbo_ticks: pl.DataFrame,
    *,
    max_workers: int = 4,
    pool_kind: str = "thread",
) -> pl.DataFrame:
    empty_schema = {
        "date": pl.Utf8,
        "sym_root": pl.Utf8,
        "sym_suffix": pl.Utf8,
        "minute": pl.Int32,
        "signed_jump_var": pl.Float32,
        "realized_skew_tick": pl.Float32,
    }
    if nbbo_ticks.is_empty():
        return pl.DataFrame(schema=empty_schema)

    date_str = str(nbbo_ticks["date"][0])
    groups = nbbo_ticks.partition_by(["sym_root", "sym_suffix"], as_dict=True)
    tasks: list[tuple] = []
    for (sym_root, sym_suffix), sub in groups.items():
        sub = sub.sort("ts_ns")
        tasks.append(
            (sym_root, sym_suffix, date_str, sub["mid"].to_numpy(), sub["minute"].to_numpy(), LM_K, LM_BETA)
        )

    chunks: list[dict] = []
    if len(tasks) <= 2 or max_workers <= 1:
        for args in tasks:
            res = _tick_worker(args)
            if res is not None:
                chunks.append(res)
    elif pool_kind == "process":
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for res in pool.map(_tick_worker, tasks):
                if res is not None:
                    chunks.append(res)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for res in pool.map(_tick_worker, tasks):
                if res is not None:
                    chunks.append(res)

    if not chunks:
        return pl.DataFrame(schema=empty_schema)
    return pl.concat([pl.DataFrame(c) for c in chunks])


def compute_o_factors(
    date_str: str,
    quote_path: Path,
    trade_path: Path,
    *,
    tick_workers: int = 4,
    tick_pool: str = "thread",
) -> dict[str, pl.DataFrame]:
    """Return dict factor_name -> long frame on full 630-minute grid."""
    trades = load_trades(str(trade_path))
    quotes = load_quotes(str(quote_path))
    universe = ticker_universe(trades, quotes)
    date_fmt = str(trades["date"][0]) if trades.height else date_str.replace("-", "")
    ticks = nbbo_mid_series(quotes)

    enriched = enrich_trades(trades, quotes)
    trade_sparse = _odd_lot_ofi(enriched).join(
        _mid_trade_factors(enriched), on=list(KEYS), how="full", coalesce=True
    )
    quote_sparse = _quote_tick_factors(ticks, max_workers=tick_workers, pool_kind=tick_pool)

    out: dict[str, pl.DataFrame] = {}
    for name in ("odd_lot_ofi", "mid_trade_share", "mid_trade_vol_share"):
        sub = trade_sparse.select(*KEYS, name)
        out[name] = expand_to_grid(sub, universe, date_fmt, [name])
    for name in ("signed_jump_var", "realized_skew_tick"):
        sub = quote_sparse.select(*KEYS, name)
        out[name] = expand_to_grid(sub, universe, date_fmt, [name])
    return out
