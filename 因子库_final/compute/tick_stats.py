"""Numba-accelerated tick-level statistics."""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


@njit(cache=True)
def lm_jump_per_tick(log_ret: np.ndarray, k: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_jump, signed_jump) with O(n) sliding-window BPV."""
    n = log_ret.shape[0]
    is_jump = np.zeros(n, dtype=np.int8)
    signed = np.zeros(n, dtype=np.float64)
    if n <= k + 1:
        return is_jump, signed

    window_sum = 0.0
    window_cnt = 0
    for j in range(1, k):
        window_sum += abs(log_ret[j]) * abs(log_ret[j - 1])
        window_cnt += 1

    for i in range(k, n):
        if window_cnt > 0:
            bpv = window_sum / window_cnt
            sigma = math.sqrt(bpv) if bpv > 1e-20 else 1e-12
            lt = abs(log_ret[i]) / sigma
            if lt > beta:
                is_jump[i] = 1
                signed[i] = _sign(log_ret[i])

        j_out = i - k + 1
        if j_out >= 1:
            window_sum -= abs(log_ret[j_out]) * abs(log_ret[j_out - 1])
            window_cnt -= 1
        window_sum += abs(log_ret[i]) * abs(log_ret[i - 1])
        window_cnt += 1

    return is_jump, signed


@njit(cache=True)
def _aggregate_minute_stats(
    minutes: np.ndarray,
    log_ret: np.ndarray,
    is_jump: np.ndarray,
    signed_jump: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass minute aggregation on sorted minutes array."""
    n = minutes.shape[0]
    if n == 0:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f

    max_out = n
    out_m = np.empty(max_out, dtype=np.int32)
    out_jc = np.empty(max_out, dtype=np.float64)
    out_sj = np.empty(max_out, dtype=np.float64)
    out_sjv = np.empty(max_out, dtype=np.float64)
    out_sjr = np.empty(max_out, dtype=np.float64)
    out_rsk = np.empty(max_out, dtype=np.float64)

    n_out = 0
    start = 0
    while start < n:
        m = minutes[start]
        end = start + 1
        while end < n and minutes[end] == m:
            end += 1

        jc = 0.0
        sj = 0.0
        rv_pos = 0.0
        rv_neg = 0.0
        m3 = 0.0
        cnt = end - start
        for j in range(start, end):
            r = log_ret[j]
            jc += is_jump[j]
            sj += signed_jump[j]
            if r > 0:
                rv_pos += r * r
            elif r < 0:
                rv_neg += r * r
            m3 += r * r * r

        rv_total = rv_pos + rv_neg
        if rv_total > 1e-20:
            sj_ratio = (rv_pos - rv_neg) / rv_total
            rskew = math.sqrt(cnt) * m3 / (rv_total ** 1.5)
        else:
            sj_ratio = np.nan
            rskew = np.nan

        out_m[n_out] = m
        out_jc[n_out] = jc
        out_sj[n_out] = sj
        out_sjv[n_out] = rv_pos - rv_neg
        out_sjr[n_out] = sj_ratio
        out_rsk[n_out] = rskew
        n_out += 1
        start = end

    return (
        out_m[:n_out],
        out_jc[:n_out],
        out_sj[:n_out],
        out_sjv[:n_out],
        out_sjr[:n_out],
        out_rsk[:n_out],
    )


@njit(cache=True)
def vpin_one_minute(price: np.ndarray, size: np.ndarray, tick_size: float, n_buckets: int = 10) -> float:
    n = price.shape[0]
    if n == 0:
        return np.nan
    total_vol = 0.0
    for i in range(n):
        total_vol += size[i]
    if total_vol <= 0:
        return np.nan
    n_buckets = min(n_buckets, max(1, n // 2))
    bucket_vol = total_vol / n_buckets
    if bucket_vol <= 0:
        return np.nan

    sig = 0.0
    if n > 1:
        acc = 0.0
        cnt = 0
        for i in range(1, n):
            d = price[i] - price[i - 1]
            acc += d * d
            cnt += 1
        if cnt > 0:
            sig = math.sqrt(acc / cnt)
    if sig < 1e-12:
        sig = tick_size

    toxic = 0.0
    filled = 0
    i = 0
    while i < n:
        acc_v = 0.0
        acc_p = 0.0
        j = i
        while j < n and acc_v < bucket_vol - 1e-9:
            acc_v += size[j]
            acc_p = price[j]
            j += 1
        if acc_v <= 0:
            break
        delta_p = acc_p - price[i]
        z = delta_p / sig
        vb = acc_v * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        vs = acc_v - vb
        toxic += abs(vb - vs) / acc_v
        filled += 1
        i = j
    if filled == 0:
        return np.nan
    return toxic / filled


@njit(cache=True)
def vpin_by_minute(
    minutes: np.ndarray,
    price: np.ndarray,
    size: np.ndarray,
    tick_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (minute_ids, vpin_values) for one ticker."""
    if minutes.shape[0] == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
    order = np.argsort(minutes)
    minutes = minutes[order]
    price = price[order]
    size = size[order]

    out_m = np.empty(minutes.shape[0], dtype=np.int32)
    out_v = np.empty(minutes.shape[0], dtype=np.float64)
    n_out = 0
    start = 0
    while start < minutes.shape[0]:
        m = minutes[start]
        end = start + 1
        while end < minutes.shape[0] and minutes[end] == m:
            end += 1
        out_m[n_out] = m
        out_v[n_out] = vpin_one_minute(price[start:end], size[start:end], tick_size)
        n_out += 1
        start = end
    return out_m[:n_out], out_v[:n_out]


def process_ticker_vpin(
    minutes: np.ndarray,
    price: np.ndarray,
    size: np.ndarray,
    tick_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    m_ids, vals = vpin_by_minute(
        minutes.astype(np.int32),
        price.astype(np.float64),
        size.astype(np.float64),
        tick_size,
    )
    return m_ids, vals


def process_ticker_ticks(
    mid: np.ndarray,
    minutes: np.ndarray,
    k: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return minute arrays: m, jump_count, signed_jump, signed_jump_var, sj_ratio, realized_skew."""
    n = mid.shape[0]
    if n < 2:
        empty = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return empty, empty_f, empty_f, empty_f, empty_f, empty_f

    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = 0.0
    for i in range(1, n):
        if mid[i - 1] > 0 and mid[i] > 0:
            log_ret[i] = math.log(mid[i] / mid[i - 1])
        else:
            log_ret[i] = 0.0

    is_jump, signed = lm_jump_per_tick(log_ret, k, beta)

    order = np.argsort(minutes)
    return _aggregate_minute_stats(
        minutes[order].astype(np.int32),
        log_ret[order],
        is_jump[order],
        signed[order],
    )
