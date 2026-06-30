from typing import Any

import numpy as np
import pandas as pd
import operator

# Panel column names from template_panel_loader (only referenced here).
_SYM = "instrument"
_TIME = "datetime"
_PANEL_CTX: pd.DataFrame | None = None


def set_ts_panel_meta(df: pd.DataFrame | None) -> None:
    """Set row keys for the current factor eval (called from template.jinjia2)."""
    global _PANEL_CTX
    if df is None or not isinstance(df, pd.DataFrame):
        _PANEL_CTX = None
        return
    if _SYM not in df.columns or _TIME not in df.columns:
        _PANEL_CTX = None
        return
    _PANEL_CTX = df[[_SYM, _TIME]].copy()


def clear_ts_panel_meta() -> None:
    global _PANEL_CTX
    _PANEL_CTX = None


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (_SYM, _TIME)]


def _ts_roll(df: pd.DataFrame, fn):
    """Rolling / time-series ops within each symbol group."""
    vc = _feat_cols(df)
    if not vc:
        raise ValueError("time-series op needs at least one feature column")
    if len(vc) == 1:
        return df.groupby(_SYM, sort=False)[vc[0]].transform(fn)
    return df.groupby(_SYM, sort=False)[vc].transform(fn)


def _xs_by_time(df: pd.DataFrame, fn):
    """Cross-sectional ops at each timestamp."""
    vc = _feat_cols(df)
    if not vc:
        return df
    if len(vc) == 1:
        return df.groupby(_TIME, sort=False)[vc[0]].transform(fn)
    return df.groupby(_TIME, sort=False)[vc].transform(fn)


def _xs_rank_pct(df: pd.DataFrame):
    vc = _feat_cols(df)
    if not vc:
        return df
    if len(vc) == 1:
        return df.groupby(_TIME, sort=False)[vc[0]].rank(pct=True)
    return df.groupby(_TIME, sort=False)[vc].rank(pct=True)


def _attach_series_to_panel(s: pd.Series) -> pd.DataFrame | None:
    """Map bare Series back to panel layout for ops that need grouping."""
    if _PANEL_CTX is None or not isinstance(s, pd.Series):
        return None
    if len(s) != len(_PANEL_CTX):
        return None
    if not s.index.equals(_PANEL_CTX.index):
        s = s.reindex(_PANEL_CTX.index)
    out = _PANEL_CTX.copy()
    out["_v"] = np.asarray(s, dtype=float)
    return out


def _as_series(obj: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        vc = _feat_cols(obj)
        if vc:
            return obj[vc[0]]
        return obj.iloc[:, 0]
    return pd.Series(obj)


def _to_numeric_series(obj: pd.DataFrame | pd.Series) -> pd.Series:
    """Feature values only; never ``instrument`` / ``datetime`` strings."""
    return pd.to_numeric(_as_series(obj), errors="coerce")


def _normalize_op_result(result: Any) -> Any:
    """Drop panel meta columns so downstream ``*`` / ``-`` stay numeric."""
    if isinstance(result, pd.DataFrame):
        vc = _feat_cols(result)
        if not vc:
            return _to_numeric_series(result)
        if len(vc) == 1:
            return pd.to_numeric(result[vc[0]], errors="coerce")
        return result[vc].apply(pd.to_numeric, errors="coerce")
    return result


def _xs_transform(df: pd.DataFrame, stat: str) -> pd.Series | pd.DataFrame:
    """Cross-sectional stat at each timestamp, broadcast to every row."""
    vc = _feat_cols(df)
    if not vc:
        raise ValueError(f"cross-sectional {stat} needs a feature column")
    num = df[vc].apply(pd.to_numeric, errors="coerce")
    grouped = num.groupby(df[_TIME], sort=False)
    if len(vc) == 1:
        return grouped[vc[0]].transform(stat)
    return grouped.transform(stat)


def datatype_adapter(func):
    def wrapper(*args):
        args = list(args)
        merged = _attach_series_to_panel(args[0]) if len(args) >= 1 else None
        if merged is not None:
            args[0] = merged
        args = tuple(args)
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            new_args = (pd.DataFrame(args[0]),)
            result = func(*new_args)
            return _normalize_op_result(result)
        if len(args) == 1 and isinstance(args[0], (float, int)):
            new_args = (pd.DataFrame([args[0]]),)
            result = func(*new_args)
            return float(result.iloc[0])
        if (len(args) == 2 and isinstance(args[0], np.ndarray) and not isinstance(args[1], np.ndarray)):
            new_args = (pd.DataFrame(args[0]), args[1])
            result = func(*new_args)
        elif (len(args) == 2 and isinstance(args[1], np.ndarray) and not isinstance(args[0], np.ndarray)):
            new_args = (args[0], pd.DataFrame(args[1]))
            result = func(*new_args)
        else:
            result = func(*args)
        return _normalize_op_result(result)

    return wrapper

@datatype_adapter
def DELTA(df:pd.DataFrame, p:int=1):
    return _ts_roll(df, lambda x: x.diff(periods=p))

@datatype_adapter
def RANK(df:pd.DataFrame):
    """Cross-sectional rank."""
    return _xs_rank_pct(df)

@datatype_adapter
def MEAN(df:pd.DataFrame):
    """Cross-sectional mean."""
    return _xs_transform(df, "mean")

@datatype_adapter
def STD(df:pd.DataFrame):
    """Cross-sectional std."""
    return _xs_transform(df, "std")

@datatype_adapter
def SKEW(df:pd.DataFrame):
    """Cross-sectional skewness."""
    from scipy.stats import skew as scipy_skew
    return _xs_by_time(
        df,
        lambda x: scipy_skew(x.dropna(), nan_policy='omit') if len(x.dropna()) >= 3 else np.nan,
    )

@datatype_adapter
def KURT(df:pd.DataFrame):
    """Cross-sectional kurtosis."""
    from scipy.stats import kurtosis
    return _xs_by_time(
        df,
        lambda x: kurtosis(x.dropna(), fisher=True, nan_policy='omit') if len(x.dropna()) >= 4 else np.nan,
    )

@datatype_adapter
def MAX(df:pd.DataFrame):
    """Cross-sectional max."""
    return _xs_transform(df, "max")

@datatype_adapter
def MIN(df:pd.DataFrame):
    """Cross-sectional min."""
    return _xs_transform(df, "min")

@datatype_adapter
def MEDIAN(df:pd.DataFrame):
    """Cross-sectional median."""
    return _xs_transform(df, "median")


@datatype_adapter
def TS_KURT(df:pd.DataFrame, p:int=5):
    """Rolling kurtosis."""
    from scipy.stats import kurtosis
    def rolling_kurt(x):
        return x.rolling(p, min_periods=min(4, p)).apply(
            lambda arr: kurtosis(arr, fisher=True, nan_policy='omit') if len(arr.dropna()) >= 4 else np.nan,
            raw=False
        )
    return _ts_roll(df, rolling_kurt)

@datatype_adapter
def TS_SKEW(df:pd.DataFrame, p:int=5):
    """Rolling skewness."""
    from scipy.stats import skew as scipy_skew
    def rolling_skew(x):
        return x.rolling(p, min_periods=min(3, p)).apply(
            lambda arr: scipy_skew(arr, nan_policy='omit') if len(arr.dropna()) >= 3 else np.nan,
            raw=False
        )
    return _ts_roll(df, rolling_skew)

@datatype_adapter
def TS_RANK(df:pd.DataFrame, p:int=5):
    """Time-series percentile rank."""
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).rank(pct=True))

@datatype_adapter
def TS_MAX(df:pd.DataFrame, p:int=5):
    """Time-series max."""
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).max())

@datatype_adapter
def TS_MIN(df:pd.DataFrame, p:int=5):
    """Time-series min."""
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).min())

@datatype_adapter
def TS_MEAN(df:pd.DataFrame, p:int=5):
    """Time-series mean."""
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).mean())

@datatype_adapter
def TS_MEDIAN(df:pd.DataFrame, p:int=5):
    """Time-series median."""
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).median())

@datatype_adapter
def PERCENTILE(df: pd.DataFrame, q: float, p: int = None):
    """
    Quantile of given data. q in [0,1]; if p given, rolling quantile.
    """
    assert 0 <= q <= 1, "Quantile q must be in [0, 1]"
    
    if p is not None:
        return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).quantile(q))
    else:
        return _ts_roll(df, lambda x: x.quantile(q))



@datatype_adapter
def TS_SUM(df:pd.DataFrame, p:int=5):
    """Time-series rolling sum."""
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).sum())


@datatype_adapter
def TS_ARGMAX(df: pd.DataFrame, p: int = 5):
    """Days since max in past p days."""
    def rolling_argmax(window):
        return len(window) - window.argmax() - 1
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).apply(rolling_argmax, raw=True))

@datatype_adapter
def TS_ARGMIN(df: pd.DataFrame, p: int = 5):
    """Days since min in past p days."""
    def rolling_argmin(window):
        return len(window) - window.argmin() - 1
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).apply(rolling_argmin, raw=True))



def MAX(x:pd.DataFrame, y:pd.DataFrame, z:pd.DataFrame=None):
    """Element-wise max of DataFrames."""
    if z is None:
        return np.maximum(x, y)
    else:
        return np.maximum(np.maximum(x, y), z)




def MIN(x:pd.DataFrame, y:pd.DataFrame, z:pd.DataFrame=None):
    """Element-wise min of DataFrames.""" 
    if z is None:
        return np.minimum(x, y)
    else:
        return np.minimum(np.minimum(x, y), z)
    


@datatype_adapter
def ABS(df:pd.DataFrame):
    """Element-wise absolute value."""   
    return _ts_roll(df, lambda x: x.abs())


@datatype_adapter
def NEG(df: pd.DataFrame):
    """Element-wise negation (unary minus)."""
    return _ts_roll(df, lambda x: -x)


@datatype_adapter
def DELAY(df:pd.DataFrame, p:int=1):
    """Delay data by p periods."""
    assert p >= 0, ValueError("DELAY period must be >= 0 (look-ahead bias)")
    return _ts_roll(df, lambda x: x.shift(p))


def _ts_pair_roll(df1, df2, p: int, how: str):
    """Rolling corr/cov of two aligned series using panel context."""
    if _PANEL_CTX is None:
        raise ValueError("TS_CORR/TS_COVARIANCE require panel context from factor template")
    a = _as_series(df1).reindex(_PANEL_CTX.index)
    b = _as_series(df2).reindex(_PANEL_CTX.index)
    frame = _PANEL_CTX.copy()
    frame["__a"] = np.asarray(a, dtype=float)
    frame["__b"] = np.asarray(b, dtype=float)

    def _per_sym(g):
        if how == "corr":
            return g["__a"].rolling(p, min_periods=2).corr(g["__b"])
        return g["__a"].rolling(p, min_periods=2).cov(g["__b"])

    return (
        frame.groupby(_SYM, sort=False)[["__a", "__b"]]
        .apply(_per_sym)
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def TS_CORR(df1: pd.Series | pd.DataFrame, df2: np.ndarray | pd.Series, p: int = 5):
    """Rolling correlation of two series."""
    if isinstance(df2, np.ndarray):
        if p != len(df2):
            p = len(df2)
        weights = np.asarray(df2, dtype=float)

        def corr(window):
            y = weights[: len(window)]
            x = window
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            cov = np.sum((x - mean_x) * (y - mean_y))
            std_x = np.sqrt(np.sum((x - mean_x) ** 2))
            std_y = np.sqrt(np.sum((y - mean_y) ** 2))
            if std_x == 0 or std_y == 0:
                return 0.0
            return cov / (std_x * std_y)

        merged = _attach_series_to_panel(_as_series(df1))
        if merged is not None:
            return merged.groupby(_SYM, sort=False)["_v"].transform(
                lambda x: x.rolling(p, min_periods=2).apply(corr, raw=True)
            )
        return _ts_roll(_as_series(df1).to_frame("_v"), lambda x: x.rolling(p, min_periods=2).apply(corr, raw=True))
    if isinstance(df2, (pd.Series, pd.DataFrame)):
        return _ts_pair_roll(df1, df2, p, "corr")
    raise TypeError(f"TS_CORR does not support df2 type: {type(df2)}")


def TS_COVARIANCE(df1: pd.DataFrame, df2: pd.DataFrame, p: int = 5):
    """Rolling covariance of two series."""
    if isinstance(df2, np.ndarray):
        if p != len(df2):
            p = len(df2)
        weights = np.asarray(df2, dtype=float)

        def cov(window):
            y = weights[: len(window)]
            return np.cov(window, y)[0, 1] if len(window) > 1 else 0.0

        merged = _attach_series_to_panel(_as_series(df1))
        if merged is not None:
            return merged.groupby(_SYM, sort=False)["_v"].transform(
                lambda x: x.rolling(p, min_periods=2).apply(cov, raw=True)
            )
        return _ts_roll(_as_series(df1).to_frame("_v"), lambda x: x.rolling(p, min_periods=2).apply(cov, raw=True))
    if isinstance(df2, (pd.Series, pd.DataFrame)):
        return _ts_pair_roll(df1, df2, p, "cov")
    raise TypeError(f"TS_COVARIANCE does not support df2 type: {type(df2)}")

@datatype_adapter
def TS_STD(df:pd.DataFrame, p:int=20):
    """Rolling standard deviation."""
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).std())





@datatype_adapter
def TS_VAR(df: pd.DataFrame, p: int = 5, ddof: int = 1):
    """Rolling variance."""
    return _ts_roll(
        df, lambda x: x.rolling(p, min_periods=1).var(ddof=ddof)
    )

@datatype_adapter
def SIGN(df: pd.DataFrame):
    """Element-wise sign."""
    return np.sign(_to_numeric_series(df))

@datatype_adapter
def SMA(df:pd.DataFrame, m:float=None, n:float=None):
    """Simple moving average. Y_{i+1} = m/n*X_i + (1 - m/n)*Y_i if n given."""
        
    if isinstance(m, int) and m >= 1 and n is None:
        return _ts_roll(df, lambda x: x.rolling(m, min_periods=1).mean())
    else:
        return _ts_roll(df, lambda x: x.ewm(alpha=n/m).mean())

@datatype_adapter
def EMA(df:pd.DataFrame, p):
    """Exponential moving average with period p."""
    return _ts_roll(df, lambda x: x.ewm(span=int(p), min_periods=1).mean())
    
@datatype_adapter
def WMA(df:pd.DataFrame, p:int=20):
    """
    Weighted moving average over p periods (recent has higher weight).
    """
    weights = [0.9**i for i in range(p)][::-1]
    def calculate_wma(window):
        return (window * weights[:len(window)]).sum() / sum(weights[:len(window)])

    return _ts_roll(df, lambda x: x.rolling(window=p, min_periods=1).apply(calculate_wma, raw=True))

@datatype_adapter
def COUNT(cond:pd.DataFrame, p:int=20):
    """
    Conditional count over rolling window p.
    """
    return _ts_roll(cond, lambda x: x.rolling(p, min_periods=1).sum())

@datatype_adapter
def SUMIF(df:pd.DataFrame, p:int, cond:pd.DataFrame):
    """
    Rolling sum of df where cond is true over window p.
    """
    prod = _to_numeric_series(df) * pd.to_numeric(_as_series(cond), errors="coerce").fillna(0)
    if _PANEL_CTX is not None and len(prod) == len(_PANEL_CTX):
        wrapped = _PANEL_CTX.copy()
        wrapped["_v"] = prod.values
        return _ts_roll(wrapped, lambda x: x.rolling(p, min_periods=1).sum())
    return prod.rolling(p, min_periods=1).sum()

@datatype_adapter
def FILTER(df:pd.DataFrame, cond:pd.DataFrame):
    """
    Filter series by condition; where cond is false, set to 0.
    """
    c = pd.to_numeric(_as_series(cond), errors="coerce").fillna(0)
    return _to_numeric_series(df) * (c != 0)
    

@datatype_adapter
def PROD(df:pd.DataFrame, p:int=5):
    """
    Rolling product over window p.
    """

    if isinstance(p, int):
        return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).apply(lambda x: x.prod(), raw=True))
    else:
        return _to_numeric_series(df) * p

@datatype_adapter
def DECAYLINEAR(df:pd.DataFrame, p:int=5):
    """
    Linearly decay weighted average over p periods.
    """
    assert isinstance(p, int), ValueError(f"DECAYLINEAR expects positive int, got {type(p).__name__}")
    decay_weights = np.arange(1, p+1, 1)
    decay_weights = decay_weights / decay_weights.sum()
    
    def calculate_deycaylinear(window):
        return (window * decay_weights[:len(window)]).sum()
    
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).apply(calculate_deycaylinear, raw=True))

@datatype_adapter
def HIGHDAY(df:pd.DataFrame, p:int=5):
    """
    Days since max in window p.
    """
    assert isinstance(p, int), ValueError(f"HIGHDAY expects positive int, got {type(p).__name__}")
    def highday(window):
        return len(window) - window.argmax(axis=0)
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).apply(highday, raw=True))

@datatype_adapter
def LOWDAY(df:pd.DataFrame, p:int=5):
    """
    Days since min in window p.
    """
    assert isinstance(p, int), ValueError(f"LOWDAY expects positive int, got {type(p).__name__}")
    def lowday(window):
        return len(window) - window.argmin(axis=0)
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).apply(lowday, raw=True))
    

@datatype_adapter
def SUMAC(df:pd.DataFrame, p:int=10):
    """
    Rolling cumulative sum over window p.
    """
    assert isinstance(p, int), ValueError(f"SUMAC expects positive int, got {type(p).__name__}")
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).sum())


# Math
@datatype_adapter
def EXP(df:pd.DataFrame):
    """
    Element-wise exp.
    """
    return np.exp(_to_numeric_series(df))

@datatype_adapter
def SQRT(df: pd.DataFrame):
    """Element-wise sqrt."""
    if isinstance(df, int):
        return np.sqrt(df)
    return np.sqrt(_to_numeric_series(df))

@datatype_adapter
def LOG(df:pd.DataFrame):
    """Natural logarithm."""
    if isinstance(df, int):
        return np.log(df)
    return np.log1p(_to_numeric_series(df))

@datatype_adapter
def INV(df: pd.DataFrame):
    """Reciprocal (1/x)."""
    s = _to_numeric_series(df)
    return 1.0 / s.replace(0, np.nan)

@datatype_adapter
def POW(df:pd.DataFrame, n:int):
    """Element-wise power."""
    return np.power(_to_numeric_series(df), n)

@datatype_adapter
def FLOOR(df:pd.DataFrame):
    """Floor (round down)."""
    return np.floor(_to_numeric_series(df))

@datatype_adapter
def TS_ZSCORE(df: pd.DataFrame, p: int = 5):
    assert isinstance(p, int), ValueError(
        f"TS_ZSCORE expects positive int, got {type(p).__name__}"
    )
    mean_roll = _ts_roll(df, lambda x: x.rolling(p, min_periods=1).mean())
    std_roll = _ts_roll(df, lambda x: x.rolling(p, min_periods=1).std())
    x = _to_numeric_series(df)
    denom = std_roll.replace(0, np.nan)
    return (x - mean_roll) / denom

@datatype_adapter
def ZSCORE(df):
    x = _to_numeric_series(df)
    if isinstance(df, pd.DataFrame) and _TIME in df.columns:
        t = df[_TIME]
    elif _PANEL_CTX is not None and len(_PANEL_CTX) == len(x):
        t = _PANEL_CTX[_TIME]
    else:
        t = x.index
    mean = x.groupby(t, sort=False).transform("mean")
    std = x.groupby(t, sort=False).transform("std").replace(0, np.nan)
    return (x - mean) / std

@datatype_adapter
def SCALE(df: pd.DataFrame, target_sum: float = 1.0):
    """Scale series so absolute sum equals target_sum."""
    x = _to_numeric_series(df)
    if isinstance(df, pd.DataFrame) and _TIME in df.columns:
        t = df[_TIME]
    elif _PANEL_CTX is not None and len(_PANEL_CTX) == len(x):
        t = _PANEL_CTX[_TIME]
    else:
        t = x.index
    abs_sum = x.abs().groupby(t, sort=False).transform("sum").replace(0, np.nan)
    return x * (target_sum / abs_sum)


@datatype_adapter
def TS_MAD(df: pd.DataFrame, p: int = 5):
    """Rolling median absolute deviation (MAD = median(|X_i - median(X)|))."""
    def rolling_mad(window):
        median_val = np.median(window)
        abs_dev = np.abs(window - median_val)
        return np.median(abs_dev)

    return _ts_roll(
        df, lambda x: x.rolling(p, min_periods=1).apply(rolling_mad, raw=True)
    )


@datatype_adapter
def TS_QUANTILE(df: pd.DataFrame, p: int = 5, q: float = 0.5):
    """Rolling quantile. Auto-detects parameter order if swapped (q, p -> p, q)."""
    if isinstance(p, float) and 0 < p < 1 and isinstance(q, (int, float)) and q > 1:
        p, q = int(q), p
    p = int(p)
    q = float(q)
    assert 0 <= q <= 1, f"Quantile q must be in [0, 1], got {q}"
    assert p >= 1, f"Window p must >= 1, got {p}"
    return _ts_roll(df, lambda x: x.rolling(p, min_periods=1).quantile(q))

@datatype_adapter
def TS_PCTCHANGE(df: pd.DataFrame, p: int = 1):
    """Percentage change over p periods (default 1)."""
    return _ts_roll(
        df, lambda x: x.pct_change(periods=p, fill_method=None).fillna(0)
    )


def _panel_binary_operand(obj: pd.DataFrame | pd.Series, ref_index: pd.Index | None = None) -> pd.Series:
    """Factor panel ops are row-wise; avoid DataFrame.mul(Series) aligning on columns."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            raise ValueError("empty DataFrame in factor expression")
        s = _to_numeric_series(obj)
    elif isinstance(obj, pd.Series):
        s = pd.to_numeric(obj, errors="coerce")
    else:
        return obj  # type: ignore[return-value]
    if ref_index is not None and not s.index.equals(ref_index):
        s = s.reindex(ref_index)
    return s


def _pandas_binary_arithmetic(df1: pd.DataFrame | pd.Series, df2: pd.DataFrame | pd.Series, op_func) -> pd.DataFrame | pd.Series:
    """
    Use pandas add/sub/mul/div instead of numpy ufuncs.

    ``np.divide(df1, df2)`` (and friends) route through ``__array_ufunc__`` and often
    raise on aligned DataFrame pairs under pandas 2.x; native ops align index/columns correctly.
    """
    name = getattr(op_func, "__name__", "") or ""
    if isinstance(df1, (pd.DataFrame, pd.Series)) and isinstance(df2, (pd.DataFrame, pd.Series)):
        ref = df1.index if len(df1.index) >= len(df2.index) else df2.index
        s1 = _panel_binary_operand(df1, ref)
        s2 = _panel_binary_operand(df2, ref)
        if name in ("divide", "true_divide"):
            return s1 / s2
        if name == "multiply":
            return s1 * s2
        if name == "add":
            return s1 + s2
        if name == "subtract":
            return s1 - s2

    if name in ("divide", "true_divide"):
        return df1.div(df2)
    if name == "multiply":
        return df1.mul(df2)
    if name == "add":
        return df1.add(df2)
    if name == "subtract":
        return df1.sub(df2)
    return op_func(df1, df2)


def ADD(df1, df2):
    """Add with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.add)

def SUBTRACT(df1, df2):
    """Subtract with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.subtract)

def MULTIPLY(df1, df2):
    """Multiply with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.multiply)

def DIVIDE(df1, df2):
    """Divide with index alignment."""
    return _arithmetic_with_alignment(df1, df2, np.divide)

def _arithmetic_with_alignment(df1, df2, op_func):
    """Arithmetic op with index alignment."""
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        df2 = df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        df1 = df1_aligned
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index)
            else:
                df2 = df2.reindex(df1.index)
        except Exception:
            pass
    
    try:
        result = _pandas_binary_arithmetic(df1, df2, op_func)
    except (ValueError, TypeError, MemoryError) as e:
        err = str(e).lower()
        if (
            "identically-labeled" in err
            or "can only compare" in err
            or "index" in err
            or "allocate" in err
            or "memory" in err
        ):
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index, fill_value=0)
            elif isinstance(df1.index, pd.MultiIndex):
                datetime_level = df1.index.get_level_values("datetime")
                df2 = df2.reindex(datetime_level, method="ffill")
                df2.index = df1.index
            result = _pandas_binary_arithmetic(df1, df2, op_func)
        else:
            raise
    
    return result
    
def AND(df1, df2):
    """Logical AND with index alignment."""
    df1_aligned, df2_aligned = _align_for_operation(df1, df2)
    return np.bitwise_and(df1_aligned.astype(np.bool_), df2_aligned.astype(np.bool_))

def OR(df1, df2):
    """Logical OR with index alignment."""
    df1_aligned, df2_aligned = _align_for_operation(df1, df2)
    return np.bitwise_or(df1_aligned.astype(np.bool_), df2_aligned.astype(np.bool_))

def WHERE(condition, true_value, false_value):
    """Conditional expression (WHERE) with index alignment."""
    
    if isinstance(condition, (pd.DataFrame, pd.Series)):
        target_index = condition.index
    elif isinstance(true_value, (pd.DataFrame, pd.Series)):
        target_index = true_value.index
    elif isinstance(false_value, (pd.DataFrame, pd.Series)):
        target_index = false_value.index
    else:
        return np.where(condition, true_value, false_value)
    
    if isinstance(true_value, (pd.DataFrame, pd.Series)) and not true_value.index.equals(target_index):
        if isinstance(target_index, pd.MultiIndex) and not isinstance(true_value.index, pd.MultiIndex):
            datetime_level = target_index.get_level_values('datetime')
            true_value = true_value.reindex(datetime_level, method='ffill')
            true_value.index = target_index
        else:
            true_value = true_value.reindex(target_index, fill_value=0)
    
    if isinstance(false_value, (pd.DataFrame, pd.Series)) and not false_value.index.equals(target_index):
        if isinstance(target_index, pd.MultiIndex) and not isinstance(false_value.index, pd.MultiIndex):
            datetime_level = target_index.get_level_values('datetime')
            false_value = false_value.reindex(datetime_level, method='ffill')
            false_value.index = target_index
        else:
            false_value = false_value.reindex(target_index, fill_value=0)
    
    if isinstance(condition, (pd.DataFrame, pd.Series)) and not condition.index.equals(target_index):
        condition = condition.reindex(target_index, fill_value=False)
    
    result = np.where(condition, true_value, false_value)
    
    if isinstance(result, np.ndarray) and isinstance(target_index, pd.MultiIndex):
        result = pd.Series(result, index=target_index)
    elif isinstance(result, np.ndarray) and isinstance(target_index, pd.Index):
        result = pd.Series(result, index=target_index)
    
    return result

def _align_for_operation(df1, df2):
    """Align two DataFrame/Series indices for binary ops."""
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return df1, df2
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return df1, df2
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return df1, df2
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        return df1, df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        return df1_aligned, df2
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2_aligned = df2.reindex(df1.index)
                return df1, df2_aligned
            else:
                df2_aligned = df2.reindex(df1.index)
                return df1, df2_aligned
        except Exception:
            return df1, df2
    
    return df1, df2

def GT(df1, df2):
    """Greater than with index alignment."""
    return _compare_with_alignment(df1, df2, operator.gt)

def LT(df1, df2):
    """Less than with index alignment."""
    return _compare_with_alignment(df1, df2, operator.lt)

def GE(df1, df2):
    """Greater or equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.ge)

def LE(df1, df2):
    """Less or equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.le)

def EQ(df1, df2):
    """Equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.eq)

def NE(df1, df2):
    """Not equal with index alignment."""
    return _compare_with_alignment(df1, df2, operator.ne)

def _compare_with_alignment(df1, df2, op_func):
    """Compare two DataFrame/Series with index alignment."""
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)) and not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        return op_func(df2, df1) if op_func in [operator.lt, operator.le] else op_func(df1, df2)
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        return op_func(df1, df2)
    
    if isinstance(df1.index, pd.MultiIndex) and not isinstance(df2.index, pd.MultiIndex):
        datetime_level = df1.index.get_level_values('datetime')
        if isinstance(df2, pd.DataFrame):
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        else:
            df2_aligned = df2.reindex(datetime_level, method='ffill')
        df2_aligned.index = df1.index
        df2 = df2_aligned
    elif not isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        datetime_level = df2.index.get_level_values('datetime')
        if isinstance(df1, pd.DataFrame):
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        else:
            df1_aligned = df1.reindex(datetime_level, method='ffill')
        df1_aligned.index = df2.index
        df1 = df1_aligned
    elif not df1.index.equals(df2.index):
        try:
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index)
            else:
                df2 = df2.reindex(df1.index)
        except Exception:
            pass
    
    try:
        result = op_func(df1, df2)
    except (ValueError, TypeError) as e:
        if 'identically-labeled' in str(e) or 'Can only compare' in str(e):
            if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reindex(df1.index, fill_value=0)
            elif isinstance(df1.index, pd.MultiIndex):
                datetime_level = df1.index.get_level_values('datetime')
                df2 = df2.reindex(datetime_level, method='ffill')
                df2.index = df1.index
            result = op_func(df1, df2)
        else:
            raise
    
    return result
