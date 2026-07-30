"""Python-compatible API for the covered pandas-ta indicator subset."""

from __future__ import annotations

import math
import sys
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ._lib import addr, empty, f64, lib


def _length(value, default: int) -> int:
    return int(value) if isinstance(value, Real) and value > 0 else default


def _number(value, default: float) -> float:
    return float(value) if isinstance(value, Real) else float(default)


def _bool(value, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _offset(value) -> int:
    return int(value) if isinstance(value, Integral) and value != 0 else 0


def _drift(value) -> int:
    return int(value) if isinstance(value, Integral) and value != 0 else 1


def _mode(value, default: str) -> str:
    return value.lower() if isinstance(value, str) else default


def _series(value, minimum: int) -> pd.Series | None:
    if isinstance(value, pd.Series) and value.size >= minimum:
        return value
    return None


def _inputs(minimum: int, *values) -> tuple[pd.Series, ...] | None:
    series = tuple(_series(value, minimum) for value in values)
    if any(value is None for value in series):
        return None
    if len({len(value) for value in series}) != 1:
        return None
    return series


def _finish(
    values: np.ndarray,
    index: pd.Index,
    name: str,
    category: str,
    offset,
    kwargs: dict,
) -> pd.Series:
    result = pd.Series(values, index=index, name=name)
    offset = _offset(offset)
    if offset:
        result = result.shift(offset)
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    result.name = name
    result.category = category
    return result


def _sma_values(values: np.ndarray, length: int, min_periods: int | None = None):
    result = empty(values.size)
    periods = length if min_periods is None else int(min_periods)
    lib().mpta_sma(addr(values), addr(result), values.size, length, periods)
    return result


def _ema_values(
    values: np.ndarray, length: int, presma: bool = True, adjust: bool = False
):
    result = empty(values.size)
    lib().mpta_ema(
        addr(values), addr(result), values.size, length, int(presma), int(adjust)
    )
    return result


def _rma_values(values: np.ndarray, length: int):
    result = empty(values.size)
    lib().mpta_rma(addr(values), addr(result), values.size, length)
    return result


def _wma_values(values: np.ndarray, length: int, ascending: bool = True):
    result = empty(values.size)
    lib().mpta_wma(
        addr(values), addr(result), values.size, length, int(ascending)
    )
    return result


def _ma_values(mode: str, values: np.ndarray, length: int, **kwargs):
    if mode == "sma":
        return _sma_values(values, length)
    if mode == "ema":
        return _ema_values(
            values,
            length,
            _bool(kwargs.get("presma"), True),
            _bool(kwargs.get("adjust"), False),
        )
    if mode == "rma":
        return _rma_values(values, length)
    if mode == "wma":
        return _wma_values(values, length)
    return None


def _extreme_values(
    values: np.ndarray, length: int, min_periods: int, find_max: bool
):
    result = empty(values.size)
    queue = np.empty(values.size, dtype=np.int64)
    lib().mpta_extreme(
        addr(values),
        addr(result),
        addr(queue),
        values.size,
        length,
        min_periods,
        int(find_max),
    )
    return result


def _true_range_values(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    drift: int,
    prenan: bool,
):
    high_low = high - low
    if np.equal(high_low, 0).any():
        high_low += sys.float_info.epsilon
    previous = empty(close.size)
    previous[drift:] = close[:-drift]
    result = np.fmax(
        np.abs(high_low),
        np.fmax(np.abs(high - previous), np.abs(previous - low)),
    )
    if prenan:
        result[:drift] = np.nan
    return result


def sma(close, length=None, talib=None, offset=None, **kwargs):
    length = _length(length, 10)
    min_periods = int(kwargs["min_periods"]) if kwargs.get("min_periods") is not None else length
    close = _series(close, max(length, min_periods))
    if close is None:
        return None
    values = f64(close.to_numpy())
    # pandas-ta's Numba SMA path currently ignores min_periods.
    result = _sma_values(values, length)
    return _finish(result, close.index, f"SMA_{length}", "overlap", offset, kwargs)


def ema(close, length=None, talib=None, presma=None, offset=None, **kwargs):
    length = _length(length, 10)
    close = _series(close, length)
    if close is None:
        return None
    values = f64(close.to_numpy())
    result = _ema_values(
        values,
        length,
        _bool(presma, True),
        _bool(kwargs.get("adjust"), False),
    )
    return _finish(result, close.index, f"EMA_{length}", "overlap", offset, kwargs)


def rma(close, length=None, offset=None, **kwargs):
    length = _length(length, 10)
    close = _series(close, length)
    if close is None:
        return None
    values = f64(close.to_numpy())
    result = _rma_values(values, length)
    return _finish(result, close.index, f"RMA_{length}", "overlap", offset, kwargs)


def wma(close, length=None, asc=None, talib=None, offset=None, **kwargs):
    length = _length(length, 10)
    close = _series(close, length)
    if close is None:
        return None
    values = f64(close.to_numpy())
    result = _wma_values(values, length, _bool(asc, True))
    return _finish(result, close.index, f"WMA_{length}", "overlap", offset, kwargs)


def hma(close, length=None, mamode=None, offset=None, **kwargs):
    length = _length(length, 10)
    close = _series(close, length + 2)
    if close is None:
        return None
    mode = _mode(mamode, "wma")
    if mode not in {"ema", "sma", "wma"}:
        return None
    values = f64(close.to_numpy())
    half = _ma_values(mode, values, int(length / 2), **kwargs)
    full = _ma_values(mode, values, length, **kwargs)
    result = _ma_values(mode, 2.0 * half - full, int(math.sqrt(length)), **kwargs)
    suffix = "" if mode == "wma" else mode[0]
    return _finish(
        result, close.index, f"HMA{suffix}_{length}", "overlap", offset, kwargs
    )


def variance(close, length=None, ddof=None, talib=None, offset=None, **kwargs):
    length = _length(length, 30)
    min_periods = int(kwargs["min_periods"]) if kwargs.get("min_periods") is not None else length
    close = _series(close, max(length, min_periods))
    if close is None:
        return None
    ddof = int(ddof) if isinstance(ddof, int) and 0 <= ddof < length else 1
    values = f64(close.to_numpy())
    result = empty(values.size)
    lib().mpta_variance(
        addr(values), addr(result), values.size, length, min_periods, ddof
    )
    return _finish(result, close.index, f"VAR_{length}", "statistics", offset, kwargs)


def stdev(close, length=None, ddof=None, talib=None, offset=None, **kwargs):
    length = _length(length, 30)
    # pandas-ta's stdev does not forward min_periods to its variance helper.
    min_periods = length
    close = _series(close, length)
    if close is None:
        return None
    ddof = int(ddof) if isinstance(ddof, int) and 0 <= ddof < length else 1
    values = f64(close.to_numpy())
    result = empty(values.size)
    lib().mpta_variance(
        addr(values), addr(result), values.size, length, min_periods, ddof
    )
    np.sqrt(result, out=result)
    return _finish(
        result, close.index, f"STDEV_{length}", "statistics", offset, kwargs
    )


def mom(close, length=None, talib=None, offset=None, **kwargs):
    length = _length(length, 10)
    close = _series(close, length + 1)
    if close is None:
        return None
    values = f64(close.to_numpy())
    result = empty(values.size)
    lib().mpta_mom_roc(addr(values), addr(result), values.size, length, 1.0, 0)
    return _finish(result, close.index, f"MOM_{length}", "momentum", offset, kwargs)


def roc(close, length=None, scalar=None, talib=None, offset=None, **kwargs):
    length = _length(length, 10)
    close = _series(close, length + 1)
    if close is None:
        return None
    values = f64(close.to_numpy())
    result = empty(values.size)
    lib().mpta_mom_roc(
        addr(values),
        addr(result),
        values.size,
        length,
        _number(scalar, 100),
        1,
    )
    return _finish(result, close.index, f"ROC_{length}", "momentum", offset, kwargs)


def rsi(
    close,
    length=None,
    scalar=None,
    mamode=None,
    talib=None,
    drift=None,
    offset=None,
    **kwargs,
):
    length = _length(length, 14)
    close = _series(close, length + 1)
    if close is None:
        return None
    scalar = _number(scalar, 100)
    drift = _drift(drift)
    mode = _mode(mamode, "rma")
    values = f64(close.to_numpy())
    result = empty(values.size)
    if mode == "rma" and not np.isnan(values).any():
        lib().mpta_rsi(
            addr(values), addr(result), values.size, length, drift, scalar
        )
    else:
        changes = empty(values.size)
        changes[drift:] = values[drift:] - values[:-drift]
        positive = changes.copy()
        negative = changes.copy()
        positive[positive < 0] = 0
        negative[negative > 0] = 0
        positive_avg = _ma_values(mode, positive, length, **kwargs)
        negative_avg = _ma_values(mode, negative, length, **kwargs)
        if positive_avg is None:
            return None
        result = scalar * positive_avg / (positive_avg + np.abs(negative_avg))
    return _finish(result, close.index, f"RSI_{length}", "momentum", offset, kwargs)


def macd(
    close,
    fast=None,
    slow=None,
    signal=None,
    talib=None,
    offset=None,
    **kwargs,
):
    fast = _length(fast, 12)
    slow = _length(slow, 26)
    signal = _length(signal, 9)
    if slow < fast:
        fast, slow = slow, fast
    close = _series(close, slow + signal - 1)
    if close is None:
        return None
    values = f64(close.to_numpy())
    line, histogram, signal_line, work = (empty(values.size) for _ in range(4))
    as_mode = bool(kwargs.setdefault("asmode", False))
    lib().mpta_macd(
        addr(values),
        addr(line),
        addr(histogram),
        addr(signal_line),
        addr(work),
        values.size,
        fast,
        slow,
        signal,
        int(as_mode),
    )
    suffix = "AS" if as_mode else ""
    props = f"_{fast}_{slow}_{signal}"
    columns = [
        f"MACD{suffix}{props}",
        f"MACD{suffix}h{props}",
        f"MACD{suffix}s{props}",
    ]
    frame = pd.DataFrame(
        dict(zip(columns, (line, histogram, signal_line))),
        index=close.index,
        copy=False,
    )
    offset = _offset(offset)
    if offset:
        frame = frame.shift(offset)
    signal_start = slow - 1 if not as_mode else slow + signal - 2
    frame.iloc[:signal_start, 2] = np.nan
    if "fillna" in kwargs:
        fill = kwargs["fillna"]
        frame.iloc[:, 0] = frame.iloc[:, 0].fillna(fill)
        frame.iloc[:, 1] = frame.iloc[:, 1].fillna(fill)
        frame.iloc[signal_start:, 2] = frame.iloc[signal_start:, 2].fillna(fill)
    frame.name = f"MACD{suffix}{props}"
    frame.category = "momentum"
    return frame


def true_range(
    high,
    low,
    close,
    talib=None,
    prenan=None,
    drift=None,
    offset=None,
    **kwargs,
):
    series = _inputs(1, high, low, close)
    if series is None:
        return None
    high, low, close = series
    h, lo, c = (f64(value.to_numpy()) for value in series)
    result = empty(c.size)
    drift = _drift(drift)
    prenan = _bool(prenan, False)
    if np.isnan(h).any() or np.isnan(lo).any() or np.isnan(c).any():
        result = _true_range_values(h, lo, c, drift, prenan)
    else:
        lib().mpta_true_range(
            addr(h),
            addr(lo),
            addr(c),
            addr(result),
            c.size,
            drift,
            int(prenan),
        )
    if np.isnan(result).all():
        return None
    return _finish(
        result,
        close.index,
        f"TRUERANGE_{drift}",
        "volatility",
        offset,
        kwargs,
    )


def atr(
    high,
    low,
    close,
    length=None,
    mamode=None,
    talib=None,
    prenan=None,
    drift=None,
    offset=None,
    **kwargs,
):
    length = _length(length, 14)
    series = _inputs(length + 1, high, low, close)
    if series is None:
        return None
    high, low, close = series
    mode = _mode(mamode, "rma")
    drift = _drift(drift)
    percent = bool(kwargs.pop("percent", False))
    h, lo, c = (f64(value.to_numpy()) for value in series)
    result, work = empty(c.size), empty(c.size)
    clean = not (np.isnan(h).any() or np.isnan(lo).any() or np.isnan(c).any())
    if mode == "rma" and _bool(kwargs.get("presma"), True) and clean:
        lib().mpta_atr(
            addr(h),
            addr(lo),
            addr(c),
            addr(result),
            addr(work),
            c.size,
            length,
            drift,
            int(_bool(prenan, False)),
            int(percent),
        )
    else:
        if clean:
            lib().mpta_true_range(
                addr(h),
                addr(lo),
                addr(c),
                addr(work),
                c.size,
                drift,
                int(_bool(prenan, False)),
            )
        else:
            work = _true_range_values(
                h, lo, c, drift, _bool(prenan, False)
            )
        presma = _bool(kwargs.pop("presma", True), True)
        if presma:
            seed = np.nanmean(work[:length])
            work[: length - 1] = np.nan
            work[length - 1] = seed
        result = _ma_values(mode, work, length, **kwargs)
        if result is None:
            return None
        if percent:
            result *= 100.0 / c
    if np.isnan(result).all():
        return None
    name = f"ATR{mode[0]}{'p' if percent else ''}_{length}"
    return _finish(result, close.index, name, "volatility", offset, kwargs)


def bbands(
    close,
    length=None,
    lower_std=None,
    upper_std=None,
    ddof=None,
    mamode=None,
    talib=None,
    offset=None,
    **kwargs,
):
    length = _length(length, 5)
    close = _series(close, length)
    if close is None:
        return None
    lower_std = _number(lower_std, 2.0)
    upper_std = _number(upper_std, 2.0)
    if lower_std <= 0:
        lower_std = 2.0
    if upper_std <= 0:
        upper_std = 2.0
    ddof = int(ddof) if isinstance(ddof, int) and 0 <= ddof < length else 1
    mode = _mode(mamode, "sma")
    values = f64(close.to_numpy())
    if mode == "sma":
        lower, mid, upper, bandwidth, percent = (
            np.empty(values.size, dtype=np.float64) for _ in range(5)
        )
        lib().mpta_bbands(
            addr(values),
            addr(lower),
            addr(mid),
            addr(upper),
            addr(bandwidth),
            addr(percent),
            values.size,
            length,
            ddof,
            lower_std,
            upper_std,
        )
    else:
        mid = _ma_values(mode, values, length, **kwargs)
        if mid is None:
            return None
        var = empty(values.size)
        lib().mpta_variance(
            addr(values), addr(var), values.size, length, length, ddof
        )
        deviation = np.sqrt(var)
        lower = mid - lower_std * deviation
        upper = mid + upper_std * deviation
        spread = upper - lower
        if np.equal(spread, 0).any():
            spread += sys.float_info.epsilon
        bandwidth = 100.0 * spread / mid
        numerator = values - lower
        if np.equal(numerator, 0).any():
            numerator += sys.float_info.epsilon
        percent = numerator / spread
    props = f"_{length}_{lower_std}_{upper_std}"
    columns = [f"{prefix}{props}" for prefix in ("BBL", "BBM", "BBU", "BBB", "BBP")]
    frame = pd.DataFrame(
        dict(zip(columns, (lower, mid, upper, bandwidth, percent))),
        index=close.index,
        copy=False,
    )
    offset = _offset(offset)
    if offset:
        frame = frame.shift(offset)
    if "fillna" in kwargs:
        frame.fillna(kwargs["fillna"], inplace=True)
    frame.name = f"BBANDS{props}"
    frame.category = "volatility"
    return frame


def stoch(
    high,
    low,
    close,
    k=None,
    d=None,
    smooth_k=None,
    mamode=None,
    talib=None,
    offset=None,
    **kwargs,
):
    k = _length(k, 14)
    d = _length(d, 3)
    smooth_k = _length(smooth_k, 3)
    series = _inputs(k + d + smooth_k, high, low, close)
    if series is None:
        return None
    high, low, close = series
    mode = _mode(mamode, "sma")
    h, lo, c = (f64(value.to_numpy()) for value in series)
    lowest = _extreme_values(lo, k, k, False)
    highest = _extreme_values(h, k, k, True)
    spread = highest - lowest
    if np.equal(spread, 0).any():
        spread += sys.float_info.epsilon
    raw = 100.0 * (c - lowest) / spread
    first = int(np.flatnonzero(~np.isnan(raw))[0])
    if smooth_k == 1:
        stoch_k = raw
    else:
        tail = _ma_values(mode, raw[first:], smooth_k, **kwargs)
        if tail is None:
            return None
        stoch_k = empty(c.size)
        stoch_k[first:] = tail
    valid = np.flatnonzero(~np.isnan(stoch_k))
    if not valid.size:
        return None
    first_k = int(valid[0])
    tail = _ma_values(mode, stoch_k[first_k:], d, **kwargs)
    if tail is None:
        return None
    stoch_d = empty(c.size)
    stoch_d[first_k:] = tail
    stoch_h = stoch_k - stoch_d
    props = f"_{k}_{d}_{smooth_k}"
    columns = [f"STOCH{part}{props}" for part in ("k", "d", "h")]
    frame = pd.DataFrame(
        dict(zip(columns, (stoch_k, stoch_d, stoch_h))),
        index=close.index,
        copy=False,
    )
    offset = _offset(offset)
    if offset:
        frame = frame.shift(offset)
    k_start = 0 if smooth_k == 1 else first
    d_start = first_k
    frame.iloc[:k_start, 0] = np.nan
    frame.iloc[:d_start, 1] = np.nan
    frame.iloc[:k_start, 2] = np.nan
    if "fillna" in kwargs:
        fill = kwargs["fillna"]
        frame.iloc[k_start:, 0] = frame.iloc[k_start:, 0].fillna(fill)
        frame.iloc[d_start:, 1] = frame.iloc[d_start:, 1].fillna(fill)
        frame.iloc[k_start:, 2] = frame.iloc[k_start:, 2].fillna(fill)
    frame.name = f"STOCH{props}"
    frame.category = "momentum"
    return frame


def donchian(
    high,
    low,
    lower_length=None,
    upper_length=None,
    offset=None,
    **kwargs,
):
    lower_length = _length(lower_length, 20)
    upper_length = _length(upper_length, 20)
    lower_periods = int(kwargs.pop("lmin_periods", lower_length))
    upper_periods = int(kwargs.pop("umin_periods", upper_length))
    series = _inputs(
        max(lower_length, lower_periods, upper_length, upper_periods), high, low
    )
    if series is None:
        return None
    high, low = series
    h, lo = (f64(value.to_numpy()) for value in series)
    lower = _extreme_values(lo, lower_length, lower_periods, False)
    upper = _extreme_values(h, upper_length, upper_periods, True)
    mid = 0.5 * (lower + upper)
    columns = [
        f"DCL_{lower_length}_{upper_length}",
        f"DCM_{lower_length}_{upper_length}",
        f"DCU_{lower_length}_{upper_length}",
    ]
    frame = pd.DataFrame(
        dict(zip(columns, (lower, mid, upper))), index=high.index, copy=False
    )
    if "fillna" in kwargs:
        frame.fillna(kwargs["fillna"], inplace=True)
    offset = _offset(offset)
    if offset:
        frame = frame.shift(offset)
    frame.name = f"DC_{lower_length}_{upper_length}"
    frame.category = "volatility"
    return frame


def obv(close, volume, talib=None, offset=None, **kwargs):
    series = _inputs(1, close, volume)
    if series is None:
        return None
    close, volume = series
    c, v = (f64(value.to_numpy()) for value in series)
    result = empty(c.size)
    lib().mpta_obv(addr(c), addr(v), addr(result), c.size)
    return _finish(result, close.index, "OBV", "volume", offset, kwargs)
