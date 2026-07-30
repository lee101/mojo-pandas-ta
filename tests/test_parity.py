"""Numerical and behavioral parity with pandas-ta 0.4.71b0."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as upstream
import pytest

import mojo_pandas_ta as ta
from mojo_pandas_ta._lib import f64, lib


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(2026)
    n = 600
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=index, name="close")
    high = close + pd.Series(np.abs(rng.normal(size=n)), index=index)
    low = close - pd.Series(np.abs(rng.normal(size=n)), index=index)
    volume = pd.Series(rng.integers(100, 50_000, size=n).astype(float), index=index)
    return high, low, close, volume


def assert_same(actual, expected, *, rtol=1e-10, atol=1e-10):
    assert type(actual) is type(expected)
    assert actual.index.equals(expected.index)
    if isinstance(actual, pd.DataFrame):
        assert actual.columns.tolist() == expected.columns.tolist()
    else:
        assert actual.name == expected.name
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol, equal_nan=True
    )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("sma", {}),
        ("sma", {"length": 31}),
        ("ema", {}),
        ("ema", {"length": 17, "presma": False, "adjust": True}),
        ("rma", {"length": 21}),
        ("wma", {}),
        ("wma", {"length": 13, "asc": False}),
        ("hma", {}),
        ("hma", {"length": 21, "mamode": "ema"}),
        ("hma", {"length": 21, "mamode": "sma"}),
    ],
)
def test_overlap_parity(market, name, kwargs):
    close = market[2]
    assert_same(
        getattr(ta, name)(close, **kwargs),
        getattr(upstream, name)(close, talib=False, **kwargs),
    )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("variance", {}),
        ("variance", {"length": 17, "ddof": 0, "min_periods": 8}),
        ("stdev", {}),
        ("stdev", {"length": 17, "ddof": 0}),
    ],
)
def test_statistics_parity(market, name, kwargs):
    close = market[2]
    assert_same(
        getattr(ta, name)(close, **kwargs),
        getattr(upstream, name)(close, talib=False, **kwargs),
    )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("mom", {}),
        ("mom", {"length": 23}),
        ("roc", {}),
        ("roc", {"length": 23, "scalar": 1}),
        ("rsi", {}),
        ("rsi", {"length": 9, "drift": 2, "scalar": 1}),
        ("rsi", {"length": 9, "drift": 2, "scalar": 1, "mamode": "ema"}),
        ("rsi", {"length": 9, "drift": 2, "scalar": 1, "mamode": "sma"}),
    ],
)
def test_scalar_momentum_parity(market, name, kwargs):
    close = market[2]
    assert_same(
        getattr(ta, name)(close, **kwargs),
        getattr(upstream, name)(close, talib=False, **kwargs),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"fast": 7, "slow": 18, "signal": 5},
        {"fast": 7, "slow": 18, "signal": 5, "asmode": True},
    ],
)
def test_macd_parity(market, kwargs):
    close = market[2]
    assert_same(ta.macd(close, **kwargs), upstream.macd(close, talib=False, **kwargs))


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    [
        ("true_range", (0, 1, 2), {}),
        ("true_range", (0, 1, 2), {"drift": 2, "prenan": True}),
        ("atr", (0, 1, 2), {}),
        ("atr", (0, 1, 2), {"length": 9, "percent": True}),
        ("atr", (0, 1, 2), {"length": 9, "mamode": "ema", "drift": 2}),
        ("bbands", (2,), {}),
        (
            "bbands",
            (2,),
            {
                "length": 11,
                "lower_std": 1.5,
                "upper_std": 2.5,
                "ddof": 0,
                "mamode": "ema",
            },
        ),
        ("donchian", (0, 1), {}),
        (
            "donchian",
            (0, 1),
            {
                "lower_length": 11,
                "upper_length": 17,
                "lmin_periods": 5,
                "umin_periods": 7,
            },
        ),
    ],
)
def test_volatility_parity(market, name, args, kwargs):
    values = tuple(market[i] for i in args)
    assert_same(
        getattr(ta, name)(*values, **kwargs),
        getattr(upstream, name)(*values, talib=False, **kwargs)
        if name != "donchian"
        else upstream.donchian(*values, **kwargs),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"k": 9, "d": 4, "smooth_k": 2},
        {"k": 9, "d": 4, "smooth_k": 2, "mamode": "ema"},
        {"k": 9, "d": 4, "smooth_k": 1},
    ],
)
def test_stoch_parity(market, kwargs):
    high, low, close, _ = market
    assert_same(
        ta.stoch(high, low, close, **kwargs),
        upstream.stoch(high, low, close, talib=False, **kwargs),
    )


def test_obv_parity(market):
    _, _, close, volume = market
    assert_same(
        ta.obv(close, volume),
        upstream.obv(close, volume, talib=False),
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("ema", (2,)),
        ("rma", (2,)),
        ("sma", (2,)),
        ("wma", (2,)),
        ("variance", (2,)),
        ("rsi", (2,)),
        ("bbands", (2,)),
        ("true_range", (0, 1, 2)),
        ("atr", (0, 1, 2)),
    ],
)
def test_nan_gap_parity(market, name, args):
    copied = [value.copy() for value in market]
    for value in copied[:3]:
        value.iloc[[0, 4, 10, 40, 41, 100]] = np.nan
    values = tuple(copied[i] for i in args)
    actual = getattr(ta, name)(*values)
    expected = getattr(upstream, name)(*values, talib=False)
    assert_same(actual, expected)


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    [
        ("sma", (2,), {"length": 17, "offset": 2, "fillna": -1}),
        ("stdev", (2,), {"length": 17, "offset": 2, "fillna": -1}),
        (
            "macd",
            (2,),
            {
                "fast": 7,
                "slow": 18,
                "signal": 5,
                "asmode": True,
                "offset": 1,
                "fillna": 0,
            },
        ),
        (
            "stoch",
            (0, 1, 2),
            {
                "k": 9,
                "d": 4,
                "smooth_k": 2,
                "mamode": "ema",
                "offset": 1,
                "fillna": 0,
            },
        ),
    ],
)
def test_offset_and_fill_parity(market, name, args, kwargs):
    values = tuple(market[i] for i in args)
    assert_same(
        getattr(ta, name)(*values, **kwargs),
        getattr(upstream, name)(*values, talib=False, **kwargs),
    )


def test_names_categories_and_index(market):
    high, low, close, _ = market
    scalar = ta.rsi(close)
    frame = ta.bbands(close)
    assert scalar.name == "RSI_14"
    assert scalar.category == "momentum"
    assert frame.name == "BBANDS_5_2.0_2.0"
    assert frame.category == "volatility"
    assert scalar.index.equals(close.index)
    assert frame.index.equals(close.index)
    assert ta.true_range(high, low, close).name == "TRUERANGE_1"


@pytest.mark.parametrize("n", [262_139, 262_147])
def test_bbands_parallel_threshold_and_simd_tail(n):
    rng = np.random.default_rng(n)
    close = pd.Series(100 + rng.normal(size=n))
    assert_same(
        ta.bbands(close, length=20),
        upstream.bbands(close, length=20, talib=False),
    )


def test_bbands_zero_spread_and_numerator_parity():
    close = pd.Series(np.full(37, 10.0))
    kwargs = {"length": 5, "lower_std": 1.5, "upper_std": 2.5, "ddof": 0}
    assert_same(
        ta.bbands(close, **kwargs),
        upstream.bbands(close, talib=False, **kwargs),
    )


def test_short_or_non_series_inputs_return_none():
    assert ta.sma(pd.Series([1.0, 2.0]), length=10) is None
    assert ta.rsi(np.arange(100.0)) is None
    assert ta.atr(
        pd.Series(np.arange(20.0)),
        pd.Series(np.arange(19.0)),
        pd.Series(np.arange(20.0)),
    ) is None


def test_strided_float_input_is_copied_safely():
    source = np.arange(120.0)[::2]
    assert not source.flags.c_contiguous
    converted = f64(source)
    assert converted.flags.c_contiguous
    close = pd.Series(source)
    assert_same(
        ta.sma(close, length=7),
        upstream.sma(close, length=7, talib=False),
    )


def test_complex_input_is_not_silently_narrowed():
    with pytest.raises(TypeError, match="complex"):
        f64(np.array([1 + 2j]))


def test_native_exports_reject_null_buffers_without_dereferencing():
    # The shared library is internal, but its C exports still fail closed before
    # constructing Mojo's non-nullable pointers.
    lib().mpta_sma(0, 0, 1, 1, 1)
    lib().mpta_obv(0, 0, 0, 1)
