"""Measured mojo-pandas-ta versus pandas-ta on identical Series."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import pandas_ta as upstream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_pandas_ta as ta  # noqa: E402


def timeit(fn, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def machine() -> str:
    model = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model}; {platform.system()} {platform.machine()}; Python {platform.python_version()}"


def main() -> None:
    rng = np.random.default_rng(2026)
    n = 2_000_000
    close = pd.Series(100 + np.cumsum(rng.normal(scale=0.5, size=n)))
    high = close + pd.Series(np.abs(rng.normal(scale=0.3, size=n)))
    low = close - pd.Series(np.abs(rng.normal(scale=0.3, size=n)))
    volume = pd.Series(rng.integers(100, 50_000, size=n).astype(float))

    cases = [
        (
            "SMA(50)",
            lambda: ta.sma(close, length=50),
            lambda: upstream.sma(close, length=50, talib=False),
        ),
        (
            "EMA(20)",
            lambda: ta.ema(close, length=20),
            lambda: upstream.ema(close, length=20, talib=False),
        ),
        (
            "RSI(14)",
            lambda: ta.rsi(close),
            lambda: upstream.rsi(close, talib=False),
        ),
        (
            "MACD(12,26,9)",
            lambda: ta.macd(close),
            lambda: upstream.macd(close, talib=False),
        ),
        (
            "BBANDS(20)",
            lambda: ta.bbands(close, length=20),
            lambda: upstream.bbands(close, length=20, talib=False),
        ),
        (
            "ATR(14)",
            lambda: ta.atr(high, low, close),
            lambda: upstream.atr(high, low, close, talib=False),
        ),
        (
            "STOCH(14,3,3)",
            lambda: ta.stoch(high, low, close),
            lambda: upstream.stoch(high, low, close, talib=False),
        ),
        (
            "DONCHIAN(20,20)",
            lambda: ta.donchian(high, low),
            lambda: upstream.donchian(high, low),
        ),
        (
            "OBV",
            lambda: ta.obv(close, volume),
            lambda: upstream.obv(close, volume, talib=False),
        ),
    ]

    print(f"Machine: {machine()}")
    print(f"Input: {n:,} rows; best of 5 warm runs")
    print()
    print("| Indicator | Mojo | pandas-ta | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_fn, upstream_fn in cases:
        mojo_fn()
        upstream_fn()
        mojo_time = timeit(mojo_fn)
        upstream_time = timeit(upstream_fn)
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {upstream_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
