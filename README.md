# mojo-pandas-ta

`mojo-pandas-ta` is a standalone Mojo implementation of common technical
analysis indicators from [`pandas-ta`](https://github.com/twopirllc/pandas-ta).
It keeps the upstream function names, argument order, defaults, pandas index,
output names, and Series/DataFrame shapes for the covered subset. The import
name is different so both implementations can be installed and compared:

```python
import mojo_pandas_ta as ta
```

The numerical loops run in a compiled Mojo shared library. The Python layer is
limited to validation, pandas result construction, and the small amount of
composition needed by multi-output indicators.

## Coverage

The current release covers 17 public indicators:

| Category | Indicators |
|---|---|
| Overlap | `sma`, `ema`, `rma`, `wma`, `hma` |
| Momentum | `mom`, `roc`, `rsi`, `macd`, `stoch` |
| Statistics | `variance`, `stdev` |
| Volatility | `true_range`, `atr`, `bbands`, `donchian` |
| Volume | `obv` |

Parity tests compare every function directly with conda-forge's
`pandas-ta==0.4.71b0` using its non-TA-Lib path. Covered moving-average modes
are SMA, EMA, RMA, and WMA where the upstream indicator accepts `mamode`.
`talib` remains in compatible signatures but does not switch implementations.

Not covered are the rest of pandas-ta's indicator catalog, its DataFrame
`.ta` accessor and strategy system, TA-Lib dispatch, and optional
`signal_indicators=True` output columns. Inputs are same-length pandas Series
and are computed as float64. These boundaries are intentional; this port
focuses on common rolling and recursive kernels where native execution does
useful work.

## Install

The pinned Mojo nightly and all Python dependencies are managed by Pixi:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-pandas-ta.so`. Tests and benchmarks are also
Pixi tasks:

```bash
pixi run test
pixi run bench
```

## Usage

```python
import numpy as np
import pandas as pd
import mojo_pandas_ta as ta

rng = np.random.default_rng(7)
close = pd.Series(100 + rng.normal(size=200).cumsum(), name="close")
high = close + 0.5
low = close - 0.5

rsi = ta.rsi(close, length=14)
macd = ta.macd(close)
atr = ta.atr(high, low, close, length=14)

print(rsi.tail())
print(macd.tail())
print(atr.tail())
```

Functions accept upstream options such as `offset`, `fillna`, `length`,
`drift`, `ddof`, `presma`, and composite-specific period arguments.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86_64, Python 3.13.14. Each entry is the best of five warm runs on the
same 2,000,000-row Series. Speedup is pandas-ta time divided by
mojo-pandas-ta time; no compilation time is included in either measurement.

| Indicator | Mojo | pandas-ta | Speedup |
|---|---:|---:|---:|
| SMA(50) | 47.87 ms | 159.72 ms | 3.34x |
| EMA(20) | 45.56 ms | 75.09 ms | 1.65x |
| RSI(14) | 35.27 ms | 253.82 ms | 7.20x |
| MACD(12,26,9) | 87.59 ms | 369.36 ms | 4.22x |
| BBANDS(20) | 46.88 ms | 302.39 ms | 6.45x |
| ATR(14) | 36.94 ms | 536.82 ms | 14.53x |
| STOCH(14,3,3) | 160.85 ms | 858.88 ms | 5.34x |
| DONCHIAN(20,20) | 81.94 ms | 185.85 ms | 2.27x |
| OBV | 17.32 ms | 50.57 ms | 2.92x |

These are single-machine measurements, not universal claims. The benchmark
script prints a fresh Markdown table so results can be reproduced rather than
inferred from this snapshot. Mojo was faster in all nine measured cases.

## How it works

`src/kernels.mojo` is one compilation unit exporting a small C ABI. Python
converts each input to a C-contiguous NumPy float64 buffer, allocates output
buffers initialized to NaN, and passes their addresses across the ABI as
64-bit integers. Mojo reconstructs mutable pointers with
`AnyOrigin[mut=True]` and writes directly into NumPy-owned memory. No Mojo
allocation crosses the FFI boundary, so there is no cross-runtime ownership
or cleanup protocol.

Rolling means and variances use constant-space online updates. Rolling extrema
use a monotonic queue supplied by NumPy scratch memory. EMA, RMA, RSI, MACD,
ATR, and OBV are single-pass recurrences. Multi-output pandas objects are
assembled without copying their NumPy columns and without changing their
original index or column naming convention.

The default SMA Bollinger Bands path fuses the rolling mean, variance, bands,
and temporary calculations into one pass. Its independent bandwidth and
percent finalization uses native-width float64 SIMD with a scalar remainder
and splits sufficiently large inputs across CPU workers.

There is no GPU path; all covered kernels execute on the CPU.

## License

MIT
