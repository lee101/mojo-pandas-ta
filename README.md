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
| SMA(50) | 18.19 ms | 116.24 ms | 6.39x |
| EMA(20) | 22.77 ms | 47.02 ms | 2.07x |
| RSI(14) | 24.59 ms | 145.79 ms | 5.93x |
| MACD(12,26,9) | 46.34 ms | 283.62 ms | 6.12x |
| BBANDS(20) | 63.32 ms | 256.56 ms | 4.05x |
| ATR(14) | 34.34 ms | 526.05 ms | 15.32x |
| STOCH(14,3,3) | 94.59 ms | 504.58 ms | 5.33x |
| DONCHIAN(20,20) | 55.43 ms | 167.14 ms | 3.02x |
| OBV | 16.73 ms | 46.50 ms | 2.78x |

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
percent finalization uses native-width float64 SIMD with a scalar remainder.
STOCH similarly fuses its extrema, smoothing, and histogram work without
temporary NumPy columns. STOCH and Donchian run their independent low/high
extrema concurrently above 262,144 rows, while smaller inputs stay serial.

There is no GPU path. The covered kernels are streaming recurrences, rolling
updates, and extrema with arithmetic intensity below the roughly two-FLOP-per-
byte break-even point; host/device transfers would cost more than GPU execution.

## License

MIT
