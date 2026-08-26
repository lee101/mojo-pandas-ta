"""Load the compiled Mojo kernels and declare their C signatures."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PANDAS_TA_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pandas-ta.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpta_sma": ([I, I, I, I, I], None),
    "mpta_ema": ([I, I, I, I, I, I], None),
    "mpta_rma": ([I, I, I, I], None),
    "mpta_wma": ([I, I, I, I, I], None),
    "mpta_variance": ([I, I, I, I, I, I], None),
    "mpta_bbands": ([I, I, I, I, I, I, I, I, I, F, F], None),
    "mpta_extreme": ([I, I, I, I, I, I, I], None),
    "mpta_donchian": ([I, I, I, I, I, I, I, I, I, I, I, I], None),
    "mpta_stoch_sma": ([I, I, I, I, I, I, I, I, I, I, I, I], None),
    "mpta_mom_roc": ([I, I, I, I, F, I], None),
    "mpta_true_range": ([I, I, I, I, I, I, I], None),
    "mpta_rsi": ([I, I, I, I, I, F], None),
    "mpta_macd": ([I, I, I, I, I, I, I, I, I, I], None),
    "mpta_atr": ([I, I, I, I, I, I, I, I, I, I], None),
    "mpta_obv": ([I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if os.environ.get("MOJO_PANDAS_TA_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(
        ["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(values) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 1:
        raise ValueError("indicator inputs must be one-dimensional")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("complex indicator inputs cannot be converted to float64")
    try:
        result = np.ascontiguousarray(source, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("indicator inputs must be convertible to float64") from exc
    return result


def empty(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float64)


def addr(values: np.ndarray) -> int:
    if not isinstance(values, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if values.ndim != 1 or values.dtype not in (
        np.dtype(np.float64),
        np.dtype(np.int64),
    ):
        raise TypeError("FFI buffers must be one-dimensional float64 or int64 arrays")
    if not values.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if values.size == 0 or values.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(values.ctypes.data)
