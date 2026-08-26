"""Numerical kernels for the covered pandas-ta indicators.

All memory is owned by NumPy. Buffers cross the C ABI as integer addresses so
the exports remain non-parametric under the Mojo 1.0 nightly compiler.
"""

from std.math import isnan, sqrt
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def nan_value() -> Float64:
    var zero = 0.0
    return zero / zero


def ewm(
    src: FPtr,
    dst: FPtr,
    n: Int,
    alpha: Float64,
    presma: Int,
    length: Int,
    adjust: Int,
):
    var start = 0
    var average = nan_value()
    if presma != 0:
        var total = 0.0
        var count = 0
        for i in range(min(length, n)):
            var value = src[i]
            if not isnan(value):
                total += value
                count += 1
        start = length - 1
        if start < n and count > 0:
            average = total / Float64(count)
            dst[start] = average

    var old_weight = 1.0
    var i = start if presma == 0 else start + 1
    while i < n:
        var value = src[i]
        if isnan(average):
            if not isnan(value):
                average = value
                dst[i] = average
                old_weight = 1.0
            i += 1
            continue

        old_weight *= 1.0 - alpha
        if not isnan(value):
            var new_weight = 1.0 if adjust != 0 else alpha
            if average != value:
                average = (
                    old_weight * average + new_weight * value
                ) / (old_weight + new_weight)
            if adjust != 0:
                old_weight += new_weight
            else:
                old_weight = 1.0
        dst[i] = average
        i += 1


def rolling_mean(
    src: FPtr, dst: FPtr, n: Int, length: Int, min_periods: Int
):
    var total = 0.0
    var valid = 0
    for i in range(n):
        var value = src[i]
        if not isnan(value):
            total += value
            valid += 1
        if i >= length:
            var old = src[i - length]
            if not isnan(old):
                total -= old
                valid -= 1
        if valid >= min_periods and valid > 0:
            dst[i] = total / Float64(valid)


def rolling_variance(
    src: FPtr,
    dst: FPtr,
    n: Int,
    length: Int,
    min_periods: Int,
    ddof: Int,
):
    var count = 0
    var mean = 0.0
    var moment2 = 0.0
    for i in range(n):
        if i >= length:
            var leaving = src[i - length]
            if not isnan(leaving):
                if count == 1:
                    count = 0
                    mean = 0.0
                    moment2 = 0.0
                else:
                    var new_count = count - 1
                    var delta = leaving - mean
                    mean -= delta / Float64(new_count)
                    moment2 -= delta * (leaving - mean)
                    count = new_count

        var entering = src[i]
        if not isnan(entering):
            count += 1
            var delta = entering - mean
            mean += delta / Float64(count)
            moment2 += delta * (entering - mean)

        if count >= min_periods and count > ddof:
            var value = moment2 / Float64(count - ddof)
            dst[i] = 0.0 if value < 0.0 else value


def bbands_finish_range(
    mid: FPtr,
    bandwidth: FPtr,
    percent: FPtr,
    start: Int,
    end: Int,
    spread_epsilon: Float64,
    numerator_epsilon: Float64,
):
    comptime W = simd_width_of[DType.float64]()
    var i = start
    var vector_end = start + ((end - start) // W) * W
    while i < vector_end:
        var spread = bandwidth.load[width=W](i) + spread_epsilon
        var numerator = percent.load[width=W](i) + numerator_epsilon
        bandwidth.store(i, 100.0 * spread / mid.load[width=W](i))
        percent.store(i, numerator / spread)
        i += W
    while i < end:
        var spread = bandwidth[i] + spread_epsilon
        bandwidth[i] = 100.0 * spread / mid[i]
        percent[i] = (percent[i] + numerator_epsilon) / spread
        i += 1


def bbands_finish(
    mid: FPtr,
    bandwidth: FPtr,
    percent: FPtr,
    n: Int,
    spread_epsilon: Float64,
    numerator_epsilon: Float64,
):
    bbands_finish_range(
        mid,
        bandwidth,
        percent,
        0,
        n,
        spread_epsilon,
        numerator_epsilon,
    )


def rolling_bbands(
    src: FPtr,
    lower: FPtr,
    mid: FPtr,
    upper: FPtr,
    bandwidth: FPtr,
    percent: FPtr,
    n: Int,
    length: Int,
    ddof: Int,
    lower_std: Float64,
    upper_std: Float64,
):
    var total = 0.0
    var valid = 0
    var count = 0
    var mean = 0.0
    var moment2 = 0.0
    var zero_spread = False
    var zero_numerator = False
    var nan = nan_value()
    for i in range(n):
        if i >= length:
            var leaving = src[i - length]
            if not isnan(leaving):
                if count == 1:
                    count = 0
                    mean = 0.0
                    moment2 = 0.0
                else:
                    var new_count = count - 1
                    var delta = leaving - mean
                    mean -= delta / Float64(new_count)
                    moment2 -= delta * (leaving - mean)
                    count = new_count

        var entering = src[i]
        if not isnan(entering):
            total += entering
            valid += 1
            count += 1
            var delta = entering - mean
            mean += delta / Float64(count)
            moment2 += delta * (entering - mean)

        if i >= length:
            var old = src[i - length]
            if not isnan(old):
                total -= old
                valid -= 1

        if valid >= length and count > ddof:
            var middle = total / Float64(valid)
            var variance = moment2 / Float64(count - ddof)
            var deviation = sqrt(0.0 if variance < 0.0 else variance)
            var lower_value = middle - lower_std * deviation
            var upper_value = middle + upper_std * deviation
            var spread = upper_value - lower_value
            var numerator = entering - lower_value
            mid[i] = middle
            lower[i] = lower_value
            upper[i] = upper_value
            bandwidth[i] = spread
            percent[i] = numerator
            if spread == 0.0:
                zero_spread = True
            if numerator == 0.0:
                zero_numerator = True
        else:
            lower[i] = nan
            mid[i] = nan
            upper[i] = nan
            bandwidth[i] = nan
            percent[i] = nan

    var epsilon = 2.220446049250313e-16
    bbands_finish(
        mid,
        bandwidth,
        percent,
        n,
        epsilon if zero_spread else 0.0,
        epsilon if zero_numerator else 0.0,
    )


def rolling_extreme(
    src: FPtr,
    dst: FPtr,
    queue: IPtr,
    n: Int,
    length: Int,
    min_periods: Int,
    find_max: Int,
):
    var head = 0
    var tail = 0
    var valid = 0
    for i in range(n):
        if i >= length and not isnan(src[i - length]):
            valid -= 1
        while head < tail and Int(queue[head]) <= i - length:
            head += 1

        var value = src[i]
        if not isnan(value):
            valid += 1
            while head < tail:
                var last = src[Int(queue[tail - 1])]
                var discard = last <= value if find_max != 0 else last >= value
                if not discard:
                    break
                tail -= 1
            queue[tail] = Int64(i)
            tail += 1

        if valid >= min_periods and head < tail:
            dst[i] = src[Int(queue[head])]


@export("mpta_sma")
def mpta_sma(
    src_addr: Int, dst_addr: Int, n: Int, length: Int, min_periods: Int
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0 or length <= 0:
        return
    if length > n or min_periods <= 0 or min_periods > length:
        return
    rolling_mean(fp(src_addr), fp(dst_addr), n, length, min_periods)


@export("mpta_ema")
def mpta_ema(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    length: Int,
    presma: Int,
    adjust: Int,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0 or length <= 0:
        return
    if length > n:
        return
    ewm(
        fp(src_addr),
        fp(dst_addr),
        n,
        2.0 / Float64(length + 1),
        presma,
        length,
        adjust,
    )


@export("mpta_rma")
def mpta_rma(
    src_addr: Int, dst_addr: Int, n: Int, length: Int
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0 or length <= 0:
        return
    if length > n:
        return
    ewm(
        fp(src_addr),
        fp(dst_addr),
        n,
        1.0 / Float64(length),
        0,
        length,
        0,
    )


@export("mpta_wma")
def mpta_wma(
    src_addr: Int, dst_addr: Int, n: Int, length: Int, ascending: Int
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0 or length <= 0:
        return
    if length > n:
        return
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var scale = 2.0 / Float64(length * length + length)
    for i in range(length - 1, n):
        var total = 0.0
        var valid = True
        for j in range(length):
            var value = src[i - length + 1 + j]
            if isnan(value):
                valid = False
                break
            var weight = j + 1 if ascending != 0 else length - j
            total += Float64(weight) * value
        if valid:
            dst[i] = total * scale


@export("mpta_variance")
def mpta_variance(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    length: Int,
    min_periods: Int,
    ddof: Int,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0 or length <= 0:
        return
    if length > n or min_periods <= 0 or min_periods > length:
        return
    if ddof < 0 or ddof >= length:
        return
    rolling_variance(
        fp(src_addr), fp(dst_addr), n, length, min_periods, ddof
    )


@export("mpta_bbands")
def mpta_bbands(
    src_addr: Int,
    lower_addr: Int,
    mid_addr: Int,
    upper_addr: Int,
    bandwidth_addr: Int,
    percent_addr: Int,
    n: Int,
    length: Int,
    ddof: Int,
    lower_std: Float64,
    upper_std: Float64,
) abi("C"):
    if src_addr == 0 or lower_addr == 0 or mid_addr == 0:
        return
    if upper_addr == 0 or bandwidth_addr == 0 or percent_addr == 0:
        return
    if n <= 0 or length <= 0 or length > n:
        return
    if ddof < 0 or ddof >= length:
        return
    rolling_bbands(
        fp(src_addr),
        fp(lower_addr),
        fp(mid_addr),
        fp(upper_addr),
        fp(bandwidth_addr),
        fp(percent_addr),
        n,
        length,
        ddof,
        lower_std,
        upper_std,
    )


@export("mpta_extreme")
def mpta_extreme(
    src_addr: Int,
    dst_addr: Int,
    queue_addr: Int,
    n: Int,
    length: Int,
    min_periods: Int,
    find_max: Int,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or queue_addr == 0:
        return
    if n <= 0 or length <= 0 or length > n:
        return
    if min_periods <= 0 or min_periods > length:
        return
    rolling_extreme(
        fp(src_addr),
        fp(dst_addr),
        ip(queue_addr),
        n,
        length,
        min_periods,
        find_max,
    )


@export("mpta_mom_roc")
def mpta_mom_roc(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    length: Int,
    scalar: Float64,
    rate: Int,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0 or length <= 0:
        return
    if length >= n:
        return
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    for i in range(length, n):
        var current = src[i]
        var previous = src[i - length]
        if isnan(current) or isnan(previous):
            continue
        var change = current - previous
        dst[i] = scalar * change / previous if rate != 0 else change


@export("mpta_true_range")
def mpta_true_range(
    high_addr: Int,
    low_addr: Int,
    close_addr: Int,
    dst_addr: Int,
    n: Int,
    drift: Int,
    prenan: Int,
) abi("C"):
    if high_addr == 0 or low_addr == 0 or close_addr == 0 or dst_addr == 0:
        return
    if n <= 0 or drift <= 0:
        return
    var high = fp(high_addr)
    var low = fp(low_addr)
    var close = fp(close_addr)
    var dst = fp(dst_addr)
    for i in range(n):
        if prenan != 0 and i < drift:
            continue
        var value = high[i] - low[i]
        if value < 0.0:
            value = -value
        if i >= drift and not isnan(close[i - drift]):
            var gap_high = high[i] - close[i - drift]
            if gap_high < 0.0:
                gap_high = -gap_high
            var gap_low = close[i - drift] - low[i]
            if gap_low < 0.0:
                gap_low = -gap_low
            value = max(value, max(gap_high, gap_low))
        dst[i] = value


@export("mpta_rsi")
def mpta_rsi(
    src_addr: Int,
    dst_addr: Int,
    n: Int,
    length: Int,
    drift: Int,
    scalar: Float64,
) abi("C"):
    if src_addr == 0 or dst_addr == 0 or n <= 0:
        return
    if length <= 0 or length >= n or drift <= 0:
        return
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var positive_average = nan_value()
    var negative_average = nan_value()
    var alpha = 1.0 / Float64(length)
    for i in range(drift, n):
        var current = src[i]
        var previous = src[i - drift]
        if isnan(current) or isnan(previous):
            continue
        var change = current - previous
        var positive = max(change, 0.0)
        var negative = min(change, 0.0)
        if isnan(positive_average):
            positive_average = positive
            negative_average = negative
        else:
            positive_average = (
                (1.0 - alpha) * positive_average + alpha * positive
            )
            negative_average = (
                (1.0 - alpha) * negative_average + alpha * negative
            )
        dst[i] = (
            scalar
            * positive_average
            / (positive_average + abs(negative_average))
        )


@export("mpta_macd")
def mpta_macd(
    src_addr: Int,
    macd_addr: Int,
    histogram_addr: Int,
    signal_addr: Int,
    work_addr: Int,
    n: Int,
    fast: Int,
    slow: Int,
    signal_length: Int,
    as_mode: Int,
) abi("C"):
    if src_addr == 0 or macd_addr == 0 or histogram_addr == 0:
        return
    if signal_addr == 0 or work_addr == 0 or n <= 0:
        return
    if fast <= 0 or slow <= 0 or signal_length <= 0:
        return
    if fast > n or slow > n or slow + signal_length - 1 > n:
        return
    var src = fp(src_addr)
    var macd = fp(macd_addr)
    var histogram = fp(histogram_addr)
    var signal_line = fp(signal_addr)
    var work = fp(work_addr)
    ewm(src, work, n, 2.0 / Float64(fast + 1), 1, fast, 0)
    ewm(src, macd, n, 2.0 / Float64(slow + 1), 1, slow, 0)
    for i in range(n):
        if not isnan(work[i]) and not isnan(macd[i]):
            macd[i] = work[i] - macd[i]

    var first = slow - 1
    var signal_at = first + signal_length - 1
    if signal_at < n:
        var total = 0.0
        var count = 0
        for i in range(first, signal_at + 1):
            if not isnan(macd[i]):
                total += macd[i]
                count += 1
        if count > 0:
            var average = total / Float64(count)
            signal_line[signal_at] = average
            var alpha = 2.0 / Float64(signal_length + 1)
            for i in range(signal_at + 1, n):
                if not isnan(macd[i]):
                    average = (1.0 - alpha) * average + alpha * macd[i]
                signal_line[i] = average

    if as_mode != 0:
        for i in range(n):
            if not isnan(macd[i]) and not isnan(signal_line[i]):
                macd[i] -= signal_line[i]
            else:
                macd[i] = nan_value()
            signal_line[i] = nan_value()

        first = signal_at
        signal_at = first + signal_length - 1
        if signal_at < n:
            var total = 0.0
            var count = 0
            for i in range(first, signal_at + 1):
                if not isnan(macd[i]):
                    total += macd[i]
                    count += 1
            if count > 0:
                var average = total / Float64(count)
                signal_line[signal_at] = average
                var alpha = 2.0 / Float64(signal_length + 1)
                for i in range(signal_at + 1, n):
                    if not isnan(macd[i]):
                        average = (
                            (1.0 - alpha) * average + alpha * macd[i]
                        )
                    signal_line[i] = average

    for i in range(n):
        if not isnan(macd[i]) and not isnan(signal_line[i]):
            histogram[i] = macd[i] - signal_line[i]


@export("mpta_atr")
def mpta_atr(
    high_addr: Int,
    low_addr: Int,
    close_addr: Int,
    dst_addr: Int,
    work_addr: Int,
    n: Int,
    length: Int,
    drift: Int,
    prenan: Int,
    percent: Int,
) abi("C"):
    if high_addr == 0 or low_addr == 0 or close_addr == 0:
        return
    if dst_addr == 0 or work_addr == 0 or n <= 0:
        return
    if length <= 0 or length > n or drift <= 0:
        return
    var high = fp(high_addr)
    var low = fp(low_addr)
    var close = fp(close_addr)
    var dst = fp(dst_addr)
    var work = fp(work_addr)

    for i in range(n):
        if prenan != 0 and i < drift:
            work[i] = nan_value()
            continue
        var value = abs(high[i] - low[i])
        if i >= drift and not isnan(close[i - drift]):
            value = max(
                value,
                max(
                    abs(high[i] - close[i - drift]),
                    abs(close[i - drift] - low[i]),
                ),
            )
        work[i] = value

    var seed_at = length - 1
    if seed_at < n:
        var total = 0.0
        var count = 0
        for i in range(length):
            if not isnan(work[i]):
                total += work[i]
                count += 1
        if count > 0:
            var average = total / Float64(count)
            dst[seed_at] = average
            var alpha = 1.0 / Float64(length)
            for i in range(seed_at + 1, n):
                if not isnan(work[i]):
                    average = (1.0 - alpha) * average + alpha * work[i]
                dst[i] = average

    if percent != 0:
        for i in range(n):
            if not isnan(dst[i]):
                dst[i] *= 100.0 / close[i]


@export("mpta_obv")
def mpta_obv(
    close_addr: Int, volume_addr: Int, dst_addr: Int, n: Int
) abi("C"):
    if close_addr == 0 or volume_addr == 0 or dst_addr == 0 or n <= 0:
        return
    var close = fp(close_addr)
    var volume = fp(volume_addr)
    var dst = fp(dst_addr)
    var total = 0.0
    for i in range(1, n):
        if isnan(close[i]) or isnan(close[i - 1]) or isnan(volume[i]):
            continue
        if close[i] > close[i - 1]:
            total += volume[i]
        elif close[i] < close[i - 1]:
            total -= volume[i]
        dst[i] = total
