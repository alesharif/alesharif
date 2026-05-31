"""Technical indicators implemented in pure Python.

Each function takes a list of floats (typically closing prices) and returns
a list of the same length. Positions that do not yet have enough data to be
computed are filled with ``None`` so indexing always lines up with the
source series.
"""

from __future__ import annotations

from typing import List, Optional

Number = Optional[float]


def sma(values: List[float], period: int) -> List[Number]:
    """Simple Moving Average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: List[Number] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: List[float], period: int) -> List[Number]:
    """Exponential Moving Average (seeded with an SMA of the first window)."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: List[Number] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1.0)
    # seed with SMA over the first `period` values
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * k + prev
        out[i] = prev
    return out


def rsi(values: List[float], period: int = 14) -> List[Number]:
    """Relative Strength Index using Wilder's smoothing."""
    if period <= 0:
        raise ValueError("period must be positive")
    n = len(values)
    out: List[Number] = [None] * n
    if n <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)

    for i in range(period + 1, n):
        change = values[i] - values[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def crossover(a: List[Number], b: List[Number], index: int) -> bool:
    """True if series ``a`` crossed *above* ``b`` at ``index``."""
    if index < 1:
        return False
    a0, a1 = a[index - 1], a[index]
    b0, b1 = b[index - 1], b[index]
    if None in (a0, a1, b0, b1):
        return False
    return a0 <= b0 and a1 > b1


def crossunder(a: List[Number], b: List[Number], index: int) -> bool:
    """True if series ``a`` crossed *below* ``b`` at ``index``."""
    if index < 1:
        return False
    a0, a1 = a[index - 1], a[index]
    b0, b1 = b[index - 1], b[index]
    if None in (a0, a1, b0, b1):
        return False
    return a0 >= b0 and a1 < b1
