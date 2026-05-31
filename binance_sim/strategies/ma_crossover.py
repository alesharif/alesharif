"""Moving-average crossover strategy.

Goes long when the fast moving average crosses above the slow one, and
exits when it crosses back below. A classic trend-following baseline.
"""

from __future__ import annotations

from typing import List

from ..client import Candle
from ..indicators import crossover, crossunder, ema, sma
from ..strategy import Signal, Strategy


class MaCrossover(Strategy):
    """Go long when the fast MA crosses above the slow MA; exit on cross down."""

    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50, ma_type: str = "ema"):
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type.lower()
        if self.ma_type not in ("ema", "sma"):
            raise ValueError("ma_type must be 'ema' or 'sma'")

    @property
    def warmup(self) -> int:
        return self.slow + 1

    def _ma(self, values: List[float], period: int):
        return ema(values, period) if self.ma_type == "ema" else sma(values, period)

    def generate_signal(self, candles: List[Candle]) -> Signal:
        closes = [c.close for c in candles]
        fast_line = self._ma(closes, self.fast)
        slow_line = self._ma(closes, self.slow)
        i = len(closes) - 1
        if crossover(fast_line, slow_line, i):
            return Signal.BUY
        if crossunder(fast_line, slow_line, i):
            return Signal.SELL
        return Signal.HOLD

    def describe(self) -> str:
        return f"{self.name}(fast={self.fast}, slow={self.slow}, type={self.ma_type})"
