"""RSI mean-reversion strategy.

Buys when RSI climbs back above the oversold threshold (``low``) and sells
when it falls back below the overbought threshold (``high``).
"""

from __future__ import annotations

from typing import List

from ..client import Candle
from ..indicators import rsi
from ..strategy import Signal, Strategy


class RsiStrategy(Strategy):
    """Buy when RSI exits oversold; sell when it exits overbought."""

    name = "rsi"

    def __init__(self, period: int = 14, low: float = 30.0, high: float = 70.0):
        if not (0 < low < high < 100):
            raise ValueError("require 0 < low < high < 100")
        self.period = period
        self.low = low
        self.high = high

    @property
    def warmup(self) -> int:
        return self.period + 2

    def generate_signal(self, candles: List[Candle]) -> Signal:
        closes = [c.close for c in candles]
        values = rsi(closes, self.period)
        cur = values[-1]
        prev = values[-2] if len(values) >= 2 else None
        if cur is None or prev is None:
            return Signal.HOLD

        # cross up through the oversold line -> enter long
        if prev <= self.low < cur:
            return Signal.BUY
        # cross down through the overbought line -> exit
        if prev >= self.high > cur:
            return Signal.SELL
        return Signal.HOLD

    def describe(self) -> str:
        return f"{self.name}(period={self.period}, low={self.low}, high={self.high})"
