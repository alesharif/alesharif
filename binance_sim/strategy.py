"""Strategy base class and the signal type strategies emit."""

from __future__ import annotations

import enum
from typing import List

from .client import Candle


class Signal(enum.Enum):
    """A trading decision for the current (most recent closed) candle."""

    BUY = "BUY"    # enter / stay long
    SELL = "SELL"  # exit / stay flat
    HOLD = "HOLD"  # do nothing, keep current position

    def __str__(self) -> str:  # nicer printing
        return self.value


class Strategy:
    """Base class for all strategies.

    Subclasses implement :meth:`generate_signal`, which receives the full
    history of *closed* candles up to and including the current one, and
    returns a single :class:`Signal`. The engine (backtest or paper) owns
    the portfolio and turns signals into simulated trades.
    """

    #: Human-readable identifier used by the CLI registry.
    name: str = "base"

    @property
    def warmup(self) -> int:
        """Number of leading candles to skip before signals are reliable.

        Override when your indicators need a minimum lookback. The engine
        will not act on signals before this many candles are available.
        """
        return 1

    def generate_signal(self, candles: List[Candle]) -> Signal:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:
        """One-line description of the configured strategy (for logs)."""
        return self.name
