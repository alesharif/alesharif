"""Historical backtesting engine and performance metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .client import Candle
from .portfolio import Portfolio, Trade
from .strategy import Signal, Strategy


@dataclass
class BacktestResult:
    symbol: str
    interval: str
    start_time: int
    end_time: int
    initial_cash: float
    final_equity: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    buy_hold_return: float = 0.0

    # -- derived metrics ----------------------------------------------
    @property
    def total_return(self) -> float:
        if self.initial_cash == 0:
            return 0.0
        return (self.final_equity - self.initial_cash) / self.initial_cash

    @property
    def num_trades(self) -> int:
        # one round trip = a BUY followed by a SELL
        return sum(1 for t in self.trades if t.side == "SELL")

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.side == "SELL" and t.pnl is not None]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed)

    @property
    def max_drawdown(self) -> float:
        peak = -math.inf
        max_dd = 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            if peak > 0:
                dd = (peak - v) / peak
                max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe(self) -> float:
        """Approximate (non-annualised) Sharpe ratio of per-candle returns."""
        rets = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            if prev > 0:
                rets.append((self.equity_curve[i] - prev) / prev)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return mean / std * math.sqrt(len(rets))

    def summary(self) -> str:
        lines = [
            f"Symbol           : {self.symbol} ({self.interval})",
            f"Candles tested   : {len(self.equity_curve)}",
            f"Initial cash     : {self.initial_cash:,.2f}",
            f"Final equity     : {self.final_equity:,.2f}",
            f"Total return     : {self.total_return * 100:+.2f}%",
            f"Buy & Hold       : {self.buy_hold_return * 100:+.2f}%",
            f"Round-trip trades: {self.num_trades}",
            f"Win rate         : {self.win_rate * 100:.1f}%",
            f"Max drawdown     : {self.max_drawdown * 100:.2f}%",
            f"Sharpe (approx)  : {self.sharpe:.2f}",
        ]
        return "\n".join(lines)


def run_backtest(
    candles: List[Candle],
    strategy: Strategy,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    position_fraction: float = 1.0,
    symbol: str = "?",
    interval: str = "?",
) -> BacktestResult:
    """Run ``strategy`` over ``candles`` and return performance results.

    Signals are evaluated on each closed candle and executed at that
    candle's close price. Long-only: a BUY with no position opens one, a
    SELL with a position closes it; HOLD does nothing.
    """
    if not candles:
        raise ValueError("No candles provided to backtest")

    pf = Portfolio(initial_cash, fee_rate=fee_rate, position_fraction=position_fraction)
    warmup = max(1, strategy.warmup)
    equity_curve: List[float] = []

    for i in range(len(candles)):
        price = candles[i].close
        if i + 1 >= warmup:
            history = candles[: i + 1]
            signal = strategy.generate_signal(history)
            if signal is Signal.BUY and not pf.in_position:
                pf.buy(price, candles[i].close_time)
            elif signal is Signal.SELL and pf.in_position:
                pf.sell(price, candles[i].close_time)
        equity_curve.append(pf.equity(price))

    # Close any open position at the final price so results are comparable.
    if pf.in_position:
        pf.sell(candles[-1].close, candles[-1].close_time)
        equity_curve[-1] = pf.cash

    first_price = candles[0].close
    last_price = candles[-1].close
    buy_hold = (last_price - first_price) / first_price if first_price else 0.0

    return BacktestResult(
        symbol=symbol,
        interval=interval,
        start_time=candles[0].open_time,
        end_time=candles[-1].close_time,
        initial_cash=initial_cash,
        final_equity=pf.equity(last_price),
        equity_curve=equity_curve,
        trades=pf.trades,
        buy_hold_return=buy_hold,
    )
