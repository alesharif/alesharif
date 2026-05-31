"""Live paper-trading engine.

Polls Binance for fresh candles, feeds *closed* candles to a strategy, and
simulates trades against a local :class:`Portfolio`. No real orders are
ever sent. State is persisted to disk after every new candle so the run can
resume after a restart (important in ephemeral environments).
"""

from __future__ import annotations

import json
import os
import signal as _signal
import time
from typing import Callable, Optional

from .client import BinanceClient, INTERVAL_MS
from .portfolio import Portfolio
from .strategy import Signal, Strategy


class PaperTrader:
    def __init__(
        self,
        client: BinanceClient,
        symbol: str,
        interval: str,
        strategy: Strategy,
        portfolio: Portfolio,
        state_file: Optional[str] = None,
        history: int = 500,
        log: Callable[[str], None] = print,
    ):
        self.client = client
        self.symbol = symbol.upper()
        self.interval = interval
        self.strategy = strategy
        self.portfolio = portfolio
        self.state_file = state_file
        self.history = max(history, strategy.warmup + 2)
        self.log = log

        self._last_candle_time: Optional[int] = None
        self._stop = False

        if state_file and os.path.exists(state_file):
            self._load_state()

    # -- state persistence --------------------------------------------
    def _save_state(self) -> None:
        if not self.state_file:
            return
        data = {
            "symbol": self.symbol,
            "interval": self.interval,
            "strategy": self.strategy.name,
            "last_candle_time": self._last_candle_time,
            "portfolio": self.portfolio.to_dict(),
        }
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self.state_file)

    def _load_state(self) -> None:
        try:
            with open(self.state_file) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"[warn] could not load state ({exc}); starting fresh")
            return
        if data.get("symbol") != self.symbol or data.get("interval") != self.interval:
            self.log("[warn] saved state is for a different market; ignoring it")
            return
        self.portfolio = Portfolio.from_dict(data["portfolio"])
        self._last_candle_time = data.get("last_candle_time")
        self.log(
            f"[state] resumed: cash={self.portfolio.cash:.2f} "
            f"position={'yes' if self.portfolio.in_position else 'no'}"
        )

    # -- main loop -----------------------------------------------------
    def _interval_ms(self) -> int:
        return INTERVAL_MS[self.interval]

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self.log("\n[stop] shutting down, saving state...")
            self._stop = True
        try:
            _signal.signal(_signal.SIGINT, handler)
            _signal.signal(_signal.SIGTERM, handler)
        except ValueError:
            # not in the main thread; skip handler installation
            pass

    def run(self, poll_seconds: Optional[float] = None, max_iterations: Optional[int] = None) -> None:
        """Run the polling loop until interrupted.

        ``poll_seconds`` defaults to 1/10 of the candle interval (min 5s).
        ``max_iterations`` is mainly for testing — stop after N polls.
        """
        self._install_signal_handlers()
        if poll_seconds is None:
            poll_seconds = max(5.0, self._interval_ms() / 1000 / 10)

        self.log(
            f"[start] paper trading {self.symbol} {self.interval} | "
            f"strategy={self.strategy.describe()} | "
            f"cash={self.portfolio.cash:.2f} fee={self.portfolio.fee_rate}"
        )

        iterations = 0
        while not self._stop:
            try:
                self._tick()
            except Exception as exc:  # keep the loop alive on transient errors
                self.log(f"[error] {exc}")

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            if self._stop:
                break
            time.sleep(poll_seconds)

        self._save_state()
        self.log("[done] state saved.")

    def _tick(self) -> None:
        candles = self.client.get_klines(
            self.symbol, self.interval, limit=self.history
        )
        if not candles:
            return
        latest = candles[-1]

        # Only act once per newly-closed candle.
        if self._last_candle_time is not None and latest.close_time <= self._last_candle_time:
            return
        self._last_candle_time = latest.close_time

        if len(candles) < self.strategy.warmup:
            return

        signal = self.strategy.generate_signal(candles)
        price = latest.close
        action = "—"

        if signal is Signal.BUY and not self.portfolio.in_position:
            trade = self.portfolio.buy(price, latest.close_time)
            if trade:
                action = f"BUY  qty={trade.qty:.6f} @ {price:.2f}"
        elif signal is Signal.SELL and self.portfolio.in_position:
            trade = self.portfolio.sell(price, latest.close_time)
            if trade:
                action = f"SELL qty={trade.qty:.6f} @ {price:.2f} pnl={trade.pnl:+.2f}"

        equity = self.portfolio.equity(price)
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(latest.close_time / 1000))
        self.log(
            f"[{ts}] {self.symbol} close={price:.2f} signal={signal} "
            f"{action} | equity={equity:.2f}"
        )
        self._save_state()
