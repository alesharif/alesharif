"""Public Binance Spot market-data client.

Only public endpoints are used, so no API key/secret is required. The
client fetches candlestick (kline) data and the latest price, with
automatic pagination so you can request long historical ranges.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import requests

# Public REST base. Override via BinanceClient(base_url=...) if you use a
# regional endpoint or the Spot Testnet (https://testnet.binance.vision).
DEFAULT_BASE_URL = "https://api.binance.com"

# Maximum klines Binance returns in a single request.
MAX_LIMIT = 1000

# Approximate milliseconds per candle for each supported interval. Used to
# paginate historical requests.
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


@dataclass(frozen=True)
class Candle:
    """A single OHLCV candlestick."""

    open_time: int   # ms epoch of candle open
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int  # ms epoch of candle close
    closed: bool = True  # False if this is the still-forming current candle

    @classmethod
    def from_raw(cls, raw: list, closed: bool = True) -> "Candle":
        # Binance kline array layout:
        # [openTime, open, high, low, close, volume, closeTime, ...]
        return cls(
            open_time=int(raw[0]),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5]),
            close_time=int(raw[6]),
            closed=closed,
        )


class BinanceError(RuntimeError):
    """Raised when the Binance API returns an error or is unreachable."""


class BinanceClient:
    """Thin wrapper around Binance public Spot market-data endpoints."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "binance-sim/0.1")

    # -- low level -----------------------------------------------------
    def _get(self, path: str, params: dict) -> object:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:  # network failure
            raise BinanceError(f"Network error calling {url}: {exc}") from exc
        if resp.status_code != 200:
            raise BinanceError(
                f"Binance returned {resp.status_code} for {url}: {resp.text[:200]}"
            )
        return resp.json()

    # -- public API ----------------------------------------------------
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        include_open: bool = False,
    ) -> List[Candle]:
        """Fetch up to ``limit`` candles, paginating transparently.

        By default the still-forming (open) last candle is dropped so that
        strategies only ever see *closed* candles. Set ``include_open=True``
        to keep it (its ``closed`` attribute will be ``False``).
        """
        symbol = symbol.upper()
        if interval not in INTERVAL_MS:
            raise ValueError(
                f"Unsupported interval {interval!r}. "
                f"Choose from: {', '.join(INTERVAL_MS)}"
            )

        candles: List[Candle] = []
        remaining = limit
        cursor = start_time

        while remaining > 0:
            batch_limit = min(remaining, MAX_LIMIT)
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": batch_limit,
            }
            if cursor is not None:
                params["startTime"] = cursor
            if end_time is not None:
                params["endTime"] = end_time

            raw = self._get("/api/v3/klines", params)
            if not raw:
                break

            batch = [Candle.from_raw(r) for r in raw]
            candles.extend(batch)

            if len(batch) < batch_limit:
                break  # no more data available

            remaining -= len(batch)
            # advance cursor to just after the last candle we received
            cursor = batch[-1].close_time + 1
            if end_time is not None and cursor >= end_time:
                break

        # Deduplicate / trim to requested limit (pagination can overshoot).
        if start_time is None and len(candles) > limit:
            candles = candles[-limit:]

        # Decide whether the most recent candle is still forming.
        if candles and not include_open:
            now_ms = int(time.time() * 1000)
            if candles[-1].close_time > now_ms:
                candles = candles[:-1]
        elif candles and include_open:
            now_ms = int(time.time() * 1000)
            if candles[-1].close_time > now_ms:
                last = candles[-1]
                candles[-1] = Candle(
                    last.open_time, last.open, last.high, last.low,
                    last.close, last.volume, last.close_time, closed=False,
                )

        return candles

    def get_price(self, symbol: str) -> float:
        """Latest traded price for a symbol."""
        data = self._get("/api/v3/ticker/price", {"symbol": symbol.upper()})
        return float(data["price"])

    def server_time(self) -> int:
        """Binance server time in ms (also a quick connectivity check)."""
        data = self._get("/api/v3/time", {})
        return int(data["serverTime"])
