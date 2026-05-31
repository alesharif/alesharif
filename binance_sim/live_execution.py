#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live execution gate: real spread + order-book-depth slippage estimate.

This is the piece that belongs in the LIVE bot. Before entering, it reads the
real order book, walks the resting orders for the intended USD size, and
computes the expected fill price and slippage. It then decides whether to
enter based on the user's rules:

  * spread must be small relative to the volatility target, and
  * the book must be deep enough that the entry slips less than a limit.

Market impact was the single biggest cost in the backtest; this gate caps it
using real depth instead of an assumed coefficient. Uses public endpoints
only (no API key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

BASE = "https://data-api.binance.vision"   # public mirror; reachable, no key
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "binance-sim/0.1"})


def _get(path: str, params: dict):
    r = _SESSION.get(BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


@dataclass
class BookQuote:
    symbol: str
    bid: float
    ask: float
    mid: float
    spread_pct: float          # (ask-bid)/mid * 100


def get_spread(symbol: str) -> BookQuote:
    d = _get("/api/v3/ticker/bookTicker", {"symbol": symbol.upper()})
    bid = float(d["bidPrice"]); ask = float(d["askPrice"])
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid * 100.0 if mid > 0 else float("inf")
    return BookQuote(symbol.upper(), bid, ask, mid, spread_pct)


@dataclass
class FillEstimate:
    symbol: str
    side: str                  # "buy" / "sell"
    usd: float
    avg_price: float           # volume-weighted fill price
    slippage_pct: float        # vs mid, signed cost (positive = worse)
    filled: bool               # could the book absorb the whole size?
    levels_used: int
    book_usd_scanned: float     # total USD liquidity seen on that side


def estimate_fill(symbol: str, usd: float, side: str = "buy",
                  limit: int = 1000) -> FillEstimate:
    """Walk the order book for a market order of `usd` and estimate the fill."""
    book = _get("/api/v3/depth", {"symbol": symbol.upper(), "limit": limit})
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 0.0

    levels = asks if side == "buy" else bids   # buy lifts asks, sell hits bids
    remaining = usd
    cost = 0.0      # USD spent/received
    qty = 0.0
    used = 0
    book_usd = 0.0
    for price_s, qty_s in levels:
        price = float(price_s); avail_qty = float(qty_s)
        level_usd = price * avail_qty
        book_usd += level_usd
        if remaining <= 0:
            continue
        take_usd = min(remaining, level_usd)
        take_qty = take_usd / price
        cost += take_qty * price
        qty += take_qty
        remaining -= take_usd
        used += 1

    filled = remaining <= 1e-6
    avg_price = (cost / qty) if qty > 0 else 0.0
    if mid > 0 and avg_price > 0:
        slip = (avg_price - mid) / mid * 100.0
        if side == "sell":
            slip = -slip          # selling below mid is also a positive cost
    else:
        slip = float("inf")
    return FillEstimate(symbol.upper(), side, usd, avg_price, slip,
                        filled, used, book_usd)


@dataclass
class EntryDecision:
    enter: bool
    reason: str
    spread_pct: float
    est_entry_slip_pct: float
    round_trip_cost_pct: float   # spread + entry slip + exit slip estimate


def entry_gate(symbol: str, usd: float, target_pct: float,
               max_spread_ratio: float = 0.15,
               max_entry_slip_pct: float = 0.30) -> EntryDecision:
    """Decide whether to enter, using real spread + book depth.

    target_pct        : expected favourable move (e.g. 100*ATR/price).
    max_spread_ratio  : spread must be <= this * target_pct.
    max_entry_slip_pct: reject if walking the book to fill `usd` slips more.

    We can only control the ENTRY; the eventual exit is a forced sell, so we
    also estimate the exit slip (selling the same size into bids) and fold it
    into the round-trip cost shown for transparency.
    """
    q = get_spread(symbol)
    buy = estimate_fill(symbol, usd, "buy")
    sell = estimate_fill(symbol, usd, "sell")   # what an exit would cost now

    rt_cost = q.spread_pct / 2 + max(buy.slippage_pct, 0) + max(sell.slippage_pct, 0)

    if q.spread_pct > max_spread_ratio * target_pct:
        return EntryDecision(False, f"spread {q.spread_pct:.3f}% > "
                             f"{max_spread_ratio}*target({target_pct:.2f}%)",
                             q.spread_pct, buy.slippage_pct, rt_cost)
    if not buy.filled:
        return EntryDecision(False, f"book too thin to fill ${usd:,.0f} "
                             f"(only ${buy.book_usd_scanned:,.0f} available)",
                             q.spread_pct, buy.slippage_pct, rt_cost)
    if buy.slippage_pct > max_entry_slip_pct:
        return EntryDecision(False, f"entry slip {buy.slippage_pct:.3f}% > "
                             f"{max_entry_slip_pct}% limit",
                             q.spread_pct, buy.slippage_pct, rt_cost)
    return EntryDecision(True, "ok", q.spread_pct, buy.slippage_pct, rt_cost)


if __name__ == "__main__":
    # quick live demo
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    usd = float(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    q = get_spread(sym)
    print(f"{sym}: spread {q.spread_pct:.4f}%  (bid {q.bid} / ask {q.ask})")
    for u in [1_000, 10_000, 50_000, 125_000]:
        f = estimate_fill(sym, u, "buy")
        print(f"  buy ${u:>8,}: slip {f.slippage_pct:>6.3f}%  filled={f.filled}  "
              f"levels={f.levels_used}")
