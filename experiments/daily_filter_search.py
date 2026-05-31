#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search the best daily-timeframe trend filter on 20 mid/small-volume coins.

Tests every candidate in daily_filters.CANDIDATES (price>EMA20/50/100/200, EMA
alignment pairs, full stack, MACD slope-up, Stochastic slope-up) as an entry
gate, on a small mid/small-cap sample first (fast). Pick the best, then re-run
on the full universe separately.

Base stack: SL 2.0 / TRAIL 0.2 + signal-age<=2, mild costs.

Run:  python experiments/daily_filter_search.py 2022-01-01 2026-01-01
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import daily_filters as DF               # noqa: E402
from binance_sim.client import BinanceClient              # noqa: E402


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def pick_midcaps(n=20, lo=3e6, hi=40e6):
    """20 USDT coins with current 24h quote volume in a mid/small band."""
    import json
    import urllib.request
    url = "https://data-api.binance.vision/api/v3/ticker/24hr"
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "binance-sim/0.1"}), timeout=30) as r:
        data = json.load(r)
    rows = []
    for t in data:
        s = t.get("symbol", "")
        if not s.endswith("USDT"):
            continue
        if S.should_exclude(s[:-4]):
            continue
        qv = float(t.get("quoteVolume", 0))
        if lo <= qv <= hi:
            rows.append((s, qv))
    rows.sort(key=lambda x: x[1])
    if len(rows) <= n:
        return [s for s, _ in rows]
    step = len(rows) / n           # spread evenly across the band
    return [rows[int(i * step)][0] for i in range(n)]


def _metrics(rep):
    c = rep.trades
    wins = [t.net_pct for t in c if t.net_pct > 0]
    losses = [t.net_pct for t in c if t.net_pct <= 0]
    gw, gl = sum(wins), -sum(losses)
    pf = (gw / gl) if gl > 0 else float("inf")
    wr = (len(wins) / len(c) * 100) if c else 0.0
    al = (sum(losses) / len(losses)) if losses else 0.0
    worst = min((t.net_pct for t in c), default=0.0)
    return rep.total_return * 100, len(c), wr, al, worst, pf, rep.max_drawdown * 100


def main():
    start = parse(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
    end = parse(sys.argv[2] if len(sys.argv) > 2 else "2026-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS

    syms = pick_midcaps(20)
    print(f"20 mid/small-cap coins: {', '.join(syms)}\n")
    raw = PIT.prefetch_universe(syms, fetch_from, end, log=lambda *a: None)
    per_symbol = {}
    for s, df in raw.items():
        d = S.attach_entry_signal(S.compute_features(df.copy()))
        d = DF.add_daily_filters(d)
        per_symbol[s] = d.set_index("close_time")
    print(f"Usable: {len(per_symbol)} coins.\n")

    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 2.0, 0.2, 0.2
    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05,
                max_signal_age=2)

    print(f"{'daily filter':<16}{'return%':>12}{'trades':>8}{'win%':>7}"
          f"{'avgL':>7}{'worst':>8}{'PF':>6}{'maxDD':>7}  desc")
    print("-" * 92)
    order = [""] + list(DF.CANDIDATES.keys())
    for col in order:
        rep = _run(per_symbol, start, end, dict(base, daily_filter_col=col))
        ret, n, wr, al, worst, pf, dd = _metrics(rep)
        label = "(none/base)" if col == "" else col
        desc = "" if col == "" else DF.CANDIDATES[col]
        print(f"{label:<16}{ret:>11,.0f}%{n:>8}{wr:>6.1f}%{al:>6.2f}%"
              f"{worst:>7.1f}%{pf:>6.2f}{dd:>6.1f}%  {desc}")


def _run(per_symbol, start, end, cfg):
    import binance_sim.pit_universe as P
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, P.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    P.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, P.prefetch_universe) = of, oc, oa, op


if __name__ == "__main__":
    main()
