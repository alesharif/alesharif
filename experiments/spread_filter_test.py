#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does spread-aware entry filtering improve net-of-cost performance?

Charges a per-coin round-trip spread cost (Corwin-Schultz estimate) on top of
mild slippage/impact, then tightens the spread filter:
    skip entry if  est_spread > ratio * (ATR / price)
i.e. only take high-spread coins when the volatility target is large enough.

If tightening the filter raises net PF / cuts drawdown while trimming the worst
trades, the user's "control the entry, filter by spread/target" idea works.

Run:  python experiments/spread_filter_test.py 2022-01-01 2026-01-01
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def _metrics(rep):
    c = rep.trades
    wins = sum(1 for t in c if t.net_pct > 0)
    gw = sum(t.net_pct for t in c if t.net_pct > 0)
    gl = -sum(t.net_pct for t in c if t.net_pct <= 0)
    pf = (gw / gl) if gl > 0 else float("inf")
    wr = (wins / len(c) * 100) if c else 0.0
    return rep.total_return * 100, len(c), wr, pf, rep.max_drawdown * 100


def main():
    start = parse(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
    end = parse(sys.argv[2] if len(sys.argv) > 2 else "2026-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS

    symbols = PIT.list_all_usdt_symbols()
    print(f"Universe: {len(symbols)} symbols; loading archive (cached) ...")
    raw = PIT.prefetch_universe(symbols, fetch_from, end, log=lambda *a: None)
    per_symbol = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for s, df in raw.items()}
    print(f"Featurised {len(per_symbol)} symbols.\n")

    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                capital_cap=1_000_000.0,
                slippage=0.0, slip_atr=0.05, slip_impact=0.05,  # mild slippage
                est_spread=True)                                 # per-coin spread cost ON

    print("Spread filter sweep (per-coin CS spread cost ON, mild slippage)")
    print(f"{'filter ratio':<16}{'return%':>16}{'trades':>9}{'win%':>8}{'PF':>7}{'maxDD%':>9}")
    print("-" * 65)
    for ratio in [0.0, 0.50, 0.25, 0.15, 0.10, 0.05]:
        rep = _run(per_symbol, start, end, dict(base, spread_filter_ratio=ratio))
        ret, n, wr, pf, dd = _metrics(rep)
        label = "OFF" if ratio == 0 else f"{ratio:.2f}"
        print(f"{label:<16}{ret:>15,.0f}%{n:>9}{wr:>7.1f}%{pf:>7.2f}{dd:>8.1f}%")


def _run(per_symbol, start, end, cfg):
    import binance_sim.pit_universe as P
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, P.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    P.prefetch_universe = lambda syms, a, b, **k: {
        s: d.reset_index() for s, d in per_symbol.items()}
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, P.prefetch_universe) = of, oc, oa, op


if __name__ == "__main__":
    main()
