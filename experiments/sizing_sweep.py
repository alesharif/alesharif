#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find the best volume-% for entries, and diagnose which cost dominates.

PART A — sweep the entry size (0.05% .. 0.10% of the coin's daily volume)
         under a realistic cost scenario, to find the best risk/return.
PART B — cost decomposition at 0.10%: turn each cost component on alone
         (spread / ATR-stop slippage / market impact) to see which one
         actually eats the edge -> answers "is it the spread?".

Featurises the universe once, then replays the walk per case.

Run:  python experiments/sizing_sweep.py 2022-01-01 2026-01-01
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
    closed = rep.trades
    wins = sum(1 for t in closed if t.net_pct > 0)
    gw = sum(t.net_pct for t in closed if t.net_pct > 0)
    gl = -sum(t.net_pct for t in closed if t.net_pct <= 0)
    pf = (gw / gl) if gl > 0 else float("inf")
    wr = (wins / len(closed) * 100) if closed else 0.0
    return rep.total_return * 100, len(closed), wr, pf, rep.max_drawdown * 100


def main():
    start = parse(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
    end = parse(sys.argv[2] if len(sys.argv) > 2 else "2026-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS

    symbols = PIT.list_all_usdt_symbols()
    print(f"Universe: {len(symbols)} symbols; loading archive (cached) ...")
    raw = PIT.prefetch_universe(symbols, fetch_from, end, log=lambda *a: None)
    per_symbol = {}
    for sym, df in raw.items():
        d = S.attach_entry_signal(S.compute_features(df.copy()))
        per_symbol[sym] = d.set_index("close_time")
    print(f"Featurised {len(per_symbol)} symbols.\n")

    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, capital_cap=1_000_000.0)

    # ---- PART A: volume-% sweep under a realistic (mild) cost ----------
    print("PART A — best entry size (mild realistic cost: spread .05, atr .05, impact .05)")
    print(f"{'vol %':<10}{'return%':>16}{'trades':>9}{'win%':>8}{'PF':>7}{'maxDD%':>9}")
    print("-" * 59)
    for pct in [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010]:
        rep = _run(per_symbol, start, end, dict(
            base, vol_size_pct=pct, slippage=0.05, slip_atr=0.05, slip_impact=0.05))
        ret, n, wr, pf, dd = _metrics(rep)
        print(f"{pct*100:<10.2f}{ret:>15,.0f}%{n:>9}{wr:>7.1f}%{pf:>7.2f}{dd:>8.1f}%")

    # ---- PART B: which cost component dominates? (at 0.10%) -------------
    print("\nPART B — cost decomposition at 0.10% (is it the spread?)")
    print(f"{'component on':<26}{'return%':>16}{'win%':>8}{'PF':>7}{'maxDD%':>9}")
    print("-" * 66)
    cases = [
        ("fee only (0.2%)",        dict(slippage=0.00, slip_atr=0.00, slip_impact=0.00)),
        ("+ spread 0.05% ONLY",    dict(slippage=0.05, slip_atr=0.00, slip_impact=0.00)),
        ("+ ATR-stop slip ONLY",   dict(slippage=0.00, slip_atr=0.05, slip_impact=0.00)),
        ("+ market impact ONLY",   dict(slippage=0.00, slip_atr=0.00, slip_impact=0.05)),
        ("all three (mild)",       dict(slippage=0.05, slip_atr=0.05, slip_impact=0.05)),
    ]
    for label, costs in cases:
        rep = _run(per_symbol, start, end, dict(base, vol_size_pct=0.0010, **costs))
        ret, n, wr, pf, dd = _metrics(rep)
        print(f"{label:<26}{ret:>15,.0f}%{wr:>7.1f}%{pf:>7.2f}{dd:>8.1f}%")


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
