#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How well does the edge survive realistic execution costs?

Featurises the full point-in-time universe once, then replays the SAME walk
under several execution-cost scenarios. The market-impact coefficient is a
genuine unknown, so we show a range rather than pretend one number is truth.

Run:  python experiments/execution_costs.py 2022-01-01 2026-01-01
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


# Correct sizing: position = 0.1% of the coin's daily volume, capped at
# portfolio/8 which is itself capped at $1M/8 = $125k. This keeps every order
# at ~0.6% of a 4h bar, so market impact is bounded and realistic.
# (label, slippage_flat, slip_atr, slip_impact)
SCENARIOS = [
    ("fee only (0.2%)",            0.00, 0.00, 0.00),
    ("+ spread 0.05%",             0.05, 0.00, 0.00),
    ("mild  (atr .05, imp .05)",   0.05, 0.05, 0.05),
    ("moderate (atr .10, imp .10)", 0.05, 0.10, 0.10),
    ("harsh (atr .20, imp .20)",   0.10, 0.20, 0.20),
]


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

    # CORRECT sizing per the user's rule: 0.1% of volume, capped at $1M/8.
    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                capital_cap=1_000_000.0)

    print(f"{'scenario':<30}{'return%':>16}{'trades':>9}{'win%':>8}{'PF':>7}{'maxDD%':>9}")
    print("-" * 79)
    for label, slip, satr, simp in SCENARIOS:
        rep = _run(per_symbol, start, end,
                   dict(base, slippage=slip, slip_atr=satr, slip_impact=simp))
        closed = rep.trades
        wins = sum(1 for t in closed if t.net_pct > 0)
        gw = sum(t.net_pct for t in closed if t.net_pct > 0)
        gl = -sum(t.net_pct for t in closed if t.net_pct <= 0)
        pf = (gw / gl) if gl > 0 else float("inf")
        wr = (wins / len(closed) * 100) if closed else 0.0
        print(f"{label:<30}{rep.total_return*100:>15,.0f}%{len(closed):>9}"
              f"{wr:>7.1f}%{pf:>7.2f}{rep.max_drawdown*100:>8.1f}%")


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
