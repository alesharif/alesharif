#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the freshness entry filters (anti top-buying / anti falling-knife).

Diagnosis: the big losses come from entering stale/extended momentum (FET: 52h
old signal at the top) or coins already far below their high (IOTA: -12%). This
sweeps the two freshness filters on Binance 2022-2026 (SL 2.0 / TRAIL 0.2) to
see if they cut tail losses / lift PF without killing the edge.

  near_high_min  : require close >= X * recent 8-bar high (anti falling-knife)
  max_signal_age : require the signal to be <= N bars old (anti stale-trend)

Run:  python experiments/freshness_filter.py 2022-01-01 2026-01-01
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
    wins = [t.net_pct for t in c if t.net_pct > 0]
    losses = [t.net_pct for t in c if t.net_pct <= 0]
    gw, gl = sum(wins), -sum(losses)
    pf = (gw / gl) if gl > 0 else float("inf")
    wr = (len(wins) / len(c) * 100) if c else 0.0
    aw = (sum(wins) / len(wins)) if wins else 0.0
    al = (sum(losses) / len(losses)) if losses else 0.0
    worst = min((t.net_pct for t in c), default=0.0)
    return rep.total_return * 100, len(c), wr, aw, al, worst, pf, rep.max_drawdown * 100


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

    # chosen geometry from the stop/trail sweep
    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 2.0, 0.2, 0.2
    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05)

    # (label, near_high_min, max_signal_age)
    variants = [
        ("BASELINE (no filter)",      0.0,  0),
        ("near-high 0.95",            0.95, 0),
        ("near-high 0.92",            0.92, 0),
        ("signal-age <= 1 (fresh)",   0.0,  1),
        ("signal-age <= 2",           0.0,  2),
        ("signal-age <= 3",           0.0,  3),
        ("near0.92 + age<=3",         0.92, 3),
        ("near0.95 + age<=2",         0.95, 2),
    ]

    print("Freshness filter sweep (SL 2.0 / TRAIL 0.2, mild costs)")
    print(f"{'variant':<24}{'return%':>13}{'trades':>8}{'win%':>7}"
          f"{'avgW':>7}{'avgL':>7}{'worst':>8}{'PF':>6}{'maxDD':>7}")
    print("-" * 87)
    for label, nh, age in variants:
        rep = _run(per_symbol, start, end, dict(base, near_high_min=nh, max_signal_age=age))
        ret, n, wr, aw, al, worst, pf, dd = _metrics(rep)
        print(f"{label:<24}{ret:>12,.0f}%{n:>8}{wr:>6.1f}%{aw:>6.2f}%{al:>6.2f}%"
              f"{worst:>7.1f}%{pf:>6.2f}{dd:>6.1f}%")


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
