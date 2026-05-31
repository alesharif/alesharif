#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test higher-timeframe (daily) trend filters on top of the freshness filter.

Builds on the chosen stack (SL 2.0 / TRAIL 0.2 + signal-age<=2) and adds:
  htf_trend  : only enter coins above their ~daily-50 trend (EMA300 on 4h)
  btc_regime : skip ALL entries when BTC's daily trend is down (market regime)

The btc_regime filter directly targets the correlation-cluster losses seen on
down-market days.

Run:  python experiments/daily_filter.py 2022-01-01 2026-01-01
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
    al = (sum(losses) / len(losses)) if losses else 0.0
    worst = min((t.net_pct for t in c), default=0.0)
    return rep.total_return * 100, len(c), wr, al, worst, pf, rep.max_drawdown * 100


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

    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 2.0, 0.2, 0.2
    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05,
                max_signal_age=2)   # keep the freshness winner

    variants = [
        ("freshness only (age<=2)",   dict()),
        ("+ htf_trend (coin daily)",  dict(htf_trend=True)),
        ("+ btc_regime (market)",     dict(btc_regime=True)),
        ("+ htf + btc",               dict(htf_trend=True, btc_regime=True)),
        ("+ htf + btc + near0.95",    dict(htf_trend=True, btc_regime=True, near_high_min=0.95)),
    ]

    print("Daily-trend filters on top of SL2.0 / age<=2 (mild costs)")
    print(f"{'variant':<26}{'return%':>13}{'trades':>8}{'win%':>7}"
          f"{'avgL':>7}{'worst':>8}{'PF':>6}{'maxDD':>7}")
    print("-" * 82)
    for label, extra in variants:
        rep = _run(per_symbol, start, end, dict(base, **extra))
        ret, n, wr, al, worst, pf, dd = _metrics(rep)
        print(f"{label:<26}{ret:>12,.0f}%{n:>8}{wr:>6.1f}%{al:>6.2f}%"
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
