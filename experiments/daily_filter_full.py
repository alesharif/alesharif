#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the top daily-trend filters on the FULL universe (551 coins).

Compares: no filter (baseline) vs BTC-regime vs the best daily candidates from
the 20-coin search (price>EMA20, price>EMA200, EMA50>EMA200 golden), on top of
the chosen stack (SL 2.0 / TRAIL 0.2 + signal-age<=2), Binance 2022-2026.

Run:  python experiments/daily_filter_full.py 2022-01-01 2026-01-01
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import daily_filters as DF               # noqa: E402


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
    print(f"Universe: {len(symbols)} symbols; loading + daily-featurising (cached) ...")
    raw = PIT.prefetch_universe(symbols, fetch_from, end, log=lambda *a: None)
    per_symbol = {}
    for s, df in raw.items():
        d = DF.add_daily_filters(S.attach_entry_signal(S.compute_features(df.copy())))
        per_symbol[s] = d.set_index("close_time")
    print(f"Featurised {len(per_symbol)} symbols.\n")

    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 2.0, 0.2, 0.2
    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05,
                max_signal_age=2)

    variants = [
        ("no filter (baseline)", dict()),
        ("btc_regime",           dict(btc_regime=True)),
        ("price > daily EMA20",  dict(daily_filter_col="d_above_ema20")),
        ("price > daily EMA200", dict(daily_filter_col="d_above_ema200")),
        ("EMA50>EMA200 golden",  dict(daily_filter_col="d_ema50_200")),
    ]

    print("FULL universe (551) | SL 2.0 / age<=2 | mild costs")
    print(f"{'variant':<24}{'return%':>13}{'trades':>8}{'win%':>7}"
          f"{'avgL':>7}{'worst':>8}{'PF':>6}{'maxDD':>7}")
    print("-" * 80)
    for label, extra in variants:
        rep = _run(per_symbol, start, end, dict(base, **extra))
        ret, n, wr, al, worst, pf, dd = _metrics(rep)
        print(f"{label:<24}{ret:>12,.0f}%{n:>8}{wr:>6.1f}%{al:>6.2f}%"
              f"{worst:>7.1f}%{pf:>6.2f}{dd:>6.1f}%")


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
