#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Out-of-sample cross-exchange validation on OKX of the improvement stack.

Compares, on OKX spot (a different venue), each layer we added on Binance:
  1. ORIGINAL    : SL 1.5 / TRAIL 0.2, no freshness, no daily filter
  2. + freshness : SL 2.0 + signal-age<=2
  3. + golden    : freshness + daily EMA50>EMA200
  4. + htf_trend : freshness + ~daily-50 trend

If the improvements hold on OKX too, they generalise; if not, they were
Binance-specific. We do NOT tune anything to OKX (held-out).

Run:  python experiments/okx_validation.py 2024-01-01 2026-05-30
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import okx_source as OKX                 # noqa: E402
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
    start = parse(sys.argv[1] if len(sys.argv) > 1 else "2024-01-01")
    end = parse(sys.argv[2] if len(sys.argv) > 2 else "2026-05-30")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS

    syms = OKX.list_usdt_symbols()
    print(f"OKX universe: {len(syms)} symbols; fetching + featurising ...")
    raw = OKX.prefetch_universe(syms, fetch_from, end, log=print)
    per_symbol = {}
    for s, df in raw.items():
        d = DF.add_daily_filters(S.attach_entry_signal(S.compute_features(df.copy())))
        per_symbol[s] = d.set_index("close_time")
    print(f"Featurised {len(per_symbol)} OKX symbols.\n")

    common = dict(rank="vol", exit_model="pessimistic", source="okx",
                  vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                  capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05)

    # (label, SL, TRAIL, extra cfg)
    variants = [
        ("ORIGINAL (SL1.5)",      1.5, 0.2, dict()),
        ("+ freshness (SL2/age2)", 2.0, 0.2, dict(max_signal_age=2)),
        ("+ golden EMA50>200",    2.0, 0.2, dict(max_signal_age=2, daily_filter_col="d_ema50_200")),
        ("+ htf_trend",           2.0, 0.2, dict(max_signal_age=2, htf_trend=True)),
    ]

    print("OKX OUT-OF-SAMPLE validation  | mild costs")
    print(f"{'variant':<24}{'return%':>12}{'trades':>8}{'win%':>7}"
          f"{'avgL':>7}{'worst':>8}{'PF':>6}{'maxDD':>7}")
    print("-" * 79)
    for label, sl, tr, extra in variants:
        S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = sl, tr, tr
        rep = _run(per_symbol, start, end, dict(common, **extra))
        ret, n, wr, al, worst, pf, dd = _metrics(rep)
        print(f"{label:<24}{ret:>11,.0f}%{n:>8}{wr:>6.1f}%{al:>6.2f}%"
              f"{worst:>7.1f}%{pf:>6.2f}{dd:>6.1f}%")


def _run(per_symbol, start, end, cfg):
    import binance_sim.okx_source as O
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, O.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    O.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, O.prefetch_universe) = of, oc, oa, op


if __name__ == "__main__":
    main()
