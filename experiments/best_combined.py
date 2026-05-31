#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best filters COMBINED, on Binance AND OKX side by side.

Stacks the validated improvements together and reports each stack on both
exchanges, so we see the full combination (not isolated layers):

  S0 ORIGINAL  : SL1.5, no filters
  S1 freshness : SL2.0 + age<=2
  S2 + golden  : + daily EMA50>EMA200
  S3 ALL       : + near-high 0.95  (golden + freshness + tail filter)
  S4 htf combo : SL2.0 + age<=2 + htf_trend + near-high 0.95

Run:  python experiments/best_combined.py
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
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
    worst = min((t.net_pct for t in c), default=0.0)
    return len(c), wr, worst, pf, rep.max_drawdown * 100


def featurise(raw):
    out = {}
    for s, df in raw.items():
        d = DF.add_daily_filters(S.attach_entry_signal(S.compute_features(df.copy())))
        out[s] = d.set_index("close_time")
    return out


# (label, SL, TRAIL, extra cfg)
STACKS = [
    ("S0 ORIGINAL (SL1.5)",       1.5, 0.2, dict()),
    ("S1 freshness (age<=2)",     2.0, 0.2, dict(max_signal_age=2)),
    ("S2 +golden",                2.0, 0.2, dict(max_signal_age=2, daily_filter_col="d_ema50_200")),
    ("S3 +golden+near0.95 (ALL)", 2.0, 0.2, dict(max_signal_age=2, daily_filter_col="d_ema50_200", near_high_min=0.95)),
    ("S4 +htf+near0.95",          2.0, 0.2, dict(max_signal_age=2, htf_trend=True, near_high_min=0.95)),
]

COMMON = dict(rank="vol", exit_model="pessimistic",
              vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
              capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05)


def main():
    # --- load both exchanges (cached) ---
    bn_start, bn_end = parse("2022-01-01"), parse("2026-01-01")
    ok_start, ok_end = parse("2024-01-01"), parse("2026-05-30")
    fb = bn_start - PB.WARMUP_BARS * PB.FOUR_H_MS
    fo = ok_start - PB.WARMUP_BARS * PB.FOUR_H_MS

    print("Loading Binance (archive) ...")
    bn = featurise(PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fb, bn_end, log=lambda *a: None))
    print(f"  Binance: {len(bn)} symbols")
    print("Loading OKX ...")
    ok = featurise(OKX.prefetch_universe(OKX.list_usdt_symbols(), fo, ok_end, log=lambda *a: None))
    print(f"  OKX: {len(ok)} symbols\n")

    print("BEST FILTERS COMBINED — Binance (2022-26) vs OKX (2024-26)")
    print(f"{'stack':<28}{'BN PF':>7}{'BN DD':>7}{'BN WR':>7}{'BN wrst':>8}"
          f"{'OK PF':>7}{'OK DD':>7}{'OK WR':>7}{'OK wrst':>8}")
    print("-" * 91)
    for label, sl, tr, extra in STACKS:
        S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = sl, tr, tr
        rb = _run(bn, bn_start, bn_end, dict(COMMON, source="archive", **extra), "archive")
        ro = _run(ok, ok_start, ok_end, dict(COMMON, source="okx", **extra), "okx")
        nb, wrb, wb, pfb, ddb = _metrics(rb)
        no, wro, wo, pfo, ddo = _metrics(ro)
        print(f"{label:<28}{pfb:>7.2f}{ddb:>6.1f}%{wrb:>6.1f}%{wb:>7.1f}%"
              f"{pfo:>7.2f}{ddo:>6.1f}%{wro:>6.1f}%{wo:>7.1f}%")


def _run(per_symbol, start, end, cfg, source):
    of, oc, oa = PB.fetch_klines_range, PB.S.compute_features, PB.S.attach_entry_signal
    op_p, op_o = PIT.prefetch_universe, OKX.prefetch_universe
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    feed = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    PIT.prefetch_universe = feed
    OKX.prefetch_universe = feed
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        PB.fetch_klines_range, PB.S.compute_features, PB.S.attach_entry_signal = of, oc, oa
        PIT.prefetch_universe, OKX.prefetch_universe = op_p, op_o


if __name__ == "__main__":
    main()
