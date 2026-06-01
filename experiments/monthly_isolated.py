#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-month isolated returns: each month starts fresh at $2,000.

Removes the compounding/cap distortion: every calendar month is its own
backtest seeded at $2,000, with prior warm-up bars so indicators (EMA200 etc.)
are valid from day one. Compares ORIGINAL vs +freshness(age<=2).

Full period, all coins, fees only, Binance.

Run:  python experiments/monthly_isolated.py
"""

from __future__ import annotations

import json
import os
import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

OUT = "results_monthly_iso"


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def run_window(per_symbol, start_ms, end_ms, age):
    """Run one isolated window seeded at $2,000."""
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, PIT.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    PIT.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    cfg = dict(rank="vol", exit_model="pessimistic", source="archive",
               vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
               capital_cap=1_000_000.0, max_signal_age=age)
    try:
        return PB.run(list(per_symbol.keys()), start_ms, end_ms, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, PIT.prefetch_universe) = of, oc, oa, op


def month_metrics(rep):
    c = rep.trades
    wins = [t.net_pct for t in c if t.net_pct > 0]
    wr = (len(wins) / len(c) * 100) if c else 0.0
    return round(rep.total_return * 100, 1), round(wr, 0), len(c)


def main():
    # one big load covering everything (with warm-up before 2022)
    full_start = parse("2022-01-01")
    full_end = parse("2026-01-01")
    fetch_from = full_start - PB.WARMUP_BARS * PB.FOUR_H_MS
    print("Loading + featurising Binance (all coins, cached) ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fetch_from, full_end, log=lambda *a: None)
    per_symbol = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for s, df in raw.items()}
    print(f"  {len(per_symbol)} symbols\n")

    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 1.5, 0.2, 0.2

    months = pd.period_range("2022-01", "2025-12", freq="M")
    rows = []
    for p in months:
        m_start = parse(str(p.start_time.date()))
        m_end = parse(str((p + 1).start_time.date()))
        # ORIGINAL (age=0) and freshness (age=2)
        rA = run_window(per_symbol, m_start, m_end, 0)
        rB = run_window(per_symbol, m_start, m_end, 2)
        retA, wrA, nA = month_metrics(rA)
        retB, wrB, nB = month_metrics(rB)
        rows.append((str(p), retA, wrA, nA, retB, wrB, nB))
        print(f"{str(p):<9}  A:{retA:>7.1f}% WR{wrA:>3.0f}% n{nA:<4}   "
              f"B:{retB:>7.1f}% WR{wrB:>3.0f}% n{nB}")

    # ---- summary ----
    def stats(idx):
        vals = [r[idx] for r in rows]
        pos = sum(1 for v in vals if v > 0)
        return (round(sum(vals) / len(vals), 2), pos, len(vals) - pos,
                round(max(vals), 1), round(min(vals), 1))
    aA = stats(1); aB = stats(4)
    print("\nSUMMARY (each month seeded at $2,000)")
    print(f"{'':<18}{'ORIGINAL':>12}{'age<=2':>12}")
    print(f"{'avg month %':<18}{aA[0]:>12}{aB[0]:>12}")
    print(f"{'positive months':<18}{aA[1]:>12}{aB[1]:>12}")
    print(f"{'negative months':<18}{aA[2]:>12}{aB[2]:>12}")
    print(f"{'best month %':<18}{aA[3]:>12}{aB[3]:>12}")
    print(f"{'worst month %':<18}{aA[4]:>12}{aB[4]:>12}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "monthly_iso.json"), "w") as f:
        json.dump(dict(rows=rows, summary=dict(original=aA, age2=aB)), f, indent=2)
    print("\nSaved -> results_monthly_iso/monthly_iso.json")


if __name__ == "__main__":
    main()
