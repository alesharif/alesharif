#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freshness sweep on the ORIGINAL strategy (no trend filter, no slip).

Compares the original vs signal-age<=1/2/3/4. Full period, all coins, fees only.
Shows monthly return% + WR, and per-variant PF / WR / PP, buckets, 2025 focus.

Run:  python experiments/freshness_clean.py
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

OUT = "results_freshness"
WIN_BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 1e9)]
LOSS_BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 1e9)]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def classify(trades):
    wins = [t.net_pct for t in trades if t.net_pct > 0]
    losses = [-t.net_pct for t in trades if t.net_pct <= 0]
    n = len(trades)
    wb = [(lo, hi, sum(1 for v in wins if lo <= v < hi),
           100 * sum(1 for v in wins if lo <= v < hi) / n if n else 0) for lo, hi in WIN_BUCKETS]
    lb = [(lo, hi, sum(1 for v in losses if lo <= v < hi),
           100 * sum(1 for v in losses if lo <= v < hi) / n if n else 0) for lo, hi in LOSS_BUCKETS]
    return wb, lb


def monthly_table(rep):
    eq = pd.Series(rep.equity_curve, index=pd.to_datetime(rep.equity_times, unit="ms"))
    m = eq.resample("ME").last()
    ret = m.pct_change() * 100
    ret.iloc[0] = (m.iloc[0] / rep.initial_capital - 1) * 100
    tdf = pd.DataFrame([(pd.to_datetime(t.exit_time, unit="ms"), t.net_pct) for t in rep.trades],
                       columns=["dt", "net"])
    out = {}
    for ts, r in ret.items():
        if pd.isna(r):
            continue
        mt = tdf[(tdf.dt.dt.year == ts.year) & (tdf.dt.dt.month == ts.month)]
        wr = (mt.net > 0).mean() * 100 if len(mt) else 0
        out[ts.strftime("%Y-%m")] = (round(r, 1), round(wr, 0))
    return out


def summary(rep):
    c = rep.trades
    wins = [t.net_pct for t in c if t.net_pct > 0]
    losses = [t.net_pct for t in c if t.net_pct <= 0]
    gw, gl = sum(wins), -sum(losses)
    pf = (gw / gl) if gl > 0 else float("inf")
    wr = (len(wins) / len(c) * 100) if c else 0.0
    aw = (sum(wins) / len(wins)) if wins else 0.0
    al = (sum(losses) / len(losses)) if losses else 0.0
    pp = (aw / -al) if al < 0 else float("inf")
    worst = min((t.net_pct for t in c), default=0.0)
    return dict(ret=round(rep.total_return * 100, 1), wr=round(wr, 1), pf=round(pf, 2),
                pp=round(pp, 2), aw=round(aw, 2), al=round(al, 2),
                worst=round(worst, 1), dd=round(rep.max_drawdown * 100, 1), n=len(c))


def run_variant(per_symbol, start, end, age):
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, PIT.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    PIT.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    cfg = dict(rank="vol", exit_model="pessimistic", source="archive",
               vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
               capital_cap=1_000_000.0, max_signal_age=age)   # age=0 -> off
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, PIT.prefetch_universe) = of, oc, oa, op


def main():
    start, end = parse("2022-01-01"), parse("2026-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS
    print("Loading + featurising Binance (all coins, cached) ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fetch_from, end, log=lambda *a: None)
    per_symbol = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for s, df in raw.items()}
    print(f"  {len(per_symbol)} symbols\n")

    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 1.5, 0.2, 0.2

    variants = [("ORIGINAL", 0), ("age<=1", 1), ("age<=2", 2), ("age<=3", 3), ("age<=4", 4)]
    reps = {lbl: run_variant(per_symbol, start, end, age) for lbl, age in variants}
    months = {lbl: monthly_table(r) for lbl, r in reps.items()}
    sums = {lbl: summary(r) for lbl, r in reps.items()}

    labels = [lbl for lbl, _ in variants]

    # ---- monthly: return% per variant ----
    print("MONTHLY RETURN %  (per variant)")
    head = f"{'month':<9}" + "".join(f"{l:>10}" for l in labels)
    print(head); print("-" * len(head))
    for k in sorted(set().union(*[set(m) for m in months.values()])):
        row = f"{k:<9}"
        for l in labels:
            v = months[l].get(k, (float('nan'), 0))[0]
            row += f"{v:>9.1f}%"
        print(row)

    # ---- monthly: WR% per variant ----
    print("\nMONTHLY WIN-RATE %  (per variant)")
    print(head); print("-" * len(head))
    for k in sorted(set().union(*[set(m) for m in months.values()])):
        row = f"{k:<9}"
        for l in labels:
            v = months[l].get(k, (0, float('nan')))[1]
            row += f"{v:>9.0f}%"
        print(row)

    # ---- summary ----
    print("\nSUMMARY")
    print(f"{'metric':<10}" + "".join(f"{l:>10}" for l in labels))
    for key, name in [("ret", "return%"), ("wr", "WR%"), ("pf", "PF"), ("pp", "PP"),
                      ("aw", "avgWin%"), ("al", "avgLoss%"), ("worst", "worst%"),
                      ("dd", "maxDD%"), ("n", "trades")]:
        print(f"{name:<10}" + "".join(f"{str(sums[l][key]):>10}" for l in labels))

    # ---- buckets (all years) ----
    print("\nTRADE BUCKETS (all years)  count / % of all trades")
    for l in labels:
        wb, lb = classify(reps[l].trades)
        n = sums[l]["n"]
        wins = sum(c for _, _, c, _ in wb); losses = sum(c for _, _, c, _ in lb)
        print(f"\n  [{l}]  trades={n} wins={wins} losses={losses}")
        ws = "   WIN: " + "  ".join(f"{lo:g}-{(hi if hi<1e8 else 99):g}%:{c}({p:.1f}%)"
                                    for lo, hi, c, p in wb)
        ls = "   LOSS:" + "  ".join(f"{lo:g}-{(hi if hi<1e8 else 99):g}%:{c}({p:.1f}%)"
                                    for lo, hi, c, p in lb)
        print(ws); print(ls)

    # ---- 2025 focus ----
    print("\n" + "=" * 50 + "\n  FOCUS: 2025\n" + "=" * 50)
    print("Monthly return% 2025:")
    print(head); print("-" * len(head))
    for k in sorted([m for m in set().union(*[set(mm) for mm in months.values()]) if m.startswith("2025")]):
        row = f"{k:<9}"
        for l in labels:
            row += f"{months[l].get(k,(float('nan'),0))[0]:>9.1f}%"
        print(row)
    print("\n2025 trade buckets:")
    for l in labels:
        t25 = [t for t in reps[l].trades if pd.to_datetime(t.exit_time, unit="ms").year == 2025]
        wb, lb = classify(t25)
        wins = sum(c for _, _, c, _ in wb); losses = sum(c for _, _, c, _ in lb)
        print(f"\n  [{l}]  2025 trades={len(t25)} wins={wins} losses={losses}")
        print("   WIN: " + "  ".join(f"{lo:g}-{(hi if hi<1e8 else 99):g}%:{c}({p:.1f}%)" for lo, hi, c, p in wb))
        print("   LOSS:" + "  ".join(f"{lo:g}-{(hi if hi<1e8 else 99):g}%:{c}({p:.1f}%)" for lo, hi, c, p in lb))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "freshness_data.json"), "w") as f:
        json.dump(dict(months=months, summary=sums), f, indent=2, default=str)
    print("\nSaved -> results_freshness/freshness_data.json")


if __name__ == "__main__":
    main()
