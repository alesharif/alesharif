#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A vs B: original vs +golden-trend filter. Full period, all coins, fees only.

A = ORIGINAL : SL1.5/TRAIL0.2, vol-sizing 0.1% capped $125k, fees only (no slip)
B = + golden : same, plus daily EMA50>EMA200 trend filter

Outputs (saved to results_AB/):
  * full monthly table (return% and win-rate%) for A and B
  * trade win/loss bucket classification for A and B
  * a dedicated 2025 section (monthly + buckets)
  * summary stats

Run:  python experiments/ab_compare.py
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
from binance_sim import daily_filters as DF               # noqa: E402

OUT = "results_AB"


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


WIN_BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 1e9)]
LOSS_BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 1e9)]  # abs of negative


def classify(trades):
    wins = [t.net_pct for t in trades if t.net_pct > 0]
    losses = [-t.net_pct for t in trades if t.net_pct <= 0]
    n = len(trades)
    wb, lb = [], []
    for lo, hi in WIN_BUCKETS:
        cnt = sum(1 for v in wins if lo <= v < hi)
        wb.append((lo, hi, cnt, 100 * cnt / n if n else 0))
    for lo, hi in LOSS_BUCKETS:
        cnt = sum(1 for v in losses if lo <= v < hi)
        lb.append((lo, hi, cnt, 100 * cnt / n if n else 0))
    return wb, lb, len(wins), len(losses)


def monthly_table(rep):
    eq = pd.Series(rep.equity_curve, index=pd.to_datetime(rep.equity_times, unit="ms"))
    m = eq.resample("ME").last()
    ret = m.pct_change() * 100
    ret.iloc[0] = (m.iloc[0] / rep.initial_capital - 1) * 100
    # per-month win rate from trades (by exit time)
    tdf = pd.DataFrame([(pd.to_datetime(t.exit_time, unit="ms"), t.net_pct) for t in rep.trades],
                       columns=["dt", "net"])
    out = {}
    for ts, r in ret.items():
        if pd.isna(r):
            continue
        key = ts.strftime("%Y-%m")
        mt = tdf[(tdf.dt.dt.year == ts.year) & (tdf.dt.dt.month == ts.month)]
        wr = (mt.net > 0).mean() * 100 if len(mt) else float("nan")
        out[key] = (round(r, 1), round(wr, 1) if len(mt) else None, len(mt))
    return out


def run_variant(per_symbol, start, end, extra):
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, PIT.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    PIT.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    cfg = dict(rank="vol", exit_model="pessimistic", source="archive",
               vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
               capital_cap=1_000_000.0, **extra)   # fees only: no slippage args
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, PIT.prefetch_universe) = of, oc, oa, op


def summary(rep):
    c = rep.trades
    wins = [t.net_pct for t in c if t.net_pct > 0]
    losses = [t.net_pct for t in c if t.net_pct <= 0]
    gw, gl = sum(wins), -sum(losses)
    pf = (gw / gl) if gl > 0 else float("inf")
    wr = (len(wins) / len(c) * 100) if c else 0.0
    worst = min((t.net_pct for t in c), default=0.0)
    return dict(trades=len(c), win_rate=round(wr, 1), pf=round(pf, 2),
                worst=round(worst, 1), maxdd=round(rep.max_drawdown * 100, 1),
                total_return=round(rep.total_return * 100, 1))


def main():
    start, end = parse("2022-01-01"), parse("2026-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS
    print("Loading + featurising Binance (all coins, cached) ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fetch_from, end, log=lambda *a: None)
    per_symbol = {s: DF.add_daily_filters(S.attach_entry_signal(S.compute_features(df.copy()))).set_index("close_time")
                  for s, df in raw.items()}
    print(f"  {len(per_symbol)} symbols\n")

    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 1.5, 0.2, 0.2   # ORIGINAL geometry

    repA = run_variant(per_symbol, start, end, dict())                              # original
    repB = run_variant(per_symbol, start, end, dict(daily_filter_col="d_ema50_200"))  # +golden

    mA, mB = monthly_table(repA), monthly_table(repB)
    sA, sB = summary(repA), summary(repB)
    wbA, lbA, nwA, nlA = classify(repA.trades)
    wbB, lbB, nwB, nlB = classify(repB.trades)

    os.makedirs(OUT, exist_ok=True)
    data = dict(months_A=mA, months_B=mB, summary_A=sA, summary_B=sB,
                winbuckets_A=wbA, lossbuckets_A=lbA,
                winbuckets_B=wbB, lossbuckets_B=lbB)
    with open(os.path.join(OUT, "ab_data.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)

    # ---- print monthly table ----
    print("MONTHLY  (A=original, B=+golden)   ret% | winrate%")
    print(f"{'month':<10}{'A ret':>8}{'A WR':>7}{'B ret':>8}{'B WR':>7}")
    print("-" * 40)
    for k in sorted(set(mA) | set(mB)):
        ra, wa, _ = mA.get(k, (float('nan'), None, 0))
        rb, wb_, _ = mB.get(k, (float('nan'), None, 0))
        print(f"{k:<10}{ra:>7.1f}%{(wa if wa is not None else 0):>6.0f}%"
              f"{rb:>7.1f}%{(wb_ if wb_ is not None else 0):>6.0f}%")

    print("\nSUMMARY        A(original)     B(+golden)")
    for key in ["total_return", "win_rate", "pf", "worst", "maxdd", "trades"]:
        print(f"  {key:<14}{str(sA[key]):>12}{str(sB[key]):>14}")

    def show_buckets(title, wb, lb, nw, nl, ntot):
        print(f"\n{title}  (total trades={ntot}, wins={nw}, losses={nl})")
        print("  WIN buckets:")
        for lo, hi, cnt, pct in wb:
            lab = f"+{lo:g}..{hi:g}%" if hi < 1e8 else f">+{lo:g}%"
            print(f"    {lab:<12}{cnt:>7}  {pct:>5.1f}%")
        print("  LOSS buckets:")
        for lo, hi, cnt, pct in lb:
            lab = f"-{lo:g}..{hi:g}%" if hi < 1e8 else f"<-{lo:g}%"
            print(f"    {lab:<12}{cnt:>7}  {pct:>5.1f}%")

    show_buckets("BUCKETS A (original) - ALL YEARS", wbA, lbA, nwA, nlA, sA["trades"])
    show_buckets("BUCKETS B (+golden)  - ALL YEARS", wbB, lbB, nwB, nlB, sB["trades"])

    # ---- 2025-only focus ----
    def only_2025(trades):
        return [t for t in trades if pd.to_datetime(t.exit_time, unit="ms").year == 2025]
    a25, b25 = only_2025(repA.trades), only_2025(repB.trades)
    wbA25, lbA25, nwA25, nlA25 = classify(a25)
    wbB25, lbB25, nwB25, nlB25 = classify(b25)
    print("\n" + "=" * 50 + "\n  FOCUS: 2025 ONLY\n" + "=" * 50)
    print("Monthly 2025:")
    print(f"{'month':<10}{'A ret':>8}{'A WR':>7}{'B ret':>8}{'B WR':>7}")
    for k in sorted([m for m in set(mA) | set(mB) if m.startswith("2025")]):
        ra, wa, _ = mA.get(k, (float('nan'), None, 0))
        rb, wb_, _ = mB.get(k, (float('nan'), None, 0))
        print(f"{k:<10}{ra:>7.1f}%{(wa if wa is not None else 0):>6.0f}%"
              f"{rb:>7.1f}%{(wb_ if wb_ is not None else 0):>6.0f}%")
    show_buckets("BUCKETS A (original) - 2025", wbA25, lbA25, nwA25, nlA25, len(a25))
    show_buckets("BUCKETS B (+golden)  - 2025", wbB25, lbB25, nwB25, nlB25, len(b25))
    data["b2025"] = dict(a_buckets=[wbA25, lbA25], b_buckets=[wbB25, lbB25],
                         a_n=len(a25), b_n=len(b25))
    with open(os.path.join(OUT, "ab_data.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)
    print("\nSaved -> results_AB/ab_data.json")


if __name__ == "__main__":
    main()
