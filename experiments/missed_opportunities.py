#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Missed-opportunity analysis: April 2025 (strong bull month, BTC +14%).

The 4h model showed +280% but the realistic hybrid only +3.36%. This finds out
WHY by measuring, for every coin:
  * its biggest favourable run during the month (best entry->later high within
    a holding horizon), i.e. the opportunity that existed.
  * whether OUR strategy ever signalled an entry on it, and what happened.

Outputs:
  A) opportunity census: how many coins had a >=X% run available.
  B) capture: of the big movers, how many did the strategy enter? when (how far
     into the run)? what was the realised vs available move?
  C) the coins the strategy entered and LOST on - were they real movers?

Run:  python experiments/missed_opportunities.py
"""

from __future__ import annotations

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FOUR_H = 4 * 3600 * 1000
HOLD_BARS = 12   # ~2 days: how long forward we look for the favourable run


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def main():
    start, end = parse("2025-04-01"), parse("2025-05-01")
    ff = start - PB.WARMUP_BARS * FOUR_H
    print("Loading + featurising April 2025 (all coins) ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for s, df in raw.items()}
    print(f"  {len(ps)} symbols\n")

    # ---- A) opportunity census: best forward run per coin during the month ----
    # For each coin, scan every 4h bar in April; the "available run" from that bar
    # is (max high over next HOLD_BARS) / close - 1. Track each coin's best run.
    best_run = {}            # symbol -> best available % run starting in April
    best_run_bar = {}        # symbol -> (entry_time, entry_close, peak)
    for sym, df in ps.items():
        idx = [t for t in df.index if start <= t <= end]
        if not idx:
            continue
        arr = df.loc[idx]
        closes = arr["close"].to_numpy()
        highs = arr["high"].to_numpy()
        n = len(closes)
        bestp = 0.0; bestinfo = None
        for i in range(n):
            j = min(i + HOLD_BARS, n)
            fwd_high = highs[i:j].max() if j > i else highs[i]
            run = (fwd_high / closes[i] - 1) * 100
            if run > bestp:
                bestp = run; bestinfo = (idx[i], closes[i], fwd_high)
        best_run[sym] = bestp
        best_run_bar[sym] = bestinfo

    runs = pd.Series(best_run)
    print("=== A) OPPORTUNITY CENSUS (best available 2-day run per coin, April 2025) ===")
    for thr in [10, 20, 30, 50, 100]:
        cnt = (runs >= thr).sum()
        print(f"  coins with a >= {thr}% run available : {cnt}")
    print(f"  median best-run across all coins: {runs.median():.1f}%")
    print(f"  top movers: " + ", ".join(f"{s}(+{runs[s]:.0f}%)" for s in runs.nlargest(8).index))

    # ---- B) what did OUR strategy actually enter? ----
    entered = {}   # symbol -> list of (entry_time, entry_close)
    for sym, df in ps.items():
        for t in [x for x in df.index if start <= x <= end]:
            row = df.loc[t]
            if bool(row["entry_signal"]):
                entered.setdefault(sym, []).append((t, float(row["close"])))

    big_movers = set(runs[runs >= 30].index)        # coins that ran >=30%
    entered_syms = set(entered)
    print(f"\n=== B) CAPTURE of big movers (>=30% run; {len(big_movers)} coins) ===")
    caught = big_movers & entered_syms
    missed = big_movers - entered_syms
    print(f"  big movers the strategy ENTERED (at least once): {len(caught)} / {len(big_movers)}")
    print(f"  big movers COMPLETELY MISSED (never signalled):  {len(missed)}")
    if missed:
        ex = sorted(missed, key=lambda s: -runs[s])[:8]
        print("  examples missed: " + ", ".join(f"{s}(+{runs[s]:.0f}%)" for s in ex))

    # For caught big movers: how late was entry vs the run start, and how much of
    # the run was still available after entry?
    print("\n  For CAUGHT big movers - entry timing vs the move:")
    print(f"  {'symbol':<12}{'run%':>7}{'entry_vs_runstart':>20}{'avail_after_entry%':>20}")
    late_data = []
    for sym in sorted(caught, key=lambda s: -runs[s])[:15]:
        df = ps[sym]
        rs_t, rs_close, peak = best_run_bar[sym]   # where the best run started
        # first entry at or after run start
        my = [(t, c) for (t, c) in entered[sym]]
        first = my[0]
        # how much upside remained from our entry to the coin's later peak
        fut = df[(df.index >= first[0]) & (df.index <= first[0] + HOLD_BARS * FOUR_H)]
        avail_after = (fut["high"].max() / first[1] - 1) * 100 if len(fut) else 0
        # entry timing: our entry close vs run-start close (how far up we bought)
        up_at_entry = (first[1] / rs_close - 1) * 100
        late_data.append(avail_after)
        print(f"  {sym:<12}{runs[sym]:>6.0f}%{up_at_entry:>18.1f}%{avail_after:>19.1f}%")
    if late_data:
        print(f"\n  median upside still available AFTER our entry: {np.median(late_data):.1f}%")
        print("  (if this is large, the move was there but our EXIT gave it back)")


if __name__ == "__main__":
    main()
