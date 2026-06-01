#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay the ORIGINAL strategy over the paper-bot's live window for comparison.

Window: 2026-05-31 -> 2026-06-01 (when the live paper bot started).
Settings mirror the live bot exactly: $2,000, 8 slots, $250 fixed per trade,
SL 1.5*ATR, TRAIL 0.2*ATR, fees only, all coins. Open positions stay open;
we report CLOSED trades only.

Run:  python experiments/live_window.py
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


def main():
    start = parse("2026-05-31")
    end = parse("2026-06-02")          # a little past the data end; engine stops at last bar
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS
    print("Loading + featurising Binance (all coins) ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fetch_from, end, log=lambda *a: None)
    per_symbol = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for s, df in raw.items()}
    print(f"  {len(per_symbol)} symbols\n")

    # mirror the live bot: $250 fixed per trade (=2000/8), SL1.5/TRAIL0.2, fees only
    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 1.5, 0.2, 0.2
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, PIT.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    PIT.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    try:
        rep = PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None,
                     rank="vol", exit_model="pessimistic", source="archive",
                     fixed_notional=True)   # $250 fixed (INITIAL_CAPITAL*POSITION_PCT)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, PIT.prefetch_universe) = of, oc, oa, op

    # only CLOSED trades that are NOT the forced end-of-data close (EOD)
    closed = [t for t in rep.trades if t.outcome != "EOD"]
    eod = [t for t in rep.trades if t.outcome == "EOD"]  # would-be still-open

    print(f"{'symbol':<12}{'entry_time':<18}{'entry':>10}{'exit_time':<18}"
          f"{'exit':>10}{'outcome':>8}{'net%':>8}{'pnl$':>8}")
    print("-" * 92)
    realized = 0.0
    for t in sorted(closed, key=lambda x: x.exit_time):
        et = pd.to_datetime(t.entry_time, unit="ms").strftime("%m-%d %H:%M")
        xt = pd.to_datetime(t.exit_time, unit="ms").strftime("%m-%d %H:%M")
        realized += t.pnl
        print(f"{t.symbol:<12}{et:<18}{t.entry_price:>10.5g}{xt:<18}"
              f"{t.exit_price:>10.5g}{t.outcome:>8}{t.net_pct:>7.2f}%{t.pnl:>7.2f}")

    wins = [t for t in closed if t.net_pct > 0]
    losses = [t for t in closed if t.net_pct <= 0]
    print("\n" + "=" * 50)
    print(f"CLOSED trades : {len(closed)}  (wins {len(wins)}, losses {len(losses)})")
    print(f"Win rate      : {len(wins)/len(closed)*100:.1f}%" if closed else "no closed trades")
    print(f"Gross win $   : {sum(t.pnl for t in wins):+.2f}")
    print(f"Gross loss $  : {sum(t.pnl for t in losses):+.2f}")
    print(f"Realized PnL $: {realized:+.2f}  -> on $2,000 = {realized/2000*100:+.2f}%")
    print(f"Still-open (would-be): {len(eod)}  -> {[t.symbol for t in eod]}")
    print(f"Equity incl. open MtM: {rep.final_capital:.2f}  ({rep.total_return*100:+.2f}%)")


if __name__ == "__main__":
    main()
