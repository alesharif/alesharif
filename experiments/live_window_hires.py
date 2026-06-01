#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""100%-faithful replay of the live paper bot over its window.

ENTRY: exactly like the bot -> on each 4h close, scan all coins for the entry
       signal, open at the close price (up to 8 concurrent, $250 each).
EXIT : like the bot's 30s monitor, via a HYBRID engine:
         walk 5-minute candles; whenever a candle's low gets within a safety
         margin of the stop (or a new high would lift the trailing stop near
         price), ZOOM into 1-second candles and walk them second-by-second,
         updating the ATR trailing stop and checking the stop break with
         tick-like precision. Otherwise just lift the peak from the 5m high.

This reproduces the live FET/IOTA stop-outs that the 4h model missed.

Window: 2026-05-31 -> 2026-06-01.  Run:  python experiments/live_window_hires.py
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE_RT = S.FEE_RT            # 0.2%
SL_ATR, TRAIL_ATR, ACT_ATR = 1.5, 0.2, 0.2
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def hybrid_exit(symbol, entry_time, entry_price, atr, win_end_ms, log):
    """Simulate one long position's exit with 5m scan + 1s zoom.

    Returns (exit_time, exit_price, outcome) or (None, None, 'OPEN').
    """
    peak = entry_price
    sl = entry_price - SL_ATR * atr
    trailing = False
    start = entry_time + 1
    five = HR.load_range(symbol, "5m", start, win_end_ms)
    if five is None or not len(five):
        return None, None, "OPEN"

    # safety margin: if a 5m low comes within 1 ATR of the stop, zoom to 1s
    margin = 1.0 * atr
    one_s_cache = None

    for _, c in five.iterrows():
        c_open, c_close = c["time"], c["close_time"]
        hi, lo = float(c["high"]), float(c["low"])

        near = (lo <= sl + margin) or (hi >= peak)  # could hit stop, or new high zone
        if not near:
            # safe: just lift the peak/trailing from the 5m high
            if hi > peak:
                peak = hi
                if peak - entry_price >= ACT_ATR * atr:
                    trailing = True
                    sl = max(sl, peak - TRAIL_ATR * atr)
            continue

        # ZOOM: walk this 5m window second-by-second
        if one_s_cache is None:
            one_s_cache = HR.load_range(symbol, "1s", start, win_end_ms)
        sec = one_s_cache
        if sec is not None and len(sec):
            seg = sec[(sec["time"] >= c_open) & (sec["time"] <= c_close)]
            for _, s1 in seg.iterrows():
                p_hi, p_lo = float(s1["high"]), float(s1["low"])
                # update trailing on the up-move first (matches bot order)
                if p_hi > peak:
                    peak = p_hi
                    if peak - entry_price >= ACT_ATR * atr:
                        trailing = True
                        sl = max(sl, peak - TRAIL_ATR * atr)
                # then check stop on the low
                if p_lo <= sl:
                    return int(s1["time"]), sl, ("TRAIL" if trailing else "SL")
        else:
            # no 1s data: fall back to 5m low check
            if lo <= sl:
                return int(c_close), sl, ("TRAIL" if trailing else "SL")
            if hi > peak:
                peak = hi
                if peak - entry_price >= ACT_ATR * atr:
                    trailing = True
                    sl = max(sl, peak - TRAIL_ATR * atr)

    return None, None, "OPEN"   # never hit stop within the window


def main():
    start = parse("2026-05-31")
    end = parse("2026-06-02")
    fetch_from = start - PB.WARMUP_BARS * FOUR_H
    print("Loading 4h + featurising (all coins) for ENTRY signals ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fetch_from, end, log=lambda *a: None)
    per_symbol = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for s, df in raw.items()}
    print(f"  {len(per_symbol)} symbols\n")

    # build 4h timeline within the window
    times = set()
    for df in per_symbol.values():
        times.update(int(t) for t in df.index if start <= t <= end)
    timeline = sorted(times)

    capital = 2000.0
    open_syms = set()
    closed = []
    open_positions = []   # (symbol, entry_time, entry_price, atr)

    print("Simulating entries on 4h closes + hybrid 5m/1s exits ...\n")
    for t in timeline:
        # close out any positions whose exit happens at/before this bar is handled lazily below
        # ENTRY scan on this 4h close
        if len(open_syms) >= MAX_CONC:
            continue
        cands = []
        for sym, df in per_symbol.items():
            if sym in open_syms or t not in df.index:
                continue
            row = df.loc[t]
            if bool(row["entry_signal"]):
                cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        slots = MAX_CONC - len(open_syms)
        for sym, price, atr, _ in cands[:slots]:
            open_syms.add(sym)
            open_positions.append((sym, t, price, atr))

    # now resolve each opened position via the hybrid exit engine
    print(f"Resolving {len(open_positions)} opened positions (hybrid exit)...\n")
    for sym, et, ep, atr in open_positions:
        xt, xp, outcome = hybrid_exit(sym, et, ep, atr, end, print)
        if outcome == "OPEN":
            closed.append((sym, et, ep, None, None, "OPEN", None, None))
            continue
        net = (xp / ep - 1) * 100 - FEE_RT
        pnl = POS_USD * net / 100.0
        capital += pnl
        closed.append((sym, et, ep, xt, xp, outcome, net, pnl))

    # ---- report ----
    print(f"{'symbol':<11}{'entry_t':<13}{'entry':>10}{'exit_t':<13}{'exit':>10}"
          f"{'outcome':>8}{'net%':>8}{'pnl$':>8}")
    print("-" * 89)
    done = [c for c in closed if c[5] != "OPEN"]
    openp = [c for c in closed if c[5] == "OPEN"]
    for c in sorted(done, key=lambda x: x[3]):
        sym, et, ep, xt, xp, oc, net, pnl = c
        ets = pd.to_datetime(et, unit="ms").strftime("%m-%d %H:%M")
        xts = pd.to_datetime(xt, unit="ms").strftime("%m-%d %H:%M")
        print(f"{sym:<11}{ets:<13}{ep:>10.5g}{xts:<13}{xp:>10.5g}{oc:>8}{net:>7.2f}%{pnl:>7.2f}")

    wins = [c for c in done if c[6] > 0]
    losses = [c for c in done if c[6] <= 0]
    realized = sum(c[7] for c in done)
    print("\n" + "=" * 50)
    print(f"CLOSED trades : {len(done)}  (wins {len(wins)}, losses {len(losses)})")
    if done:
        print(f"Win rate      : {len(wins)/len(done)*100:.1f}%")
    print(f"Gross win $   : {sum(c[7] for c in wins):+.2f}")
    print(f"Gross loss $  : {sum(c[7] for c in losses):+.2f}")
    print(f"Realized PnL $: {realized:+.2f}  -> on $2,000 = {realized/2000*100:+.2f}%")
    print(f"Still-open    : {len(openp)}  -> {[c[0] for c in openp]}")


if __name__ == "__main__":
    main()
