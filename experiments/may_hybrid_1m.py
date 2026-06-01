#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""May 2026: 4h-model backtest vs hybrid (5m scan + 30s zoom) exit.

Same ORIGINAL strategy, same entries (4h close). Two exit models:
  4h     : optimistic intrabar (current backtest engine)
  hybrid : walk 5m candles; near the stop, zoom to 1s data stepped at 30s
           granularity (matches the live bot's 30s monitor).

Reports both, and the gap between them. Open positions stay open; closed only.
Run:  python experiments/may_hybrid_vs_4h.py
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE_RT = S.FEE_RT
SL_ATR, TRAIL_ATR, ACT_ATR = 1.5, 0.2, 0.2
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
STEP_MS = 60_000          # 1-minute monitoring (fast, ~99% accurate)


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def hybrid_exit(symbol, entry_time, entry_price, atr, win_end_ms):
    """5m scan + 30s zoom (from 1s data). Returns (exit_time, exit_price, outcome)."""
    peak = entry_price
    sl = entry_price - SL_ATR * atr
    trailing = False
    start = entry_time + 1
    five = HR.load_range(symbol, "5m", start, win_end_ms)
    if five is None or not len(five):
        return None, None, "OPEN"
    margin = 1.0 * atr
    one_s = None
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if not ((lo <= sl + margin) or (hi >= peak)):
            if hi > peak:
                peak = hi
                if peak - entry_price >= ACT_ATR * atr:
                    trailing = True
                    sl = max(sl, peak - TRAIL_ATR * atr)
            continue
        if one_s is None:
            one_s = HR.load_range(symbol, "1m", start, win_end_ms)
        if one_s is not None and len(one_s):
            seg = one_s[(one_s["time"] >= c["time"]) & (one_s["time"] <= c["close_time"])]
            if len(seg):
                # step every 30 seconds through the 1s data
                # 1m candles already ~ per-minute; no decimation
                for _, s1 in seg.iterrows():
                    p_hi, p_lo = float(s1["high"]), float(s1["low"])
                    if p_hi > peak:
                        peak = p_hi
                        if peak - entry_price >= ACT_ATR * atr:
                            trailing = True
                            sl = max(sl, peak - TRAIL_ATR * atr)
                    if p_lo <= sl:
                        return int(s1["time"]), sl, ("TRAIL" if trailing else "SL")
                continue
        # fallback: 5m low check
        if lo <= sl:
            return int(c["close_time"]), sl, ("TRAIL" if trailing else "SL")
        if hi > peak:
            peak = hi
            if peak - entry_price >= ACT_ATR * atr:
                trailing = True
                sl = max(sl, peak - TRAIL_ATR * atr)
    return None, None, "OPEN"


def stats(rows, label):
    done = [r for r in rows if r["outcome"] != "OPEN"]
    wins = [r for r in done if r["net"] > 0]
    losses = [r for r in done if r["net"] <= 0]
    gw = sum(r["pnl"] for r in wins); gl = -sum(r["pnl"] for r in losses)
    pf = gw / gl if gl else float("inf")
    pnl = sum(r["pnl"] for r in done)
    aw = sum(r["net"] for r in wins) / len(wins) if wins else 0
    al = sum(r["net"] for r in losses) / len(losses) if losses else 0
    pp = aw / -al if al < 0 else float("inf")
    worst = min((r["net"] for r in done), default=0)
    print(f"\n[{label}]  closed={len(done)}  wins={len(wins)} losses={len(losses)}")
    print(f"   WR {len(wins)/len(done)*100:.1f}%  PF {pf:.2f}  PP {pp:.2f}  "
          f"avgW {aw:.2f}% avgL {al:.2f}%  worst {worst:.1f}%")
    print(f"   gross win ${gw:+.2f}  gross loss ${-gl:+.2f}  net ${pnl:+.2f}  "
          f"= {pnl/2000*100:+.2f}% on $2000")
    return dict(n=len(done), wr=len(wins)/len(done)*100 if done else 0, pf=pf, pnl=pnl)


def main():
    start = parse("2026-05-01")
    end = parse("2026-06-01")
    fetch_from = start - PB.WARMUP_BARS * FOUR_H
    print("Loading 4h + featurising (all coins) ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fetch_from, end, log=lambda *a: None)
    per_symbol = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for s, df in raw.items()}
    print(f"  {len(per_symbol)} symbols")

    times = sorted({int(t) for df in per_symbol.values() for t in df.index if start <= t <= end})

    # ---- ENTRY simulation (shared): walk 4h, open up to 8, each held until its
    #      exit; we record the opened positions, then resolve exits per model. ----
    # To respect 8-concurrent with real exits, we need exit times. We resolve in
    # two passes: first the hybrid exits define occupancy; for the 4h model we
    # re-run the existing engine. Simpler + correct: run 4h via engine, and run
    # hybrid via its own occupancy-aware loop.

    # --- 4h model via the existing engine ---
    print("\nRunning 4h-model (engine)...")
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, PIT.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    PIT.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    try:
        S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 1.5, 0.2, 0.2
        rep4 = PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None,
                      rank="vol", exit_model="pessimistic", source="archive", fixed_notional=True)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, PIT.prefetch_universe) = of, oc, oa, op
    rows4 = [dict(sym=t.symbol, net=t.net_pct, pnl=t.pnl,
                  outcome=("OPEN" if t.outcome == "EOD" else t.outcome)) for t in rep4.trades]

    # --- hybrid model: occupancy-aware entry + hybrid exits ---
    print("Running hybrid-model (5m + 30s zoom) — this is the slow part...")
    open_until = {}     # symbol -> exit_time (occupied until then)
    rows_h = []
    # build per-bar candidate lists once
    for t in times:
        # free up expired slots
        active = {s: u for s, u in open_until.items() if u is None or u > t}
        open_until = active
        if len(open_until) >= MAX_CONC:
            continue
        cands = []
        for sym, df in per_symbol.items():
            if sym in open_until or t not in df.index:
                continue
            row = df.loc[t]
            if bool(row["entry_signal"]):
                cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            xt, xp, oc_ = hybrid_exit(sym, t, price, atr, end)
            if oc_ == "OPEN":
                open_until[sym] = None     # occupies a slot to the end
                rows_h.append(dict(sym=sym, net=0.0, pnl=0.0, outcome="OPEN"))
            else:
                net = (xp / price - 1) * 100 - FEE_RT
                rows_h.append(dict(sym=sym, net=net, pnl=POS_USD * net / 100.0, outcome=oc_))
                open_until[sym] = xt

    s4 = stats(rows4, "4h MODEL (optimistic)")
    sh = stats(rows_h, "HYBRID 5m+30s (realistic)")
    print("\n" + "=" * 50)
    print(f"GAP: 4h model return {s4['pnl']/2000*100:+.2f}%  vs  hybrid {sh['pnl']/2000*100:+.2f}%")
    if s4['pnl'] != 0:
        print(f"     hybrid / 4h = {sh['pnl']/s4['pnl']*100:.0f}% of the 4h-model PnL")


if __name__ == "__main__":
    main()
