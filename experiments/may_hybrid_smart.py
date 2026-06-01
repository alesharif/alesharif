#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""May 2026: 4h-model vs SMART hybrid exit (5m scan, lazy 1s zoom per-day).

Key fix: 1-second data is fetched ONLY for the specific day of the 5m candle
that approaches the stop/target — never the whole month up front. load_day
caches each day once, so repeated zooms on the same day are free.

ENTRY: exactly like the bot (4h close, 8 concurrent, $250 each).
EXIT : walk 5m candles; when a candle's low nears the stop OR makes a new high
       near price, zoom into that day's 1s candles (stepped at 30s) for a
       tick-accurate trailing-stop check.

Run:  python experiments/may_hybrid_smart.py
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
DAY_MS = 24 * 3600 * 1000


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


# cache of {(symbol, 'YYYY-MM-DD'): 1s dataframe} — lazy, per day
_SEC_CACHE = {}


def _sec_day(symbol, day_str):
    key = (symbol, day_str)
    if key not in _SEC_CACHE:
        _SEC_CACHE[key] = HR.load_day(symbol, "1s", day_str)
    return _SEC_CACHE[key]


def hybrid_exit(symbol, entry_time, entry_price, atr, win_end_ms):
    """5m scan + lazy per-day 1s zoom (stepped 30s). Returns (xt, xp, outcome)."""
    peak = entry_price
    sl = entry_price - SL_ATR * atr
    trailing = False
    start = entry_time + 1
    five = HR.load_range(symbol, "5m", start, win_end_ms)
    if five is None or not len(five):
        return None, None, "OPEN"
    margin = 1.0 * atr
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if not ((lo <= sl + margin) or (hi >= peak)):
            if hi > peak:                       # safe candle: lift peak from 5m high
                peak = hi
                if peak - entry_price >= ACT_ATR * atr:
                    trailing = True
                    sl = max(sl, peak - TRAIL_ATR * atr)
            continue
        # ZOOM: load ONLY this candle's day at 1s, step every 30s
        day_str = pd.Timestamp(int(c["time"]), unit="ms").strftime("%Y-%m-%d")
        sec = _sec_day(symbol, day_str)
        if sec is not None and len(sec):
            seg = sec[(sec["time"] >= c["time"]) & (sec["time"] <= c["close_time"])]
            if len(seg):
                for _, s1 in seg.iloc[::30].iterrows():   # 30-second steps
                    p_hi, p_lo = float(s1["high"]), float(s1["low"])
                    if p_hi > peak:
                        peak = p_hi
                        if peak - entry_price >= ACT_ATR * atr:
                            trailing = True
                            sl = max(sl, peak - TRAIL_ATR * atr)
                    if p_lo <= sl:
                        return int(s1["time"]), sl, ("TRAIL" if trailing else "SL")
                continue
        # fallback if no 1s data for that day
        if lo <= sl:
            return int(c["close_time"]), sl, ("TRAIL" if trailing else "SL")
        if hi > peak:
            peak = hi
            if peak - entry_price >= ACT_ATR * atr:
                trailing = True
                sl = max(sl, peak - TRAIL_ATR * atr)
    return None, None, "OPEN"


def report(rows, label):
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
          f"avgW {aw:.2f}%  avgL {al:.2f}%  worst {worst:.1f}%")
    print(f"   net ${pnl:+.2f} = {pnl/2000*100:+.2f}% on $2000")
    return pnl


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

    # --- 4h model (engine) ---
    print("Running 4h-model ...")
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
    rows4 = [dict(net=t.net_pct, pnl=t.pnl,
                  outcome=("OPEN" if t.outcome == "EOD" else t.outcome)) for t in rep4.trades]

    # --- hybrid model (occupancy-aware, smart lazy zoom) ---
    print("Running SMART hybrid (5m scan, lazy per-day 1s zoom @30s) ...")
    open_until = {}
    rows_h = []
    processed = 0
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
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
            processed += 1
            if processed % 25 == 0:
                print(f"   ...resolved {processed} positions")
            if oc_ == "OPEN":
                open_until[sym] = None
                rows_h.append(dict(net=0.0, pnl=0.0, outcome="OPEN"))
            else:
                net = (xp / price - 1) * 100 - FEE_RT
                rows_h.append(dict(net=net, pnl=POS_USD * net / 100.0, outcome=oc_))
                open_until[sym] = xt

    p4 = report(rows4, "4h MODEL (optimistic)")
    ph = report(rows_h, "HYBRID 5m+30s (realistic)")
    print("\n" + "=" * 52)
    print(f"GAP: 4h {p4/2000*100:+.2f}%   vs   hybrid {ph/2000*100:+.2f}%")
    if p4:
        print(f"     hybrid is {ph/p4*100:.0f}% of the 4h-model PnL")


if __name__ == "__main__":
    main()
