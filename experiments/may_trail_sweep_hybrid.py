#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hybrid trailing-stop sweep (May 2026): which TRAIL value survives reality?

The 4h model said wider trails hurt, but that was optimistic. This re-tests
TRAIL = 0.2/0.5/1.0/1.5/2.0 ATR using the realistic hybrid exit (5m scan +
per-day 1s zoom stepped at 30s), to find which trailing distance actually
performs best once intrabar wiggle is modelled. SL fixed at 1.5 ATR.

Data (5m/1s) is cached from the prior run, so this is fast.
Run:  python experiments/may_trail_sweep_hybrid.py
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
SL_ATR = 1.5
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000

_SEC_CACHE = {}
_FIVE_CACHE = {}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def _sec_day(symbol, day_str):
    key = (symbol, day_str)
    if key not in _SEC_CACHE:
        _SEC_CACHE[key] = HR.load_day(symbol, "1s", day_str)
    return _SEC_CACHE[key]


def _five(symbol, start, end):
    key = (symbol, start)
    if key not in _FIVE_CACHE:
        _FIVE_CACHE[key] = HR.load_range(symbol, "5m", start, end)
    return _FIVE_CACHE[key]


def hybrid_exit(symbol, entry_time, entry_price, atr, win_end_ms, trail_atr, act_atr):
    peak = entry_price
    sl = entry_price - SL_ATR * atr
    trailing = False
    start = entry_time + 1
    five = _five(symbol, start, win_end_ms)
    if five is None or not len(five):
        return None, None, "OPEN"
    margin = 1.0 * atr
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if not ((lo <= sl + margin) or (hi >= peak)):
            if hi > peak:
                peak = hi
                if peak - entry_price >= act_atr * atr:
                    trailing = True
                    sl = max(sl, peak - trail_atr * atr)
            continue
        day_str = pd.Timestamp(int(c["time"]), unit="ms").strftime("%Y-%m-%d")
        sec = _sec_day(symbol, day_str)
        if sec is not None and len(sec):
            seg = sec[(sec["time"] >= c["time"]) & (sec["time"] <= c["close_time"])]
            if len(seg):
                for _, s1 in seg.iloc[::30].iterrows():
                    p_hi, p_lo = float(s1["high"]), float(s1["low"])
                    if p_hi > peak:
                        peak = p_hi
                        if peak - entry_price >= act_atr * atr:
                            trailing = True
                            sl = max(sl, peak - trail_atr * atr)
                    if p_lo <= sl:
                        return int(s1["time"]), sl, ("TRAIL" if trailing else "SL")
                continue
        if lo <= sl:
            return int(c["close_time"]), sl, ("TRAIL" if trailing else "SL")
        if hi > peak:
            peak = hi
            if peak - entry_price >= act_atr * atr:
                trailing = True
                sl = max(sl, peak - trail_atr * atr)
    return None, None, "OPEN"


def run_trail(per_symbol, times, start, end, trail_atr):
    act_atr = trail_atr   # activation threshold tracks the trail distance (as in the bot)
    open_until = {}
    rows = []
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
            xt, xp, oc_ = hybrid_exit(sym, t, price, atr, end, trail_atr, act_atr)
            if oc_ == "OPEN":
                open_until[sym] = None
                rows.append(dict(net=0.0, pnl=0.0, outcome="OPEN"))
            else:
                net = (xp / price - 1) * 100 - FEE_RT
                rows.append(dict(net=net, pnl=POS_USD * net / 100.0, outcome=oc_))
                open_until[sym] = xt
    return rows


def metrics(rows):
    done = [r for r in rows if r["outcome"] != "OPEN"]
    wins = [r for r in done if r["net"] > 0]
    losses = [r for r in done if r["net"] <= 0]
    gw = sum(r["pnl"] for r in wins); gl = -sum(r["pnl"] for r in losses)
    pf = gw / gl if gl else float("inf")
    pnl = sum(r["pnl"] for r in done)
    aw = sum(r["net"] for r in wins) / len(wins) if wins else 0
    al = sum(r["net"] for r in losses) / len(losses) if losses else 0
    pp = aw / -al if al < 0 else float("inf")
    wr = len(wins) / len(done) * 100 if done else 0
    worst = min((r["net"] for r in done), default=0)
    return len(done), wr, pf, pp, aw, al, worst, pnl


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

    print("\nHYBRID trailing-stop sweep (May 2026, SL 1.5, 30s monitor)")
    print(f"{'TRAIL':<8}{'closed':>8}{'WR%':>7}{'PF':>7}{'PP':>7}"
          f"{'avgW%':>8}{'avgL%':>8}{'worst%':>8}{'return%':>9}")
    print("-" * 70)
    for trail in [0.2, 0.5, 1.0, 1.5, 2.0]:
        rows = run_trail(per_symbol, times, start, end, trail)
        n, wr, pf, pp, aw, al, worst, pnl = metrics(rows)
        print(f"{trail:<8}{n:>8}{wr:>6.1f}%{pf:>7.2f}{pp:>7.2f}"
              f"{aw:>7.2f}%{al:>7.2f}%{worst:>7.1f}%{pnl/2000*100:>8.2f}%")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
