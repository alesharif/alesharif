#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATR-range filter sweep (April+May 2026, hybrid TRAIL 1.5).

Compares upper-only caps vs golden-range (min+max) ATR/price filters:
  <=10%, <=15%            (cap only)
  5-10%, 5-15%            (user's requested ranges)
  6-12%, 4-12%            (extra ranges to probe the golden band)
Reports per month: trades, WR, PF, worst, return.

Run:  python experiments/atr_range_sweep.py
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
SL_ATR = 1.5
TRAIL = 1.5
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
_SEC = {}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def hybrid_exit(symbol, et, ep, atr, end):
    peak = ep; sl = ep - SL_ATR * atr; trailing = False
    five = HR.load_range(symbol, "5m", et + 1, end)
    if five is None or not len(five):
        return None, None, "OPEN"
    margin = 1.0 * atr
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if not ((lo <= sl + margin) or (hi >= peak)):
            if hi > peak:
                peak = hi
                if peak - ep >= TRAIL * atr:
                    trailing = True; sl = max(sl, peak - TRAIL * atr)
            continue
        day = pd.Timestamp(int(c["time"]), unit="ms").strftime("%Y-%m-%d")
        s = sec_day(symbol, day)
        if s is not None and len(s):
            seg = s[(s["time"] >= c["time"]) & (s["time"] <= c["close_time"])]
            if len(seg):
                for _, s1 in seg.iloc[::30].iterrows():
                    ph, pl = float(s1["high"]), float(s1["low"])
                    if ph > peak:
                        peak = ph
                        if peak - ep >= TRAIL * atr:
                            trailing = True; sl = max(sl, peak - TRAIL * atr)
                    if pl <= sl:
                        return int(s1["time"]), sl, ("TRAIL" if trailing else "SL")
                continue
        if lo <= sl:
            return int(c["close_time"]), sl, ("TRAIL" if trailing else "SL")
        if hi > peak:
            peak = hi
            if peak - ep >= TRAIL * atr:
                trailing = True; sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, lo_ar, hi_ar):
    open_until = {}; rows = []
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
        if len(open_until) >= MAX_CONC:
            continue
        cands = []
        for sym, df in ps.items():
            if sym in open_until or t not in df.index:
                continue
            row = df.loc[t]
            if not bool(row["entry_signal"]):
                continue
            ar = float(row["atr"]) / float(row["close"])
            if lo_ar > 0 and ar < lo_ar:
                continue
            if hi_ar > 0 and ar > hi_ar:
                continue
            cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            xt, xp, oc = hybrid_exit(sym, t, price, atr, end)
            if oc == "OPEN":
                open_until[sym] = None
            else:
                net = (xp / price - 1) * 100 - FEE
                rows.append(net)
                open_until[sym] = xt
    return rows


def stats(nets):
    wins = [n for n in nets if n > 0]; losses = [n for n in nets if n <= 0]
    gw = sum(wins); gl = -sum(losses)
    pf = gw / gl if gl else float("inf")
    wr = len(wins) / len(nets) * 100 if nets else 0
    ret = sum(POS_USD * n / 100 for n in nets) / 2000 * 100
    worst = min(nets, default=0)
    return len(nets), wr, pf, worst, ret


def load(s_str, e_str):
    start, end = parse(s_str), parse(e_str)
    ff = start - PB.WARMUP_BARS * FOUR_H
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for s, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    return ps, times, end


def main():
    months = {"April": ("2026-04-01", "2026-05-01"), "May": ("2026-05-01", "2026-06-01")}
    base = {n: load(s, e) for n, (s, e) in months.items()}

    # (label, min, max)
    filters = [
        ("cap <=10%", 0.0, 0.10),
        ("cap <=15%", 0.0, 0.15),
        ("range 5-10%", 0.05, 0.10),
        ("range 5-15%", 0.05, 0.15),
        ("range 6-12%", 0.06, 0.12),
        ("range 4-12%", 0.04, 0.12),
    ]
    print("ATR-range filter sweep (hybrid TRAIL 1.5)")
    print(f"{'filter':<14}{'month':<8}{'trades':>8}{'WR':>7}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 62)
    for label, lo, hi in filters:
        agg = []
        for name in months:
            ps, times, end = base[name]
            nets = simulate(ps, times, end, lo, hi)
            agg += nets
            n, wr, pf, worst, ret = stats(nets)
            print(f"{label:<14}{name:<8}{n:>8}{wr:>6.0f}%{pf:>7.2f}{worst:>8.1f}%{ret:>8.2f}%")
            sys.stdout.flush()
        n, wr, pf, worst, ret = stats(agg)
        print(f"{label:<14}{'BOTH':<8}{n:>8}{wr:>6.0f}%{pf:>7.2f}{worst:>8.1f}%{ret:>8.2f}%")
        print("-" * 62)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
