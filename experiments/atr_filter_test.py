#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATR/price volatility filter: analysis + hybrid sweep (April+May 2026).

PART A: performance of trades bucketed by entry ATR/price ratio (finds where
        extreme-volatility entries destroy results -> the STOUSDT -77% case).
PART B: hybrid TRAIL=1.5 with a max ATR/price filter at several thresholds,
        on April and May separately, to pick a threshold that removes the
        toxic entries without cutting good ones.

Run:  python experiments/atr_filter_test.py
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


def simulate(ps, times, end, max_ar):
    """Hybrid TRAIL=1.5 with optional max ATR/price filter. Returns trade rows."""
    open_until = {}
    rows = []
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
            if max_ar > 0 and ar > max_ar:
                continue
            cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"]), ar))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _, ar in cands[:MAX_CONC - len(open_until)]:
            xt, xp, oc = hybrid_exit(sym, t, price, atr, end)
            if oc == "OPEN":
                open_until[sym] = None
            else:
                net = (xp / price - 1) * 100 - FEE
                rows.append((ar, net, oc))
                open_until[sym] = xt
    return rows


def stats(rows):
    nets = [r[1] for r in rows]
    wins = [n for n in nets if n > 0]; losses = [n for n in nets if n <= 0]
    gw = sum(wins); gl = -sum(losses)
    pf = gw / gl if gl else float("inf")
    wr = len(wins) / len(nets) * 100 if nets else 0
    pnl_pct = sum(POS_USD * n / 100 for n in nets) / 2000 * 100
    worst = min(nets, default=0)
    return len(nets), wr, pf, worst, pnl_pct


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
    base = {}
    for name, (s, e) in months.items():
        print(f"Loading {name} ...")
        base[name] = load(s, e)

    # PART A: bucket by ATR ratio (no filter)
    print("\nPART A — performance by entry ATR/price (no filter), TRAIL 1.5")
    print(f"{'ATR/price':<12}{'trades':>8}{'WR':>7}{'avgNet':>9}{'worst':>9}")
    print("-" * 46)
    allrows = []
    for name in months:
        ps, times, end = base[name]
        allrows += simulate(ps, times, end, 0.0)
    arr = pd.DataFrame(allrows, columns=["ar", "net", "oc"])
    for lo, hi in [(0, .03), (.03, .05), (.05, .08), (.08, .10), (.10, .15), (.15, 9)]:
        sub = arr[(arr.ar >= lo) & (arr.ar < hi)]
        if len(sub):
            print(f"{lo*100:.0f}-{hi*100 if hi<1 else 99:.0f}%{'':<6}{len(sub):>8}"
                  f"{(sub.net>0).mean()*100:>6.0f}%{sub.net.mean():>8.2f}%{sub.net.min():>8.1f}%")

    # PART B: filter-threshold sweep per month
    print("\nPART B — max ATR/price filter (hybrid TRAIL 1.5)")
    print(f"{'threshold':<12}{'month':<8}{'trades':>8}{'WR':>7}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 60)
    for thr in [0.0, 0.15, 0.10, 0.08, 0.05]:
        for name in months:
            ps, times, end = base[name]
            rows = simulate(ps, times, end, thr)
            n, wr, pf, worst, ret = stats(rows)
            lbl = "OFF" if thr == 0 else f"{thr*100:.0f}%"
            print(f"{lbl:<12}{name:<8}{n:>8}{wr:>6.0f}%{pf:>7.2f}{worst:>8.1f}%{ret:>8.2f}%")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
