#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAST ATR cap sweep across four months (hybrid TRAIL 1.5).

Speed trick: the hybrid exit of a given (symbol, entry_bar) is INDEPENDENT of
the ATR cap (the cap only decides whether the trade is eligible). So we:
  1) for every entry signal in the month, compute its hybrid exit ONCE,
     caching (entry_time, exit_time, net, atr_ratio, vol).
  2) for each cap, cheaply re-simulate the 8-slot occupancy using the cached
     exits, skipping signals whose atr_ratio exceeds the cap.

This computes the heavy hybrid exits ~4x (once per month) instead of 20x.

Run:  python experiments/atr_cap_fast.py
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


def precompute(ps, times, end):
    """Compute hybrid exit ONCE for every entry signal. Returns list of dicts.

    Frees the 1s cache periodically so memory stays bounded on busy months.
    """
    signals = []
    seen = set()   # (symbol, entry_time) dedup
    for t in times:
        for sym, df in ps.items():
            if t not in df.index:
                continue
            row = df.loc[t]
            if not bool(row["entry_signal"]):
                continue
            key = (sym, t)
            if key in seen:
                continue
            seen.add(key)
            price = float(row["close"]); atr = float(row["atr"])
            ar = atr / price; vol = float(row["vol_pit"])
            signals.append(dict(sym=sym, t=t, price=price, atr=atr, ar=ar, vol=vol))
    # compute exits (heavy) once; flush the 1s cache every N to bound memory
    import gc
    for i, s in enumerate(signals):
        xt, xp, oc = hybrid_exit(s["sym"], s["t"], s["price"], s["atr"], end)
        if oc == "OPEN":
            s["exit_t"] = None; s["net"] = None
        else:
            s["exit_t"] = xt
            s["net"] = (xp / s["price"] - 1) * 100 - FEE
        if (i + 1) % 200 == 0:
            _SEC.clear()
            gc.collect()
            print(f"    ...{i+1}/{len(signals)} exits (mem flushed)", flush=True)
    return signals


def simulate_cap(signals, times, cap):
    """Cheap 8-slot occupancy re-sim using precomputed exits, filtered by cap."""
    # index signals by entry bar
    by_t = {}
    for s in signals:
        if cap > 0 and s["ar"] > cap:
            continue
        by_t.setdefault(s["t"], []).append(s)
    open_until = {}
    nets = []
    for t in times:
        open_until = {sym: u for sym, u in open_until.items() if u is None or u > t}
        if len(open_until) >= MAX_CONC:
            continue
        cands = [s for s in by_t.get(t, []) if s["sym"] not in open_until]
        cands.sort(key=lambda x: x["vol"], reverse=True)
        for s in cands[:MAX_CONC - len(open_until)]:
            if s["exit_t"] is None:
                open_until[s["sym"]] = None
            else:
                nets.append(s["net"])
                open_until[s["sym"]] = s["exit_t"]
    return nets


def stats(nets):
    wins = [n for n in nets if n > 0]; losses = [n for n in nets if n <= 0]
    gw = sum(wins); gl = -sum(losses)
    pf = gw / gl if gl else float("inf")
    wr = len(wins) / len(nets) * 100 if nets else 0
    ret = sum(POS_USD * n / 100 for n in nets) / 2000 * 100
    return len(nets), wr, pf, min(nets, default=0), ret


def load(s_str, e_str):
    start, end = parse(s_str), parse(e_str)
    ff = start - PB.WARMUP_BARS * FOUR_H
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for s, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    return ps, times, end


def main():
    months = {
        "2025-04": ("2025-04-01", "2025-05-01"),
        "2025-12": ("2025-12-01", "2026-01-01"),
        "2026-04": ("2026-04-01", "2026-05-01"),
        "2026-05": ("2026-05-01", "2026-06-01"),
    }
    pre = {}
    for name, (s, e) in months.items():
        print(f"Precomputing {name} (hybrid exits once) ...", flush=True)
        ps, times, end = load(s, e)
        sigs = precompute(ps, times, end)
        # keep only the light fields needed later; drop heavy frames
        pre[name] = (sigs, times)
        print(f"  {name}: {len(sigs)} signals", flush=True)
        # free per-month memory: 1s cache + featurised frames
        _SEC.clear()
        del ps
        import gc
        gc.collect()
    print()

    caps = [0.08, 0.09, 0.10, 0.12, 0.15]
    print("FAST ATR cap sweep, FOUR months (hybrid TRAIL 1.5)")
    print(f"{'cap':<7}{'month':<9}{'trades':>8}{'WR':>7}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 56)
    for cap in caps:
        agg = []
        for name in months:
            signals, times = pre[name]
            nets = simulate_cap(signals, times, cap)
            agg += nets
            n, wr, pf, worst, ret = stats(nets)
            print(f"{cap*100:.0f}%{'':<4}{name:<9}{n:>8}{wr:>6.0f}%{pf:>7.2f}{worst:>8.1f}%{ret:>8.2f}%")
        n, wr, pf, worst, ret = stats(agg)
        print(f"{cap*100:.0f}%{'':<4}{'ALL':<9}{n:>8}{wr:>6.0f}%{pf:>7.2f}{worst:>8.1f}%{ret:>8.2f}%")
        print("-" * 56)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
