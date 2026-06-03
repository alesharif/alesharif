#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAUSAL adaptive exit by real-time pump SPEED — capture the oracle headroom.

The shape test showed: FAST pumps want the 15m exit, SLOW grinds want the 1h
exit (corr with t_half = -0.324; oracle +168% vs fixed +30%). t_half is
look-ahead. Here we use a CAUSAL real-time speed proxy: the 1h velocity.

At each 15m bar: vel_1h = % move over the last 4 bars (1h).
  vel_1h >= V  -> FAST regime  -> exit when 15m close < EMA20(15m)  (responsive)
  vel_1h <  V  -> SLOW regime  -> exit when 15m close < EMA80(15m)  (~1h EMA20)
Plus a 2*ATR hard stop (checked on 1m). Fully causal -> implementable live.

Compared to fixed-1h (+30%), fixed-15m (+9%), and the look-ahead oracle (+168%).
Tests several speed thresholds V. Run:  python experiments/causal_adaptive_exit.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
MAX_CONC = 8
FOUR = 4 * 3600 * 1000
HOLD_H = 48
HARD_SL_ATR = 2.0
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0
VTHRESHS = [2.0, 3.0, 4.0, 6.0]   # 1h-velocity % thresholds for "fast"

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def resample(tmin, hi, lo, cl, tf_min):
    grp = (tmin - tmin[0]) // (tf_min * 60 * 1000)
    H = []; L = []; C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            H.append(hi[m].max()); L.append(lo[m].min()); C.append(cl[m][-1]); T.append(tmin[m][-1])
    return np.array(H), np.array(L), np.array(C), np.array(T)


def exit_fixed(tmin, hi, lo, cl, ep, atr, tf_min, span):
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    H, L, C, T = resample(tmin, hi, lo, cl, tf_min)
    et = epx = None
    if len(C) >= span + 2:
        e = ema(C, span)
        for j in range(span + 1, len(C)):
            if C[j] < e[j]:
                et = T[j]; epx = C[j]; break
    sl_t = tmin[sl_i] if sl_i is not None else None
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_i]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def exit_adaptive(tmin, hi, lo, cl, ep, atr, V):
    """Causal: per 15m bar pick fast(EMA20) or slow(EMA80) exit by 1h velocity."""
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    sl_t = tmin[sl_i] if sl_i is not None else None
    H, L, C, T = resample(tmin, hi, lo, cl, 15)
    if len(C) < 82:
        return exit_fixed(tmin, hi, lo, cl, ep, atr, 60, 20)
    e20 = ema(C, 20); e80 = ema(C, 80)
    et = epx = None
    for j in range(81, len(C)):
        vel = (C[j] / C[j - 4] - 1) * 100 if j >= 4 else 0.0
        fast = vel >= V
        brk = (C[j] < e20[j]) if fast else (C[j] < e80[j])
        if brk:
            et = T[j]; epx = C[j]; break
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_i]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def main():
    cols = ["FIXED_1h", "FIXED_15m"] + [f"ADAPT_V{int(v)}" for v in VTHRESHS]
    acc = {c: {m: [] for m in MONTHS} for c in cols}
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e); ff = start - PB.WARMUP_BARS * FOUR
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        sr = PB.build_stable_ratio(ff, end)
        sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
        open_until = {}
        for t in times:
            open_until = {sy: u for sy, u in open_until.items() if u > t}
            if sblock.get(t, False) or len(open_until) >= MAX_CONC:
                continue
            cands = []
            for sym, df in ps.items():
                if sym in open_until or t not in df.index:
                    continue
                row = df.loc[t]
                if not bool(row["entry_signal"]):
                    continue
                price = float(row["close"]); atr = float(row["atr"]); adx = float(row["adx"])
                if atr / price < ATR_MIN or adx < ADX_MIN:
                    continue
                cands.append((sym, price, atr, float(row["vol_pit"])))
            cands.sort(key=lambda x: x[3], reverse=True)
            for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 90:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                acc["FIXED_1h"][mname].append(exit_fixed(tmin, hi, lo, cl, price, atr, 60, 20))
                acc["FIXED_15m"][mname].append(exit_fixed(tmin, hi, lo, cl, price, atr, 15, 20))
                for v in VTHRESHS:
                    acc[f"ADAPT_V{int(v)}"][mname].append(exit_adaptive(tmin, hi, lo, cl, price, atr, v))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### CAUSAL ADAPTIVE EXIT (base entries) #####")
    print(f"{'exit':<12}{'trades':>7}{'WR':>6}{'PF':>7}{'avg%':>8}{'ret_ALL':>9}   per-month")
    print("-" * 82)
    for c in cols:
        alln = [x for m in MONTHS for x in acc[c][m]]
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        pm = " ".join(f"{m[2:]}:{np.array(acc[c][m]).sum()*0.125:+.0f}" for m in MONTHS)
        print(f"{c:<12}{len(a):>7}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{a.mean():>7.2f}%"
              f"{a.sum()*0.125:>8.1f}%   {pm}", flush=True)
    print("\n(fixed 1h ~+30% baseline; oracle was +168% — how much do we capture causally?)")
    print("\nDONE_CAUSAL_ADAPT.", flush=True)


if __name__ == "__main__":
    main()
