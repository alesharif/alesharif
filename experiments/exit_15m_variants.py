#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""15m-frame EXIT shootout — give the 15m exit a thorough, fair test.

The user wants to exit on the 15m frame to capture closer to the peak. Prior
EMA15_20 whipsawed (PF 1.09 vs EMA60_20's 1.17). Here we test MANY 15m-frame
exits on the base entries (4h signal + fear 1.15 + atr>=5%+adx>=40), all with a
2*ATR hard stop, on the real 1m path (resampled to 15m):

  EMA60_20    : benchmark (1h close < EMA20)
  EMA15_20    : 15m close < EMA20
  EMA15_30    : 15m close < EMA30
  EMA15_50    : 15m close < EMA50  (~12.5h, slower 15m)
  EMA15_20_c2 : 15m close < EMA20 for 2 consecutive bars (anti-whipsaw)
  EMA15_50_c2 : 15m close < EMA50 for 2 consecutive bars

Run:  python experiments/exit_15m_variants.py
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

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def resample(tmin, hi, lo, cl, tf_min):
    n = len(cl); grp = (tmin - tmin[0]) // (tf_min * 60 * 1000)
    H = []; L = []; C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            H.append(hi[m].max()); L.append(lo[m].min()); C.append(cl[m][-1]); T.append(tmin[m][-1])
    return np.array(H), np.array(L), np.array(C), np.array(T)


def exit_ema(tmin, hi, lo, cl, ep, atr, tf_min, span, confirm=1):
    """Hard 2ATR stop (1m) + exit when tf close<EMA(span) for `confirm` bars."""
    sl = ep - HARD_SL_ATR * atr
    sl_i = None
    for i in range(len(lo)):
        if lo[i] <= sl:
            sl_i = i; break
    H, L, C, T = resample(tmin, hi, lo, cl, tf_min)
    et = epx = None
    if len(C) >= span + confirm + 1:
        e = ema(C, span); run = 0
        for j in range(span + 1, len(C)):
            if C[j] < e[j]:
                run += 1
                if run >= confirm:
                    et = T[j]; epx = C[j]; break
            else:
                run = 0
    sl_t = tmin[sl_i] if sl_i is not None else None
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_i]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


VARIANTS = [("EMA60_20", 60, 20, 1), ("EMA15_20", 15, 20, 1), ("EMA15_30", 15, 30, 1),
            ("EMA15_50", 15, 50, 1), ("EMA15_20_c2", 15, 20, 2), ("EMA15_50_c2", 15, 50, 2)]


def main():
    acc = {v[0]: {m: [] for m in MONTHS} for v in VARIANTS}
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
                if d is None or len(d) < 60:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                for lbl, tf, sp, cf in VARIANTS:
                    acc[lbl][mname].append(exit_ema(tmin, hi, lo, cl, price, atr, tf, sp, cf))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### 15m-FRAME EXIT SHOOTOUT (base entries) #####")
    print(f"{'exit':<14}{'trades':>7}{'WR':>6}{'PF':>7}{'avg%':>8}{'ret_ALL':>9}   per-month")
    print("-" * 82)
    for lbl, *_ in VARIANTS:
        alln = [x for m in MONTHS for x in acc[lbl][m]]
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        pm = " ".join(f"{m[2:]}:{np.array(acc[lbl][m]).sum()*0.125:+.0f}" for m in MONTHS)
        print(f"{lbl:<14}{len(a):>7}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{a.mean():>7.2f}%"
              f"{a.sum()*0.125:>8.1f}%   {pm}", flush=True)
    print("\nDONE_15M_EXIT.", flush=True)


if __name__ == "__main__":
    main()
