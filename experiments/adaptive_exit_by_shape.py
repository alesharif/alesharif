#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADAPTIVE exit by pump character — match the exit timeframe to the move.

User's insight: pumps differ (single-spike vs vertical vs slow grind), driven by
traders on different timeframes; the exit frame should match. We test whether a
CAUSAL classifier tells us which exit (fast 15m vs slow 1h) to use per trade.

Per base trade (4h signal + fear1.15 + atr>=5%+adx>=40) on the real 1m path:
  net_15m  : EMA15_20 exit (fast)        net_1h : EMA60_20 exit (slow)
  classifiers known AT/BEFORE entry: atr_ratio, adx
  pump-speed (for oracle analysis): hours to reach half of the peak (t_half)
We report:
  - which fixed exit wins (baseline)
  - corr( net_15m - net_1h , classifier )  -> does any feature pick the exit?
  - ADAPTIVE split (fast if classifier high else slow) vs fixed
  - ORACLE (pick the better exit per trade) -> the upper bound if classification
    were perfect (tells us if there is anything to gain at all)
Run:  python experiments/adaptive_exit_by_shape.py
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
    grp = (tmin - tmin[0]) // (tf_min * 60 * 1000)
    H = []; L = []; C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            H.append(hi[m].max()); L.append(lo[m].min()); C.append(cl[m][-1]); T.append(tmin[m][-1])
    return np.array(H), np.array(L), np.array(C), np.array(T)


def exit_ema(tmin, hi, lo, cl, ep, atr, tf_min, span):
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


def main():
    rows = []   # (net15, net1h, atr_ratio, adx, t_half, month)
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
                cands.append((sym, price, atr, adx, float(row["vol_pit"])))
            cands.sort(key=lambda x: x[4], reverse=True)
            for sym, price, atr, adx, _ in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 60:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                n15 = exit_ema(tmin, hi, lo, cl, price, atr, 15, 20)
                n1h = exit_ema(tmin, hi, lo, cl, price, atr, 60, 20)
                # pump speed: hours to reach half of the peak
                pk = int(np.argmax(hi)); peak = hi[pk]
                half = price + 0.5 * (peak - price)
                hi_idx = np.where(hi >= half)[0]
                t_half = (hi_idx[0] / 60.0) if len(hi_idx) else 48.0   # hours
                rows.append((n15, n1h, atr / price, adx, t_half, mname))
        print(f"  {mname} done ({len(rows)} trades)", flush=True)
        del ps, raw; gc.collect()

    n15 = np.array([r[0] for r in rows]); n1h = np.array([r[1] for r in rows])
    ar = np.array([r[2] for r in rows]); ad = np.array([r[3] for r in rows])
    th = np.array([r[4] for r in rows]); mon = [r[5] for r in rows]
    diff = n15 - n1h
    def tot(a): return a.sum() * 0.125
    print(f"\n##### ADAPTIVE EXIT BY SHAPE ({len(rows)} trades) #####")
    print(f"fixed 15m:  total {tot(n15):+.1f}%   PF {n15[n15>0].sum()/-n15[n15<=0].sum():.2f}")
    print(f"fixed 1h :  total {tot(n1h):+.1f}%   PF {n1h[n1h>0].sum()/-n1h[n1h<=0].sum():.2f}")
    print(f"\ncorr(net15-net1h , feature) — does a feature pick the right exit?")
    for nm, v in [("atr_ratio", ar), ("adx", ad), ("t_half(speed)", th)]:
        print(f"  {nm:<14}: {np.corrcoef(v, diff)[0,1]:+.3f}")
    # adaptive: fast(15m) when t_half small (fast pump) else slow(1h) — ORACLE-ish via speed
    for thr in (1, 2, 4, 8):
        pick = np.where(th <= thr, n15, n1h)
        print(f"  adaptive[fast if t_half<= {thr}h]: total {tot(pick):+.1f}%")
    # ORACLE upper bound (perfect per-trade choice)
    oracle = np.maximum(n15, n1h)
    print(f"\nORACLE (perfect per-trade choice): total {tot(oracle):+.1f}%   "
          f"(vs best fixed {max(tot(n15),tot(n1h)):+.1f}%)  => max gain if classifiable")
    print("\nDONE_ADAPTIVE_SHAPE.", flush=True)


if __name__ == "__main__":
    main()
