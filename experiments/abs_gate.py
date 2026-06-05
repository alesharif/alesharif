#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ABSOLUTE supply gate (not relative median). The median gate always lets ~half
the signals through, even in a uniform drought (2025 threshold fell to 4%). An
ABSOLUTE gate trades breakout only when market runner-supply exceeds a fixed level
-> automatically silent through the whole 2025 drought, loud in 2024 alt-season.

Sweep ABS_THR; report FULL-period and 2025-only portfolio (return, drawdown).
breakout when supply>=THR else CASH. Run: python experiments/abs_gate.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000
S, E, SF = "2022-06-01", "2026-06-01", "2024-01-01"
BWIN = 120*24*3600*1000; B_SL = 0.12; B_TP = 2.0
START = 10_000.0; SLOTS = 8; ALLOC = 0.10
THRS = [6, 9, 12, 15, 18, 22]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("ABSOLUTE supply gate for breakout — full period + 2025-only\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    runs = {}; tots = {}; sigs = []        # (ent_t, exit_t, ret_frac)
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        d = pd.DataFrame({"o": g["open"].resample("D").first(), "h": g["high"].resample("D").max(),
                          "l": g["low"].resample("D").min(), "c": g["close"].resample("D").last(),
                          "v": g["volume"].resample("D").sum(), "t": g["time"].resample("D").last()}).dropna()
        c = d["c"].to_numpy(); n = len(c)
        if n < 220:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        for i in range(60, n):
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            day = int(t[i]) // DAY; tots[day] = tots.get(day, 0) + 1
            if c[i] / c[i-60] - 1 >= 0.50:
                runs[day] = runs.get(day, 0) + 1
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        for i in range(200, n-1):
            if int(t[i]) < ms(SF):
                continue
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i]); sl = P0*(1-B_SL); tp = P0*B_TP
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+BWIN, side="right")
            rr = None; xt = None
            for k in range(j0, j1):
                if L[k] <= sl: rr = -B_SL - 0.01; xt = int(T[k]); break
                if H[k] >= tp: rr = (B_TP-1) - 0.01; xt = int(T[k]); break
            if rr is None:
                ke = min(j1, len(C)-1); rr = (C[ke]/P0-1) - 0.01; xt = int(T[ke])
            sigs.append((ent_t, xt, rr))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tots); sup_t = np.array([dd*DAY for dd in days])
    sup_v = np.array([runs.get(dd, 0)/tots[dd]*100 for dd in days])
    def supply(tt):
        i = np.searchsorted(sup_t, tt, side="left") - 1
        return sup_v[i] if i >= 0 else np.nan
    sup_at = np.array([supply(s[0]) for s in sigs])
    ss = sorted(zip(sigs, sup_at), key=lambda x: x[0][0])
    w0, w1 = ms("2025-01-01"), ms("2026-01-01")

    def portfolio(thr, win=None):
        equity = START; op = []; eqt = []; eqv = []; nt = 0
        for (et, xt, rr), sup in ss:
            op.sort()
            while op and op[0][0] <= et:
                a, pnl = op.pop(0); equity += pnl; eqt.append(a); eqv.append(equity)
            if win and not (win[0] <= et < win[1]):
                continue
            if not (sup >= thr) or len(op) >= SLOTS:
                continue
            stake = equity*ALLOC; equity -= stake; op.append((xt, stake*(1+rr))); nt += 1
        for xt, pnl in sorted(op):
            equity += pnl; eqt.append(xt); eqv.append(equity)
        if not eqv:
            return None
        eqt = np.array(eqt); eqv = np.array(eqv); o = np.argsort(eqt); eqt = eqt[o]; eqv = eqv[o]
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        return eqv[-1], dd, nt

    print(f"عرض الرابضين: وسيط={np.nanmedian(sup_at):.0f}%  أقصى={np.nanmax(sup_at):.0f}%  |  إشارات {len(sigs)}\n")
    print(f"{'عتبة مطلقة':>10}{'صفقات(كل)':>11}{'كامل عائد%':>12}{'كامل تراجع':>12}   ||{'2025 عائد%':>11}{'2025 تراجع':>12}")
    print("-"*72)
    # reorder ss for 2025 window run (need separate equity)
    for thr in THRS:
        full = portfolio(thr)
        y25 = portfolio(thr, (w0, w1))
        if full is None:
            continue
        ff, fd, fn = full
        if y25 is None:
            print(f"{thr:>9}%{fn:>11}{(ff/START-1)*100:>+11.0f}%{fd:>+11.0f}%   ||{'لا صفقات':>23}")
        else:
            yf, yd, yn = y25
            print(f"{thr:>9}%{fn:>11}{(ff/START-1)*100:>+11.0f}%{fd:>+11.0f}%   ||{(yf/START-1)*100:>+10.0f}%{yd:>+11.0f}%  (n{yn})")
    print("-"*72)
    print("الهدف: عتبة تُبقي ربح 2024 وتُطفئ اختراق 2025 (يصبح قريباً من 0، نقد).")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_ABS.", flush=True)


if __name__ == "__main__":
    main()
