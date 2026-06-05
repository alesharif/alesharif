#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEVELOP the squeeze on 2025 using the 50/50 finding: 'down-first' entries fail
96% of the time. So CUT FAST at -5% the moment a trade proves it goes down first,
instead of waiting for the -12% stop. Does early-cut turn the 2025 squeeze positive?

Compare on the SAME 2025 squeeze entries (return per trade + 2025 portfolio):
  A baseline      : TP+100 / SL-12
  B early-cut -5% : exit at -5% if it hits -5% before +5%; else TP+100 / SL-12
  C early-cut -4% : same with -4%
  D early-cut -6% : same with -6%
Run:  python experiments/squeeze_earlycut.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; DAY = 86400000
S, E = "2022-06-01", "2026-06-01"; W0, W1 = "2025-01-01", "2026-01-01"
B_TP = 2.0; B_SL = 0.12; COST = 0.01
START = 10_000.0; SLOTS = 6; ALLOC = 0.10


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def exit_rr(Hs, Ls, lastc, P0, cut):
    """cut=None -> plain TP/SL. cut=x -> if price reaches P0*(1-x) before +5%, exit at -x."""
    tp = P0*B_TP; sl = P0*(1-B_SL)
    up5 = P0*1.05
    if cut is not None:
        cutpx = P0*(1-cut)
        for k in range(len(Hs)):
            # decide which came first: the -cut level or +5%
            if Ls[k] <= cutpx and not (Hs[k] >= up5):
                return -cut-COST                 # down-first -> cut
            if Hs[k] >= up5:
                break                            # proved up-first -> manage normally below
            if Ls[k] <= cutpx:
                return -cut-COST
    for k in range(len(Hs)):
        if Ls[k] <= sl: return -B_SL-COST
        if Hs[k] >= tp: return (B_TP-1)-COST
    return (lastc/P0-1)-COST


def main():
    print("Squeeze 2025: early-cut the down-first failures\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    w0, w1 = ms(W0), ms(W1); rows = []        # (ent_t, exit_t, Hs, Ls, lastc, P0)
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
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n-1):
            if not (w0 <= int(t[i]) < w1):
                continue
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            rows.append((ent_t, T[min(j1, len(T)-1)], H[j0:j1].copy(), L[j0:j1].copy(), C[min(j1, len(C)-1)], P0))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def portfolio(rrs):    # rrs aligned with rows: list of (ent_t, exit_t, rr)
        ss = sorted(rrs, key=lambda x: x[0]); equity = START; op = []; eqt = []; eqv = []; nt = 0; wins = 0
        for et, xt, rr in ss:
            op.sort()
            while op and op[0][0] <= et:
                a, pnl = op.pop(0); equity += pnl; eqt.append(a); eqv.append(equity)
            if len(op) >= SLOTS:
                continue
            stake = equity*ALLOC; equity -= stake; op.append((xt, stake*(1+rr))); nt += 1; wins += rr > 0
        for xt, pnl in sorted(op):
            equity += pnl; eqt.append(xt); eqv.append(equity)
        if not eqv:
            return None
        eqv = np.array(eqv)
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        return (eqv[-1]/START-1)*100, dd, nt, wins/nt*100 if nt else float("nan")

    print(f"إشارات الانقباض في 2025: {len(rows)}\n")
    print(f"{'إعداد':<18}{'متوسط/صفقة':>12}{'win%':>7}{'محفظة%':>9}{'تراجع':>8}")
    print("-"*54)
    for lab, cut in [("A أساس (وقف-12)", None), ("B قطع -4%", 0.04), ("C قطع -5%", 0.05), ("D قطع -6%", 0.06)]:
        rrs = []
        for ent_t, xt, Hs, Ls, lastc, P0 in rows:
            rrs.append((ent_t, int(xt), exit_rr(Hs, Ls, lastc, P0, cut)))
        avg = np.mean([x[2] for x in rrs])*100; wr = np.mean([x[2] > 0 for x in rrs])*100
        res = portfolio(rrs)
        pr, dd, nt, _ = res
        print(f"{lab:<18}{avg:>+11.1f}%{wr:>6.0f}%{pr:>+8.0f}%{dd:>+7.0f}%")
    print("-"*54)
    print("الفكرة: قطع النازل-أولاً مبكراً يقلّص الخسارة الكبرى لكنه يقتل الـ4% التي ترتدّ.")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة).")
    print("\nDONE_CUT.", flush=True)


if __name__ == "__main__":
    main()
