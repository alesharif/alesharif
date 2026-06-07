#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the deep hard-stop across thresholds 37%..42% (best system kept: DCA
[40-60-100-100-100], rebound completion, trailing stop 20%). Per stop level report
per-trade avg by year, % stopped out, and full-sample portfolio CAGR/DD.
Run:  python experiments/squeeze_stop_sweep.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0; DAY = 86400000
S, E = "2022-06-01", "2026-06-01"
LV = [0.0, 0.06, 0.12, 0.18, 0.24]; DIST = [40.0, 60.0, 100.0, 100.0, 100.0]; REB = [0.03, 0.06, 0.09]
CAP = 3.0; TRAIL = 0.20; START = 2000.0
STOPS = [0.37, 0.38, 0.39, 0.40, 0.41, 0.42]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def run(Hs, Ls, lastc, P0, trail, stop0):
    stoppx = P0*(1-stop0); cappx = P0*(1+CAP)
    invested = DIST[0]; coins = DIST[0]/P0; dfill = [False]*5; dfill[0] = True
    rdone = [False]*3; near = False; dipped = False; bottom = P0; active = False; peak = P0
    for k in range(len(Hs)):
        lo = Ls[k]; hi = Hs[k]
        if lo < bottom:
            bottom = lo; rdone = [False]*3
        if lo <= P0*0.97:                                     # meaningful dip below entry (>=3%)
            dipped = True
        for di in range(1, 5):
            if not dfill[di] and lo <= P0*(1-LV[di]) and invested < BUDGET-1e-9:
                price = P0*(1-LV[di]); a = min(DIST[di], BUDGET-invested)
                coins += a/price; invested += a; dfill[di] = True
        if not active and lo <= P0*(1-stop0):                 # deep stop while still in the dip
            return (coins*P0*(1-stop0)-invested)/BUDGET*100, True
        if active and lo <= stoppx:                           # trailing stop (after recovery)
            return (coins*stoppx-invested)/BUDGET*100, False
        if dipped and invested < BUDGET-1e-9:
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min((BUDGET-invested)/(3-ri), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        # activate trailing only after a real dip + recovery to entry, OR a straight-up +10%
        if not active and ((dipped and hi >= P0) or hi >= P0*1.10):
            active = True; peak = max(peak, hi); stoppx = peak*(1-trail)
        if active:
            peak = max(peak, hi); stoppx = max(stoppx, peak*(1-trail))
        if hi >= cappx:
            return (coins*cappx-invested)/BUDGET*100, False
    return (coins*lastc-invested)/BUDGET*100, False


def main():
    print("Deep-stop sweep 37%..42% (DCA + trailing 20%)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    sigs = []      # (ent_t, yr, Hs, Ls, lastc, P0)
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
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i]); yr = int(pd.Timestamp(ent_t, unit="ms").year)
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            sigs.append((ent_t, yr, H[j0:j1].copy(), L[j0:j1].copy(), C[min(j1, len(C)-1)], P0))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def portfolio(rr_list):     # rr_list aligned to sigs
        ss = sorted([(sigs[i][0], sigs[i][0]+WIN, rr_list[i]) for i in range(len(sigs))])
        cash = START; op = []; eqt = []; eqv = []
        for et, xt, rr in ss:
            op.sort()
            while op and op[0][0] <= et:
                xt0, p, s = op.pop(0); cash += p; eqt.append(xt0); eqv.append(cash + sum(z for _, _2, z in op))
            if len(op) >= 10 or cash < 100:
                continue
            stake = 100.0; cash -= stake; op.append((xt, stake*(1+rr/100), stake))
        for xt, p, s in sorted(op):
            cash += p; eqt.append(xt); eqv.append(cash)
        eqt = np.array(eqt); eqv = np.array(eqv); od = np.argsort(eqt); eqt = eqt[od]; eqv = eqv[od]
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        yrs = (eqt[-1]-eqt[0])/(365.25*DAY); cagr = ((eqv[-1]/START)**(1/yrs)-1)*100
        return cagr, dd

    yrs_arr = np.array([s[1] for s in sigs])
    print(f"إجمالي الإشارات: {len(sigs)}\n")
    print(f"{'الوقف':<8}{'2024':>9}{'2025':>9}{'2026':>9}{'الكل':>9}{'ضرب الوقف%':>12}{'CAGR':>7}{'سحب':>7}")
    print("-"*70)
    for st in STOPS:
        rr = np.array([run(s[2], s[3], s[4], s[5], TRAIL, st)[0] for s in sigs])
        hs = np.array([run(s[2], s[3], s[4], s[5], TRAIL, st)[1] for s in sigs])
        cagr, dd = portfolio(rr.tolist())
        def ya(y): return rr[yrs_arr == y].mean() if (yrs_arr == y).any() else float("nan")
        print(f"−{int(st*100)}%{'':<3}{ya(2024):>+8.1f}%{ya(2025):>+8.1f}%{ya(2026):>+8.1f}%{rr.mean():>+8.1f}%{hs.mean()*100:>11.0f}%{cagr:>+6.0f}%{dd:>+6.0f}%")
    print("-"*70)
    print("الأرقام = متوسط العائد/صفقة لكل سنة + محفظة 10×$100 (CAGR/سحب).")
    print("⚠️ متفائل بانحياز البقاء.")
    print("\nDONE_STOPSWEEP.", flush=True)


if __name__ == "__main__":
    main()
