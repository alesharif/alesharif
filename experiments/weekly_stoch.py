#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WEEKLY Stochastic bullish crossover (%K crosses above %D), slow stochastic 14,3,3
on weekly candles. Daily execution + DCA [40-60-100-100-100] + trailing 20%.
Variants: pure cross | + slow line %D > 10. Run: python experiments/weekly_stoch.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 180*24*3600*1000; BUDGET = 400.0; STOP0 = 0.42; DAY = 86400000
S, E = "2021-01-01", "2026-06-01"
LV = [0.0, 0.06, 0.12, 0.18, 0.24]; DIST = [40.0, 60.0, 100.0, 100.0, 100.0]; REB = [0.03, 0.06, 0.09]
CAP = 3.0; TRAIL = 0.20; START = 2000.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)


def run(Hs, Ls, lastc, P0, trail, stop0):
    stoppx = P0*(1-stop0); cappx = P0*(1+CAP)
    invested = DIST[0]; coins = DIST[0]/P0; dfill = [False]*5; dfill[0] = True
    rdone = [False]*3; near = False; dipped = False; bottom = P0; active = False; peak = P0
    for k in range(len(Hs)):
        lo = Ls[k]; hi = Hs[k]
        if lo < bottom:
            bottom = lo; rdone = [False]*3
        if lo <= P0*0.97:
            dipped = True
        for di in range(1, 5):
            if not dfill[di] and lo <= P0*(1-LV[di]) and invested < BUDGET-1e-9:
                price = P0*(1-LV[di]); a = min(DIST[di], BUDGET-invested)
                coins += a/price; invested += a; dfill[di] = True
        if not active and lo <= P0*(1-stop0):
            return (coins*P0*(1-stop0)-invested)/BUDGET*100
        if active and lo <= stoppx:
            return (coins*stoppx-invested)/BUDGET*100
        if dipped and invested < BUDGET-1e-9:
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min((BUDGET-invested)/(3-ri), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        if not active and ((dipped and hi >= P0) or hi >= P0*1.10):
            active = True; peak = max(peak, hi); stoppx = peak*(1-trail)
        if active:
            peak = max(peak, hi); stoppx = max(stoppx, peak*(1-trail))
        if hi >= cappx:
            return (coins*cappx-invested)/BUDGET*100
    return (coins*lastc-invested)/BUDGET*100


def main():
    print("WEEKLY Stochastic bullish crossover (14,3,3) + DCA + trailing 20%\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    rows = []
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        dd = pd.DataFrame({"c": g["close"].resample("D").last(), "v": g["volume"].resample("D").sum(),
                           "t": g["time"].resample("D").last()}).dropna()
        dc = dd["c"].to_numpy(); dt_ = dd["t"].to_numpy(); ddv = (dd["c"]*dd["v"]).to_numpy()
        wk = pd.DataFrame({"h": g["high"].resample("W").max(), "l": g["low"].resample("W").min(),
                           "c": g["close"].resample("W").last(), "t": g["time"].resample("W").last()}).dropna()
        wc = wk["c"].to_numpy(); wh = wk["h"].to_numpy(); wl = wk["l"].to_numpy(); wt = wk["t"].to_numpy(); nw = len(wc)
        if nw < 30:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        ll = pd.Series(wl).rolling(14).min().to_numpy(); hh = pd.Series(wh).rolling(14).max().to_numpy()
        rng = np.where((hh-ll) == 0, 1e-9, hh-ll)
        kfast = 100*(wc-ll)/rng
        kslow = pd.Series(kfast).rolling(3).mean().to_numpy()
        dslow = pd.Series(kslow).rolling(3).mean().to_numpy()
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for w in range(18, nw-1):
            if not (np.isfinite(dslow[w]) and kslow[w-1] <= dslow[w-1] and kslow[w] > dslow[w]):
                continue
            ent_t = int(wt[w]); di = np.searchsorted(dt_, ent_t, side="right") - 1
            if di < 30 or np.nanmean(ddv[max(0, di-30):di]) <= LIQ_MIN:
                continue
            P0 = float(dc[di]); yr = int(pd.Timestamp(ent_t, unit="ms").year)
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            rr = run(H[j0:j1], L[j0:j1], C[min(j1, len(C)-1)], P0, TRAIL, STOP0)
            rows.append((ent_t, yr, rr, float(dslow[w])))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()
    rows.sort()

    def portfolio(sel):
        ss = sorted([(r[0], r[0]+WIN, r[2]) for r in sel]); cash = START; op = []; eqt = []; eqv = []
        for et, xt, x in ss:
            op.sort()
            while op and op[0][0] <= et:
                xt0, p, s = op.pop(0); cash += p; eqt.append(xt0); eqv.append(cash + sum(z for _, _2, z in op))
            if len(op) >= 10 or cash < 100:
                continue
            cash -= 100.0; op.append((xt, 100.0*(1+x/100), 100.0))
        for xt, p, s in sorted(op):
            cash += p; eqt.append(xt); eqv.append(cash)
        if not eqv:
            return float("nan"), float("nan")
        eqt = np.array(eqt); eqv = np.array(eqv); od = np.argsort(eqt); eqt = eqt[od]; eqv = eqv[od]
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        ynr = (eqt[-1]-eqt[0])/(365.25*DAY); cagr = ((eqv[-1]/START)**(1/ynr)-1)*100
        return cagr, dd

    def report(sel, title):
        if not sel:
            print(f"===== {title}: 0 =====\n"); return
        yrs = np.array([r[1] for r in sel]); rr = np.array([r[2] for r in sel])
        print(f"===== {title}: {len(sel)} إشارة =====")
        print(f"{'السنة':<8}{'إشارات':>8}{'عائد/صفقة':>12}{'نسبة الربح':>11}")
        for y in [2023, 2024, 2025, 2026, None]:
            m = (yrs == y) if y else np.ones(len(sel), bool)
            if not m.any():
                continue
            a = rr[m]; lab = "الكل" if y is None else str(y)
            print(f"{lab:<8}{m.sum():>8}{a.mean():>+11.1f}%{(a > 0).mean()*100:>10.0f}%")
        cagr, dd = portfolio(sel)
        print(f"محفظة $2000 (10×$100): CAGR {cagr:+.0f}%   سحب {dd:+.0f}%\n")

    report(rows, "صِرف (بلا عتبة)")
    for thr in [10, 15, 20, 25, 30]:
        report([r for r in rows if r[3] > thr], f"+ الخط البطيء %D > {thr}")
    print("⚠️ متفائل بانحياز البقاء.")
    print("\nDONE_WSTOCH.", flush=True)


if __name__ == "__main__":
    main()
