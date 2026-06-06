#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REAL numbers for a $2,000 spot account. DCA needs $400/trade -> max 5 concurrent.
Best param [40-60-100-100-100], stop -42%, rebound completion, trailing stop 20%.
Two sizings: FIXED $400/trade (flat) and COMPOUND 20%/trade. Reports yearly in $.
Run:  python experiments/dca_2000.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0; STOP0 = 0.42; DAY = 86400000
S, E = "2022-06-01", "2026-06-01"
LV = [0.0, 0.06, 0.12, 0.18, 0.24]; DIST = [40.0, 60.0, 100.0, 100.0, 100.0]; REB = [0.03, 0.06, 0.09]
CAP = 3.0; TRAIL = 0.20; START = 2000.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def run(Hs, Ls, lastc, P0, trail):
    stoppx = P0*(1-STOP0); cappx = P0*(1+CAP)
    invested = DIST[0]; coins = DIST[0]/P0; dfill = [False]*5; dfill[0] = True
    rdone = [False]*3; near = False; dipped = False; bottom = P0; active = False; peak = P0
    for k in range(len(Hs)):
        lo = Ls[k]; hi = Hs[k]
        if lo < bottom:
            bottom = lo; rdone = [False]*3
        if lo < P0:
            dipped = True
        for di in range(1, 5):
            if not dfill[di] and lo <= P0*(1-LV[di]) and invested < BUDGET-1e-9:
                price = P0*(1-LV[di]); a = min(DIST[di], BUDGET-invested)
                coins += a/price; invested += a; dfill[di] = True
        if lo <= stoppx:
            return (coins*stoppx-invested)/BUDGET*100
        if dipped and invested < BUDGET-1e-9:
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min((BUDGET-invested)/(3-ri), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        if hi >= P0:
            active = True
        if active:
            peak = max(peak, hi); stoppx = max(stoppx, peak*(1-trail))
        if hi >= cappx:
            return (coins*cappx-invested)/BUDGET*100
    return (coins*lastc-invested)/BUDGET*100


def main():
    print("REAL $2,000 spot account — 5 concurrent $400 DCA trades, trail 20%\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    trades = []
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
            P0 = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            rr = run(H[j0:j1], L[j0:j1], C[min(j1, len(C)-1)], P0, TRAIL)
            trades.append((ent_t, ent_t+WIN, rr))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()
    trades.sort()

    def sim(slots, budget):
        cash = START; op = []; eqt = []; eqv = []; taken = 0
        for et, xt, rr in trades:
            op.sort()
            while op and op[0][0] <= et:
                xt0, p, s = op.pop(0); cash += p; eqt.append(xt0); eqv.append(cash + sum(z for _, _2, z in op))
            if len(op) >= slots:
                continue
            stake = budget
            if stake > cash + 1e-6:
                continue
            cash -= stake; op.append((xt, stake*(1+rr/100), stake)); taken += 1
        for xt, p, s in sorted(op):
            cash += p; eqt.append(xt); eqv.append(cash)
        eqt = np.array(eqt); eqv = np.array(eqv); od = np.argsort(eqt); eqt = eqt[od]; eqv = eqv[od]
        sY = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("YE").last()
        yrly = {ts.year: val for ts, val in sY.items()}
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        yrs = (eqt[-1]-eqt[0])/(365.25*DAY); cagr = ((eqv[-1]/START)**(1/yrs)-1)*100
        return eqv[-1], cagr, dd, yrly, taken

    print(f"إشارات الانقباض الكلية: {len(trades)}\n")
    print(f"{'الإعداد':<26}{'نهائي$':>9}{'CAGR':>7}{'سحب':>7}{'صفقات':>8}   2024/2025/2026 ($)")
    print("-"*78)
    CONF = [(5, 400, "5×$400 (100% نشر)"), (10, 200, "10×$200 (100%)"), (15, 133, "15×$133 (100%)"),
            (20, 100, "20×$100 (100%)"), (10, 100, "10×$100 (50% نشر)"), (20, 50, "20×$50 (50%)")]
    for slots, budget, lab in CONF:
        fin, cagr, dd, yrly, taken = sim(slots, budget)
        ys = "/".join(f"${yrly.get(y, float('nan')):,.0f}" for y in [2024, 2025, 2026])
        print(f"{lab:<26}{fin:>8,.0f}{cagr:>+6.0f}%{dd:>+6.0f}%{taken:>8}   {ys}")
    print("-"*78)
    print("التنويع (صفقات أكثر أصغر) يخفّض السحب؛ تقليل النشر (نقد احتياطي) يخفّضه أكثر.")
    print("⚠️ متفائل بانحياز البقاء. عمولة/انزلاق غير محسوبة — تؤذي الصفقات الصغيرة أكثر.")
    print("\nDONE_2000.", flush=True)


if __name__ == "__main__":
    main()
