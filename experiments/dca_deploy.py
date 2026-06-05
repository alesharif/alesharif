#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLAN 2: deploy MORE capital. The per-trade edge is +8.3% (trail 20%); returns
should scale with deployment until drawdown bites. Sweep (SLOTS x ALLOC) = total
capital deployed, on the SAME trades, and report annual compounded return + max DD.

Trades fixed: squeeze entry, DCA [40-60-100-100-100], stop -42%, rebound completion,
trailing stop 20% after recovery. Run:  python experiments/dca_deploy.py
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
CAP = 3.0; TRAIL = 0.20; START = 10_000.0


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
    print("PLAN 2: scale capital deployment (trail 20%), annual return vs drawdown\n", flush=True)
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

    def portfolio(slots, alloc):
        equity = START; op = []; eqt = []; eqv = []
        for et, xt, rr in trades:
            op.sort()
            while op and op[0][0] <= et:
                _, p = op.pop(0); equity += p; eqt.append(_); eqv.append(equity)
            if len(op) >= slots or equity <= 0:
                continue
            stake = equity*alloc; equity -= stake; op.append((xt, stake*(1+rr/100)))
        for xt, p in sorted(op):
            equity += p; eqt.append(xt); eqv.append(equity)
        eqt = np.array(eqt); eqv = np.array(eqv); od = np.argsort(eqt); eqt = eqt[od]; eqv = eqv[od]
        s = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("YE").last()
        yrly = {}; prev = START
        for ts, val in s.items():
            yrly[ts.year] = (val/prev-1)*100; prev = val
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        yrs = (eqt[-1]-eqt[0])/(365.25*DAY); cagr = ((eqv[-1]/START)**(1/yrs)-1)*100
        return eqv[-1], cagr, dd, yrly

    print(f"إجمالي الصفقات: {len(trades)}\n")
    print(f"{'النشر':<22}{'نهائي$':>11}{'CAGR':>7}{'سحب':>7}   2024 / 2025 / 2026")
    print("-"*74)
    CONF = [(6, 0.10, "60% (الحالي)"), (8, 0.125, "100%"), (10, 0.10, "100% (خانات أكثر)"),
            (8, 0.15, "120% رافعة"), (10, 0.15, "150% رافعة"), (12, 0.167, "200% رافعة")]
    for slots, alloc, lab in CONF:
        fin, cagr, dd, yrly = portfolio(slots, alloc)
        ys = " / ".join(f"{yrly.get(y, float('nan')):+.0f}%" for y in [2024, 2025, 2026])
        print(f"{lab:<22}{fin:>10,.0f}{cagr:>+6.0f}%{dd:>+6.0f}%   {ys}")
    print("-"*74)
    print("النشر >100% يعني رافعة (هامش) — يضخّم الربح والسحب معاً وخطر التصفية.")
    print("⚠️ متفائل بانحياز البقاء — الرقم الحقيقي أقل، والسحب أكبر فعلياً.")
    print("\nDONE_DEP.", flush=True)


if __name__ == "__main__":
    main()
