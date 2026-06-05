#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Position sizing by LIQUIDITY: stake = 0.2% of the coin's $volume (30d avg).
Thin coins (volatile) get sized DOWN; liquid coins keep normal size -> should cut
drawdown (variance) without touching the per-trade edge. Best param + trail 20%.

Modes:
  fixed10  : baseline, 10% of equity per trade (slots=10).
  volpure  : stake = 0.2% * $vol, capped only by available cash.
  volcap10 : stake = min(0.2% * $vol, 10% of equity, cash)  -> only shrinks thin coins.
Reports CAGR, max drawdown, yearly, avg stake, % trades shrunk below 10%.
Run:  python experiments/dca_volsize.py
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
CAP = 3.0; TRAIL = 0.20; START = 10_000.0; VOLPCT = 0.002


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
    print("Liquidity-based sizing: stake = 0.2% of coin $volume\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    trades = []   # (ent_t, exit_t, rr, vol30)
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
            vol30 = np.nanmean(dv[max(0, i-30):i])
            if vol30 <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            rr = run(H[j0:j1], L[j0:j1], C[min(j1, len(C)-1)], P0, TRAIL)
            trades.append((ent_t, ent_t+WIN, rr, float(vol30)))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()
    trades.sort()

    def portfolio(mode):
        cash = START; op = []; eqt = []; eqv = []; stakes = []; shrunk = 0
        for et, xt, rr, vol30 in trades:
            op.sort()
            while op and op[0][0] <= et:
                xt0, p, _s = op.pop(0); cash += p
                eqt.append(xt0); eqv.append(cash + sum(s for _, _2, s in op))
            tot = cash + sum(s for _, _2, s in op)          # total account value (deployed at cost)
            if cash <= 1:
                continue
            if mode == "fixed10":
                if len(op) >= 10:
                    continue
                stake = tot*0.10
            elif mode == "volpure":
                stake = VOLPCT*vol30
            else:  # volcap10
                stake = min(VOLPCT*vol30, tot*0.10)
                if VOLPCT*vol30 < tot*0.10:
                    shrunk += 1
            stake = min(stake, cash)
            if stake < 1:
                continue
            cash -= stake; op.append((xt, stake*(1+rr/100), stake)); stakes.append(stake)
        for xt, p, _s in sorted(op):
            cash += p; eqt.append(xt); eqv.append(cash)
        eqt = np.array(eqt); eqv = np.array(eqv); od = np.argsort(eqt); eqt = eqt[od]; eqv = eqv[od]
        s = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("YE").last()
        yrly = {}; prev = START
        for ts, val in s.items():
            yrly[ts.year] = (val/prev-1)*100; prev = val
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        yrs = (eqt[-1]-eqt[0])/(365.25*DAY); cagr = ((eqv[-1]/START)**(1/yrs)-1)*100
        return eqv[-1], cagr, dd, yrly, np.median(stakes) if stakes else 0, len(stakes), shrunk

    print(f"إجمالي الصفقات: {len(trades)}")
    v = np.array([x[3] for x in trades])
    print(f"فوليوم العملات: وسيط ${np.median(v):,.0f}/يوم → 0.2% = ${0.002*np.median(v):,.0f} وسيط حجم الصفقة\n")
    print(f"{'الطريقة':<14}{'نهائي$':>10}{'CAGR':>7}{'سحب':>7}{'وسيط الصفقة':>13}   2024/2025/2026")
    print("-"*76)
    for mode, lab in [("fixed10", "ثابت 10%"), ("volcap10", "فوليوم بسقف10%"), ("volpure", "فوليوم خالص")]:
        fin, cagr, dd, yrly, medst, nt, shrunk = portfolio(mode)
        ys = "/".join(f"{yrly.get(y, float('nan')):+.0f}" for y in [2024, 2025, 2026])
        print(f"{lab:<14}{fin:>9,.0f}{cagr:>+6.0f}%{dd:>+6.0f}%{medst:>12,.0f}$   {ys}%")
    print("-"*76)
    print("الفكرة: تصغير العملات الرقيقة (الأكثر تذبذباً) يخفّض السحب بنفس الحافة.")
    print("⚠️ متفائل بانحياز البقاء.")
    print("\nDONE_VOL.", flush=True)


if __name__ == "__main__":
    main()
