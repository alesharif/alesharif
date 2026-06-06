#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add a WEEKLY EMA200 trend filter to the squeeze strategy: require daily close >
weekly EMA200 (lagged, last completed weekly bar -> no lookahead). Compare baseline
vs +filter, per year: signal count, avg per-trade return (trail-20 DCA), win-rate.
Run:  python experiments/squeeze_weekly_ema200.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0; STOP0 = 0.42
S, E = "2022-06-01", "2026-06-01"
LV = [0.0, 0.06, 0.12, 0.18, 0.24]; DIST = [40.0, 60.0, 100.0, 100.0, 100.0]; REB = [0.03, 0.06, 0.09]
CAP = 3.0; TRAIL = 0.20


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
    print("Weekly EMA200 trend filter on the squeeze — baseline vs +filter\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    rows = []      # (yr, rr, passes_weekly)
    short_hist = 0
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
        # weekly EMA200 (lagged): value from the last completed weekly bar
        wk = g["close"].resample("W").last().dropna()
        we200 = pd.Series(wk.to_numpy()).ewm(span=200, adjust=False).mean().to_numpy()
        wk_t = np.array([ts.value // 10**6 for ts in wk.index])   # weekly bar end (ms)
        nweeks = len(wk)
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
            wi = np.searchsorted(wk_t, ent_t, side="right") - 1   # last completed weekly bar
            passes = wi >= 0 and P0 > we200[wi]
            if nweeks < 60:                                       # flag thin weekly history
                short_hist += 1
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            rr = run(H[j0:j1], L[j0:j1], C[min(j1, len(C)-1)], P0, TRAIL)
            rows.append((yr, rr, passes))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def stat(sel):
        if not sel:
            return (0, float("nan"), float("nan"))
        r = np.array([x[1] for x in sel]); return (len(r), r.mean(), (r > 0).mean()*100)

    print(f"إجمالي الإشارات: {len(rows)}   |   منها بتاريخ أسبوعي قصير (<60 أسبوع): {short_hist}\n")
    print(f"{'السنة':<8}{'إشارات(بدون)':>14}{'عائد(بدون)':>12}{'إشارات(+فلتر)':>15}{'عائد(+فلتر)':>13}{'تنجو%':>8}")
    print("-"*72)
    for yr in [2024, 2025, 2026, None]:
        base = [x for x in rows if (yr is None or x[0] == yr)]
        filt = [x for x in base if x[2]]
        nb, mb, _ = stat(base); nf, mf, _ = stat(filt)
        lab = "الكل" if yr is None else str(yr)
        surv = nf/nb*100 if nb else float("nan")
        print(f"{lab:<8}{nb:>14}{mb:>+11.1f}%{nf:>15}{mf:>+12.1f}%{surv:>7.0f}%")
    print("-"*72)
    print("السؤال: هل فلتر EMA200 الأسبوعي يرفع العائد (يقصّ الصفقات الرديئة) أم يقصّ إشارات كثيرة بلا فائدة؟")
    print("⚠️ متفائل بانحياز البقاء. EMA200 أسبوعي على عملات حديثة = تاريخ ناقص.")
    print("\nDONE_WK200.", flush=True)


if __name__ == "__main__":
    main()
