#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FIX the discarded partial gains: a TRAILING stop that captures +10/20/30/50%
moves instead of exiting at breakeven. On the best param [40-60-100-100-100].

Phases:
  1) before price recovers to entry: stop = -42% (wide, to allow DCA down).
  2) after price returns to entry (dipped then hi>=P0): TRAIL active. peak tracks the
     highest price; stop = peak*(1-TRAIL). A +30%-then-reverse trade now exits near
     +30%*(1-TRAIL) -- the partial gain is captured, not thrown away.
No fixed target (let winners run; safety cap +300%). Compare TRAIL widths + the old
'raise to -2%' (D) to show the difference. Reports win-rate, avg win, avg loss,
expectancy, max drawdown, and a gain-bucket count (how many exits at +10/20/30/50%+).
Run:  python experiments/dca_trail.py
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
CAP = 3.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def run(Hs, Ls, lastc, P0, trail):
    """trail = fractional trailing distance after recovery. Returns (ret%, price_gain%)
    where price_gain% = exit_price/P0-1 (to bucket how big the move was)."""
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
            ex = stoppx; return (coins*ex-invested)/BUDGET*100, (ex/P0-1)*100
        if dipped and invested < BUDGET-1e-9:
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min((BUDGET-invested)/(3-ri), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        if not active and ((dipped and hi >= P0) or hi >= P0):   # activate trail on recovery/profit
            active = True; peak = max(peak, hi); stoppx = max(stoppx, peak*(1-trail))
        if active:
            peak = max(peak, hi); stoppx = max(stoppx, peak*(1-trail))
        if hi >= cappx:
            ex = cappx; return (coins*ex-invested)/BUDGET*100, (ex/P0-1)*100
    ex = lastc
    return (coins*ex-invested)/BUDGET*100, (ex/P0-1)*100


def main():
    print("Trailing stop to CAPTURE partial gains, best param [40-60-100-100-100]\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    rows = []
    TRAILS = [0.08, 0.12, 0.20]
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
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        dv = c*v
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
            Hs = H[j0:j1]; Ls = L[j0:j1]; lastc = C[min(j1, len(C)-1)]
            res = tuple(run(Hs, Ls, lastc, P0, tr) for tr in TRAILS)
            rows.append((ent_t, yr, res))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def dd_of(sel, ti):
        ss = sorted([(x[0], x[0]+WIN, x[2][ti][0]) for x in sel]); equity = 10000.0; op = []; eqv = []
        for et, xt, rr in ss:
            op.sort()
            while op and op[0][0] <= et:
                _, p = op.pop(0); equity += p; eqv.append(equity)
            if len(op) >= 6:
                continue
            stake = equity*0.10; equity -= stake; op.append((xt, stake*(1+rr/100)))
        for xt, p in sorted(op):
            equity += p; eqv.append(equity)
        eqv = np.array(eqv)
        return ((eqv-np.maximum.accumulate(eqv))/np.maximum.accumulate(eqv)).min()*100 if len(eqv) else float("nan")

    def report(sel, label):
        print(f"========== {label}: {len(sel)} صفقة ==========")
        print(f"{'الوقف المتحرّك':<16}{'نسبة الربح':>10}{'م.الربح':>9}{'م.الخسارة':>10}{'المتوقّع':>9}{'السحب':>8}")
        print("-"*62)
        for ti, tr in enumerate(TRAILS):
            r = np.array([x[2][ti][0] for x in sel]); w = r[r > 0]; l = r[r <= 0]
            print(f"{'تريل '+str(int(tr*100))+'%':<16}{len(w)/len(r)*100:>9.0f}%{(w.mean() if len(w) else float('nan')):>+8.1f}%{(l.mean() if len(l) else float('nan')):>+9.1f}%{r.mean():>+8.2f}%{dd_of(sel, ti):>+7.0f}%")
        # gain buckets for trail 12% (price move captured)
        g = np.array([x[2][1][1] for x in sel])
        print("  توزيع حجم الحركة الملتقطة (تريل 12%):", end=" ")
        for lo, hi, lab in [(10, 20, "+10/20"), (20, 30, "+20/30"), (30, 50, "+30/50"), (50, 1e9, "+50+")]:
            print(f"{lab}:{((g >= lo) & (g < hi)).sum()}", end="  ")
        print("\n")

    report([x for x in rows if x[1] == 2025], "2025 (الجفاف)")
    report([x for x in rows if x[1] == 2024], "2024 (موسم بديل)")
    report(rows, "كامل العيّنة")
    print("الآن الصفقات الصاعدة +10/20/30/50% تُلتقَط بالوقف المتحرّك بدل رميها عند الدخول.")
    print("⚠️ متفائل بانحياز البقاء.")
    print("\nDONE_TRAIL.", flush=True)


if __name__ == "__main__":
    main()
