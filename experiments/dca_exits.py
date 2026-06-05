#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test EXIT methods on the best capital-management param [40-60-100-100-100].

Best param (fixed): $40 @ entry, $60@-6, $100@-12, $100@-18, $100@-24; stop -42%;
rebound completion to full $400 before returning to entry. Entry = squeeze.

EXIT variants compared:
  A  +1% over entry        : user's capital-mgmt assumption (baseline).
  B  target +100%, stop -42 fixed.
  C  target +100%, stop -42 -> RAISE to -2% once price returns to entry (user's idea).
  D  target +50%,  stop -42 -> RAISE to -2% once price returns to entry.
Reports win-rate, avg win, avg loss, expectancy, max drawdown, per 2025/2024/full.
Run:  python experiments/dca_exits.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0; STOP0 = 0.42; TRAIL_TO = 0.02
S, E = "2022-06-01", "2026-06-01"
LV = [0.0, 0.06, 0.12, 0.18, 0.24]; DIST = [40.0, 60.0, 100.0, 100.0, 100.0]; REB = [0.03, 0.06, 0.09]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def build(Hs, Ls, P0):
    """Deploy DCA and return per-bar arrays of (invested, coins) state is complex;
    instead we run the exit logic inline per variant. This returns nothing; see run()."""


def run(Hs, Ls, lastc, P0, mode):
    """mode: 'p1' (+1% exit) | 'b' (+100 fixed) | 'c' (+100 trail) | 'd' (+50 trail)."""
    tp = {"p1": 0.01, "b": 1.0, "c": 1.0, "d": 0.50}[mode]
    trail = mode in ("c", "d")
    exitpx = P0*(1+tp); stoppx = P0*(1-STOP0)
    invested = DIST[0]; coins = DIST[0]/P0; dfill = [False]*5; dfill[0] = True
    rdone = [False]*3; near = False; dipped = False; bottom = P0; raised = False
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
        if lo <= stoppx:                              # stop (current level)
            return (coins*stoppx-invested)/BUDGET*100
        if dipped and invested < BUDGET-1e-9:         # rebound completion
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min((BUDGET-invested)/(3-ri), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        if trail and not raised and dipped and hi >= P0:   # recovered to entry -> raise stop
            stoppx = P0*(1-TRAIL_TO); raised = True
        if hi >= exitpx:
            return (coins*exitpx-invested)/BUDGET*100
    return (coins*min(lastc, P0 if mode == "p1" else lastc)-invested)/BUDGET*100


def main():
    print("Exit methods on best param [40-60-100-100-100], stop -42%\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    rows = []
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
            rows.append((ent_t, yr,
                         run(Hs, Ls, lastc, P0, "p1"), run(Hs, Ls, lastc, P0, "b"),
                         run(Hs, Ls, lastc, P0, "c"), run(Hs, Ls, lastc, P0, "d")))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def dd_of(sel, idx):
        ss = sorted([(x[0], x[0]+WIN, x[idx]) for x in sel]); equity = 10000.0; op = []; eqv = []
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
        print(f"{'الخروج':<28}{'نسبة الربح':>10}{'م.الربح':>9}{'م.الخسارة':>10}{'المتوقّع':>9}{'السحب':>8}")
        print("-"*74)
        for idx, name in [(2, "A +1% فوق الدخول"), (3, "B هدف +100% وقف ثابت"),
                          (4, "C هدف +100% + رفع وقف"), (5, "D هدف +50% + رفع وقف")]:
            r = np.array([x[idx] for x in sel]); w = r[r > 0]; l = r[r <= 0]
            print(f"{name:<28}{len(w)/len(r)*100:>9.0f}%{(w.mean() if len(w) else float('nan')):>+8.1f}%{(l.mean() if len(l) else float('nan')):>+9.1f}%{r.mean():>+8.2f}%{dd_of(sel, idx):>+7.0f}%")
        print()

    report([x for x in rows if x[1] == 2025], "2025 (الجفاف)")
    report([x for x in rows if x[1] == 2024], "2024 (موسم بديل)")
    report(rows, "كامل العيّنة")
    print("الفكرة الجديدة = C/D: ادعم نزولاً، وإذا عاد للدخول ارفع الوقف لـ−2% ودع الباقي يركض للهدف.")
    print("⚠️ متفائل بانحياز البقاء.")
    print("\nDONE_EXITS.", flush=True)


if __name__ == "__main__":
    main()
