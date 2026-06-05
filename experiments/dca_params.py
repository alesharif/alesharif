#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backtest the user's capital-management parameters, ISOLATED from target size.

Per the doc: evaluate capital management ONLY. Do NOT use the big targets in the
data. Assume every winning trade rebounds and exits at the ORIGINAL entry point.
=> EXIT = return to entry P0 (after a dip). Profit comes purely from how much the
DCA lowered the average cost. Upside is capped at entry (target size removed).

Params:  $40 @ entry, $60 @ -6%, $100 @ -12%, $100 @ -18%, $100 @ -24%  (=$400).
Stop loss -42%. Rebound completion deploys the remainder so the position reaches
$400 before price returns to entry. Entry signal = squeeze (unchanged).
Metrics: win-rate, avg win, avg loss, expectancy, max drawdown.
Run:  python experiments/dca_params.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0; STOP = 0.42
S, E = "2022-06-01", "2026-06-01"
LV = [0.0, 0.06, 0.12, 0.18, 0.24]
DIST = [40.0, 60.0, 100.0, 100.0, 100.0]
REB = [0.03, 0.06, 0.09]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def simulate(Hs, Ls, lastc, P0, dca=True):
    """Exit = return to entry P0 after a dip. Stop -42%. Timeout capped at entry.
    dca=False -> simple single $400 entry at P0 (same exit logic)."""
    stoppx = P0*(1-STOP)
    invested = 0.0; coins = 0.0; dfill = [False]*5; rdone = [False]*3; near = False
    dipped = False; bottom = P0
    if dca:
        coins += DIST[0]/P0; invested += DIST[0]; dfill[0] = True
    else:
        coins += BUDGET/P0; invested += BUDGET
    for k in range(len(Hs)):
        lo = Ls[k]; hi = Hs[k]
        if lo < bottom:
            bottom = lo; rdone = [False]*3
        if lo < P0:
            dipped = True
        if dca:
            for di in range(1, 5):
                if not dfill[di] and lo <= P0*(1-LV[di]) and invested < BUDGET-1e-9:
                    price = P0*(1-LV[di]); a = min(DIST[di], BUDGET-invested)
                    coins += a/price; invested += a; dfill[di] = True
        if lo <= stoppx:                              # hard stop -42%
            return (coins*stoppx-invested)/BUDGET*100, "stop"
        if dca and dipped and invested < BUDGET-1e-9:
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min((BUDGET-invested)/(3-ri), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        if dipped and hi >= P0:                       # winner: returned to entry
            if invested < BUDGET-1e-9:                # ensure full deployment at entry
                a = BUDGET-invested; coins += a/P0; invested += a
            return (coins*P0-invested)/BUDGET*100, "win"
    if invested < BUDGET-1e-9:
        a = BUDGET-invested; coins += a/min(lastc, P0); invested += a
    expx = min(lastc, P0)                              # timeout: cap upside at entry
    return (coins*expx-invested)/BUDGET*100, "timeout"


def main():
    print("Capital-management params, isolated (exit=return to entry, stop -42%)\n", flush=True)
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
            rd, tag = simulate(Hs, Ls, lastc, P0, True)
            rs, _ = simulate(Hs, Ls, lastc, P0, False)
            rows.append((ent_t, yr, rd, tag, rs))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def dd_of(sel, idx):
        ss = sorted([(x[0], x[0]+WIN, x[idx]) for x in sel])
        equity = 10000.0; op = []; eqv = []
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
        if not sel:
            print(f"{label}: لا صفقات\n"); return
        r = np.array([x[2] for x in sel]); tags = [x[3] for x in sel]; rs = np.array([x[4] for x in sel])
        wins = r[r > 0]; losses = r[r <= 0]
        nwin = sum(t == "win" for t in tags); nstop = sum(t == "stop" for t in tags); nto = sum(t == "timeout" for t in tags)
        print(f"### {label}: {len(sel)} صفقة ###")
        print(f"  عادت للدخول (ربح): {nwin} ({nwin/len(sel)*100:.0f}%)  |  وقف −42%: {nstop} ({nstop/len(sel)*100:.0f}%)  |  انتهت النافذة: {nto} ({nto/len(sel)*100:.0f}%)")
        print(f"  نسبة الربح (عوائد موجبة): {len(wins)/len(r)*100:.0f}%")
        print(f"  متوسط الربح:  {wins.mean() if len(wins) else float('nan'):+.1f}%")
        print(f"  متوسط الخسارة: {losses.mean() if len(losses) else float('nan'):+.1f}%")
        print(f"  القيمة المتوقّعة/صفقة: {r.mean():+.2f}%")
        print(f"  أقصى سحب للمحفظة: {dd_of(sel, 2):+.0f}%")
        print(f"  (للمقارنة) دخول بسيط $400 بنفس المنطق: قيمة متوقّعة {rs.mean():+.2f}%/صفقة\n")

    report([x for x in rows if x[1] == 2025], "2025 (الجفاف)")
    report([x for x in rows if x[1] == 2024], "2024 (موسم بديل)")
    report(rows, "كامل العيّنة 2022-2026")
    print("ملاحظة: الخروج = العودة للدخول (أُلغي أثر حجم الهدف). الربح كلّه من تخفيض التكلفة.")
    print("⚠️ متفائل بانحياز البقاء (الخاسرات الميتة غير ممثّلة).")
    print("\nDONE_PARAMS.", flush=True)


if __name__ == "__main__":
    main()
