#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPTIMIZE the $400 capital distribution INSIDE the trade (entry/exit UNCHANGED).

Per the user's clarification:
  * original strategy = squeeze entry, exit = first touch of +50% (TP). Unchanged.
  * the dip values are WINNERS: max drawdown before they rebounded to the TP.
  * task = find the best split of $400 across down-levels [0,-8,-16,-24,-32] PLUS a
    rebound-completion mechanism (deploy the remainder as price recovers, so the
    full $400 is in before price returns to entry).

We test several down-distributions. For each we report:
  WINNERS (reach +50%): avg return on $400 — this is what the user's framing optimizes.
  FULL  (winners+losers): honest avg + worst trades + portfolio — what live trading gives.
No tight stop (DCA must hold through the dip); non-winners marked to market at 120d.
Run:  python experiments/dca_optimize.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0; TGT = 0.50
S, E = "2022-06-01", "2026-06-01"; W0 = 2025
LV = [0.0, 0.08, 0.16, 0.24, 0.32]      # down levels (fraction below entry)
REB = [0.03, 0.06, 0.09]                # rebound steps from bottom (each deploys 1/3 of remainder)

SCHEMES = {
    "الملف [40-60-100-100-100]": [40, 60, 100, 100, 100],
    "بسيط [400-0-0-0-0]":        [400, 0, 0, 0, 0],
    "أمامي [250-75-40-20-15]":   [250, 75, 40, 20, 15],
    "متساوٍ [80×5]":             [80, 80, 80, 80, 80],
    "خلفي [20-40-80-120-140]":   [20, 40, 80, 120, 140],
    "خلفي-معتدل [40-80-100-90-90]": [40, 80, 100, 90, 90],
    "نصف-دخول [200-50-50-50-50]": [200, 50, 50, 50, 50],
}


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def simulate(Hs, Ls, lastc, P0, dist):
    """dist = $ at each of LV. Rebound deploys the remainder (1/3 of remaining at
    +3/6/9% from bottom; rest near entry). Exit at +50% over entry."""
    exitpx = P0*(1+TGT)
    invested = 0.0; coins = 0.0; dfill = [False]*5; rdone = [False]*3; near = False
    dipped = False; bottom = P0
    # level 0 (entry) fills immediately
    if dist[0] > 0:
        coins += dist[0]/P0; invested += dist[0]; dfill[0] = True
    rem_after_down = BUDGET - sum(dist)          # (should be ~0; rebound handles leftover)
    for k in range(len(Hs)):
        lo = Ls[k]; hi = Hs[k]
        if lo < bottom:
            bottom = lo; rdone = [False]*3
        if lo < P0:
            dipped = True
        for di in range(1, 5):                    # down-fills
            if not dfill[di] and dist[di] > 0 and lo <= P0*(1-LV[di]) and invested < BUDGET-1e-9:
                price = P0*(1-LV[di]); a = min(dist[di], BUDGET-invested)
                coins += a/price; invested += a; dfill[di] = True
        if dipped and invested < BUDGET-1e-9:     # rebound completion of the remainder
            rebpct = hi/bottom - 1
            for ri in range(3):
                if not rdone[ri] and rebpct >= REB[ri] and invested < BUDGET-1e-9:
                    fp = bottom*(1+REB[ri]); a = min((BUDGET-invested)/ (3-ri), BUDGET-invested)
                    coins += a/fp; invested += a; rdone[ri] = True
            if not near and hi >= P0*0.98 and invested < BUDGET-1e-9:
                fp = P0*0.98; a = BUDGET-invested; coins += a/fp; invested += a; near = True
        if hi >= exitpx:
            return (coins*exitpx-invested)/BUDGET*100, True, invested
    # not a winner: deploy any leftover at last price (kept simple) and mark to market
    if invested < BUDGET-1e-9:
        a = BUDGET-invested; coins += a/lastc; invested += a
    return (coins*lastc-invested)/BUDGET*100, False, invested


def main():
    print("Optimize $400 distribution inside the trade (entry/exit fixed, TP +50%)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    trades = []        # (ent_t, yr, Hs, Ls, lastc, P0)
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
            trades.append((ent_t, yr, H[j0:j1].copy(), L[j0:j1].copy(), C[min(j1, len(C)-1)], P0))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()
    print(f"إجمالي صفقات الانقباض: {len(trades)}\n")

    def evalscheme(dist, yr=None):
        win_r = []; full_r = []
        ss = []
        for ent_t, y, Hs, Ls, lastc, P0 in trades:
            if yr is not None and y != yr:
                continue
            r, won, inv = simulate(Hs, Ls, lastc, P0, dist)
            full_r.append(r); ss.append((ent_t, ent_t+WIN, r))
            if won:
                win_r.append(r)
        full_r = np.array(full_r); win_r = np.array(win_r)
        # portfolio
        equity = 10000.0; op = []; eqv = []
        for et, xt, rr in sorted(ss):
            op.sort()
            while op and op[0][0] <= et:
                _, p = op.pop(0); equity += p; eqv.append(equity)
            if len(op) >= 6:
                continue
            stake = equity*0.10; equity -= stake; op.append((xt, stake*(1+rr/100)))
        for xt, p in sorted(op):
            equity += p; eqv.append(equity)
        port = (eqv[-1]/10000-1)*100 if eqv else float("nan")
        wr = len(win_r)/len(full_r)*100 if len(full_r) else float("nan")
        wavg = win_r.mean() if len(win_r) else float("nan")
        favg = full_r.mean() if len(full_r) else float("nan")
        worst = np.sort(full_r)[:3] if len(full_r) else []
        return wr, wavg, favg, port, worst

    for scope, yr in [("2025 (الجفاف)", 2025), ("كامل العيّنة", None)]:
        print(f"========== {scope} ==========")
        print(f"{'التوزيع':<30}{'رابحات%':>8}{'ربح الرابحة':>12}{'متوسط الكل':>12}{'محفظة':>9}{'أسوأ3':>20}")
        print("-"*91)
        best = None
        for name, dist in SCHEMES.items():
            wr, wavg, favg, port, worst = evalscheme(dist, yr)
            w3 = ",".join(f"{x:+.0f}" for x in worst)
            if best is None or favg > best[0]:
                best = (favg, name)
            print(f"{name:<30}{wr:>7.0f}%{wavg:>+11.1f}%{favg:>+11.1f}%{port:>+8.0f}%{w3:>20}")
        print(f"الأفضل (متوسط الكل): {best[1]}  ({best[0]:+.1f}%)\n")
    print("ملاحظة أمينة: «ربح الرابحة» يتحسّن بالتحميل الخلفي (شراء أعمق)، لكن نفس التحميل")
    print("الخلفي يدمّر «متوسط الكل» لأنه يضخّ كامل $400 في الخاسرات الهابطة. لا نعرف الرابحة مسبقاً.")
    print("⚠️ متفائل بانحياز البقاء.")
    print("\nDONE_OPT.", flush=True)


if __name__ == "__main__":
    main()
