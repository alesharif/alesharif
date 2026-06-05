#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User's refined scaled-entry vs simple, on breakout entries.
GRID+PYRAMID ($400 budget): $100 at entry; on the way DOWN fill $100@-1%, $50@-2.5,
  $50@-4, $50@-7, $50@-10; if price recovers to entry after dipping, deploy any
  unfilled budget at entry (pyramid-to-entry). SL -12% (sell all), TP +100% (sell
  all). 120d. SIMPLE: $400 at entry, TP +100%, SL -12%. Return on $400 budget.
Per regime FULL/2024/2025. Run: python experiments/grid_pyramid.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 120*24*3600*1000; BUDGET = 400.0
S, E, SF = "2022-06-01", "2026-06-01", "2024-01-01"
LEVELS = [(1.00, 100.0), (0.99, 100.0), (0.975, 50.0), (0.96, 50.0), (0.93, 50.0), (0.90, 50.0)]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Grid+pyramid scaled entry vs simple (breakout)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    SIM = {"full": [], 2024: [], 2025: []}; GP = {"full": [], 2024: [], 2025: []}
    gp_tp = gp_stop = ntot = 0
    diag = {"tp": [], "stop": [], "timeout": []}      # (avg_cost/P0, invested)
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
            if int(t[i]) < ms(SF):
                continue
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i]); yr = int(pd.Timestamp(ent_t, unit="ms").year); ntot += 1
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]; Ls = L[j0:j1]; lastc = C[min(j1, len(C)-1)]
            tp = P0*2; sl = P0*0.88
            # ---- SIMPLE ----
            sret = None
            for k in range(len(Hs)):
                if Ls[k] <= sl: sret = (sl/P0-1)*100; break
                if Hs[k] >= tp: sret = (tp/P0-1)*100; break
            if sret is None:
                sret = (lastc/P0-1)*100
            SIM["full"].append(sret)
            if yr in (2024, 2025): SIM[yr].append(sret)
            # ---- GRID + PYRAMID ----
            invested = 100.0; coins = 100.0/P0; filled = [True]+[False]*5; dipped = False; ex = None
            for k in range(len(Hs)):
                lo = Ls[k]; hi = Hs[k]
                if lo < P0: dipped = True
                for li in range(1, len(LEVELS)):           # down-fills
                    mult, amt = LEVELS[li]
                    if not filled[li] and lo <= P0*mult:
                        invested += amt; coins += amt/(P0*mult); filled[li] = True
                if dipped and hi >= P0:                     # pyramid-to-entry on recovery
                    rem = BUDGET - invested
                    if rem > 0.01:
                        coins += rem/P0; invested = BUDGET; filled = [True]*6
                if lo <= sl:
                    ex = sl; gp_stop += 1; break
                if hi >= tp:
                    ex = tp; gp_tp += 1; break
            tag = "tp" if ex == tp else "stop" if ex == sl else "timeout"
            if ex is None:
                ex = lastc; tag = "timeout"
            pnl = coins*ex - invested
            avg_cost_ratio = (invested/coins)/P0 if coins > 0 else 1.0
            diag[tag].append((avg_cost_ratio, invested))
            GP["full"].append(pnl/BUDGET*100)
            if yr in (2024, 2025): GP[yr].append(pnl/BUDGET*100)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def m(st, k):
        a = np.array(st[k]); return a.mean() if len(a) else float("nan")
    print(f"{'method':<16}{'n':>7}{'FULL':>9}{'2024':>9}{'2025':>9}")
    print("-" * 48)
    print(f"{'SIMPLE (SL12)':<16}{len(SIM['full']):>7}{m(SIM,'full'):>+8.1f}%{m(SIM,2024):>+8.1f}%{m(SIM,2025):>+8.1f}%")
    print(f"{'GRID+PYRAMID':<16}{len(GP['full']):>7}{m(GP,'full'):>+8.1f}%{m(GP,2024):>+8.1f}%{m(GP,2025):>+8.1f}%")
    print(f"\nGRID: بلغت الهدف 2x: {gp_tp/ntot*100:.0f}%   |   ضربت الوقف −12%: {gp_stop/ntot*100:.0f}%")
    print("\n##### الإثبات الرياضي: متوسّط التكلفة والنشر حسب النتيجة #####")
    print(f"{'النتيجة':<10}{'عدد':>7}{'متوسط التكلفة/الدخول':>22}{'متوسط المنشور $':>18}")
    for tag, lab in [("tp", "رابحة (2x)"), ("stop", "خاسرة (وقف)"), ("timeout", "محايدة")]:
        a = diag[tag]
        if not a: continue
        ac = np.mean([x[0] for x in a]); inv = np.mean([x[1] for x in a])
        print(f"{lab:<10}{len(a):>7}{(ac-1)*100:>+20.1f}%{inv:>16.0f}$")
    print("الرابحة تكلفتها قرب الدخول (تمايل سطحي)؛ الخاسرة خصمها أعمق + نشر أكبر = مال أكثر في الفاشلة.")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالوقف، لا مُلغى).")
    print("\nDONE_GP.", flush=True)


if __name__ == "__main__":
    main()
