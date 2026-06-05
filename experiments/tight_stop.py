#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User's idea: a TIGHT stop on the breakout. Since winners rise immediately and
losers crater, a tight stop should cut losers fast (small loss) without hitting
winners much. Test SL in {4,4.5,5,7,10}% with TP+100%, 4h first-touch (fine
enough for tight stops), per regime. Run: python experiments/tight_stop.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; TP = 1.0; WIN = 90*24*3600*1000
S, E, SF = "2023-08-01", "2026-06-01", "2024-01-01"
SLS = [0.04, 0.045, 0.05, 0.07, 0.10]; COST = 1.0; STOP_SLIP = 1.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Tight-stop test on breakout (TP+100, SL grid, 4h first-touch)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {sl: {"full": [], 2024: [], 2025: []} for sl in SLS}
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
            ent = c[i]; ent_t = int(t[i]); yr = int(pd.Timestamp(ent_t, unit="ms").year)
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]; Ls = L[j0:j1]; lastc = C[min(j1, len(C)-1)]
            tp_px = ent*(1+TP)
            tph = int(np.argmax(Hs >= tp_px)) if (Hs >= tp_px).any() else 10**9
            for sl in SLS:
                lb = Ls <= ent*(1-sl); slh = int(np.argmax(lb)) if lb.any() else 10**9
                if slh <= tph and slh < 10**9:
                    nr = -sl*100 - COST - STOP_SLIP
                elif tph < 10**9:
                    nr = TP*100 - COST
                else:
                    nr = (lastc/ent-1)*100 - COST
                res[sl]["full"].append(nr);
                if yr in (2024, 2025): res[sl][yr].append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"{'SL':>6}{'n':>7}{'win%':>7}{'FULL':>9}{'2024':>9}{'2025':>9}")
    print("-" * 48)
    for sl in SLS:
        a = np.array(res[sl]["full"])
        wr = (a > 0).mean()*100 if len(a) else float("nan")
        def m(k):
            x = np.array(res[sl][k]); return x.mean() if len(x) else float("nan")
        print(f"{int(sl*1000)/10:>5}%{len(a):>7}{wr:>6.0f}%{m('full'):>+8.1f}%{m(2024):>+8.1f}%{m(2025):>+8.1f}%")
    print("\nهل وقف أضيق يرفع التوقّع (يقصّ الخاسر بسرعة) أم يقصّ رابحات نزلت قليلاً أولاً؟")
    print("\nDONE_TIGHT.", flush=True)


if __name__ == "__main__":
    main()
