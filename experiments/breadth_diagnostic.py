#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User's breadth diagnostic: is the OPPORTUNITY DROUGHT predictable from broad
market breadth? At the START of each month, measure what % of liquid coins are in
an uptrend over 60 / 99 / 120 days (price now > price N days ago). Then compare
that breadth on the developed system's WINNING vs LOSING months — and see which
lookback best separates them. If winning months start with high breadth and
losing months with low breadth, the drought IS predictable -> a breadth filter
can target the root cause. In-sample. Run:  python experiments/breadth_diagnostic.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

SL = 0.10; TP = 1.0; WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
LOOKBACKS = [60, 99, 120]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Breadth diagnostic: market uptrend% (60/99/120) on winning vs losing months\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)

    # month-start timestamps
    mbs = pd.date_range(pd.Timestamp(SF, tz="UTC").normalize().replace(day=1),
                        pd.Timestamp(E, tz="UTC"), freq="MS")
    mb_ms = [int(x.timestamp()*1000) for x in mbs]
    mb_key = [x.strftime("%Y-%m") for x in mbs]
    # breadth counters per month: up[N], total
    up = {k: {N: 0 for N in LOOKBACKS} for k in mb_key}
    tot = {k: 0 for k in mb_key}
    pnl = {}          # month -> list of trade net returns (developed system)

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
        dv = c*v
        # ---- breadth contribution at each month start ----
        for mk, mbt in zip(mb_key, mb_ms):
            idx = np.searchsorted(t, mbt, side="left") - 1
            if idx < max(LOOKBACKS) or idx >= n:
                continue
            if np.nanmean(dv[max(0, idx-30):idx]) <= LIQ_MIN:
                continue
            tot[mk] += 1
            for N in LOOKBACKS:
                if c[idx] > c[idx-N]:
                    up[mk][N] += 1
        # ---- developed-system trades for month classification ----
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float)
        L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n):
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and int(t[i]) >= ms(SF) and c[i] > e200[i]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            ent_t, ent_px = int(t[i]), float(c[i]); j0 = np.searchsorted(T, ent_t)
            if j0 >= len(T):
                continue
            sl = ent_px*(1-SL); tp = ent_px*(1+TP); end_t = ent_t+WINDOW_MS
            r = None; j = j0
            while j < len(T) and T[j] <= end_t:
                if L[j] <= sl: r = (-SL*100, True); break
                if H[j] >= tp: r = (TP*100, False); break
                j += 1
            if r is None:
                r = ((C[min(j, len(T)-1)]/ent_px-1)*100, False)
            nr = r[0] - 1.0 - (1.0 if r[1] else 0.0)
            pnl.setdefault(pd.Timestamp(ent_t, unit="ms").strftime("%Y-%m"), []).append(nr)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    # breadth per month + classification
    print(f"{'month':<9}{'win?':>6}{'mean%':>8}" + "".join(f"{'br'+str(N):>7}" for N in LOOKBACKS))
    win_b = {N: [] for N in LOOKBACKS}; los_b = {N: [] for N in LOOKBACKS}
    for mk in mb_key:
        if tot[mk] == 0 or mk not in pnl:
            continue
        mean = np.mean(pnl[mk]); iswin = mean > 0
        brs = {N: up[mk][N]/tot[mk]*100 for N in LOOKBACKS}
        for N in LOOKBACKS:
            (win_b if iswin else los_b)[N].append(brs[N])
        print(f"{mk:<9}{('WIN' if iswin else 'lose'):>6}{mean:>+7.1f}%"
              + "".join(f"{brs[N]:>6.0f}%" for N in LOOKBACKS))

    print(f"\n##### avg market breadth: WINNING vs LOSING months #####")
    print(f"{'lookback':<10}{'WIN months':>12}{'LOSE months':>13}{'gap':>8}")
    for N in LOOKBACKS:
        w = np.mean(win_b[N]) if win_b[N] else float("nan")
        l = np.mean(los_b[N]) if los_b[N] else float("nan")
        print(f"{str(N)+'d up%':<10}{w:>11.0f}%{l:>12.0f}%{w-l:>+7.0f}%")
    print("\nأكبر gap = الفلتر الذي يميّز الشهر الرابح من الخاسر => أفضل مقياس اتّساع.")
    print("إن كان اتّساع الأشهر الرابحة >> الخاسرة => الجفاف متنبّأ به ويمكن فلترته.")
    print("\nDONE_BREADTH.", flush=True)


if __name__ == "__main__":
    main()
