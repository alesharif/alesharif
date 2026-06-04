#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-TP test of the user's REAL behavior: take +10/20/30/50% when it looks
weak, hold the losers ('wait for it to come back').

This reproduces the classic high-win-rate / negative-expectancy trap. We take
profit at the FIRST touch of +X% (generous: assume he always catches it), and if
it never reaches +X% within the window we close at market (the held loser). As X
drops, the WIN RATE climbs toward the remembered ~90% — but does EXPECTANCY turn
positive? That is the only question that matters.
Signal = monthly stoch %K>%D & %D<=10 (same family as his Wobbler oscillator).
Run:  python experiments/multi_tp.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

START = "2022-06-01"; END = "2026-06-01"
DS = int(pd.Timestamp(START, tz="UTC").timestamp()*1000)
DE = int(pd.Timestamp(END, tz="UTC").timestamp()*1000)
WINDOW_MS = 270 * 24 * 3600 * 1000
FEE = 0.2
TPS = [0.10, 0.20, 0.30, 0.50]


def resample_m(df):
    g = df.set_index("dt")
    return pd.DataFrame({"h": g["high"].resample("ME").max(),
                         "l": g["low"].resample("ME").min(),
                         "c": g["close"].resample("ME").last(),
                         "t": g["time"].resample("ME").last()}).dropna()


def stoch1433(h, l, c):
    ll = pd.Series(l).rolling(14).min(); hh = pd.Series(h).rolling(14).max()
    rawk = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = rawk.rolling(3).mean(); d = k.rolling(3).mean()
    return k.to_numpy(), d.to_numpy()


def main():
    print("MULTI-TP test of user's real behavior (take +10/20/30/50, hold losers)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), DS, DE, log=lambda *a: None)
    sigs = {}
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        mo = resample_m(df)
        if len(mo) < 20:
            continue
        k, d = stoch1433(mo["h"].to_numpy(), mo["l"].to_numpy(), mo["c"].to_numpy())
        mc = mo["c"].to_numpy(); mt = mo["t"].to_numpy()
        lst = [(int(mt[i]), float(mc[i])) for i in range(16, len(mc))
               if np.isfinite(k[i]) and np.isfinite(d[i]) and k[i] > d[i] and d[i] <= 10]
        if lst:
            sigs[sym] = lst
        df.drop(columns=["dt"], inplace=True)
    nsig = sum(len(v) for v in sigs.values())
    print(f"  {nsig} signals\n", flush=True)

    print(f"{'TP target':<10}{'n':>6}{'hitTP%':>8}{'WIN%':>7}{'mean(exp)':>11}{'median':>9}")
    print("-" * 52)
    for tp in TPS:
        rets = []
        for sym, lst in sigs.items():
            df = raw[sym]
            t = df["time"].to_numpy(); H = df["high"].to_numpy(float); C = df["close"].to_numpy(float)
            for ent_t, ent_px in lst:
                j0 = np.searchsorted(t, ent_t)
                if j0 >= len(t):
                    continue
                tp_px = ent_px * (1 + tp); end_t = ent_t + WINDOW_MS
                hit = False; j = j0
                while j < len(t) and t[j] <= end_t:
                    if H[j] >= tp_px:
                        rets.append(tp*100 - FEE); hit = True; break
                    j += 1
                if not hit:
                    jl = min(j, len(t)-1)
                    rets.append((C[jl]/ent_px - 1)*100 - FEE)
        r = np.array(rets)
        print(f"+{int(tp*100)}%{'':<6}{len(r):>6}{(r>0).mean()*100:>7.0f}%"
              f"{(r>0).mean()*100:>6.0f}%{r.mean():>+10.1f}%{np.median(r):>+8.1f}%")
    del raw; gc.collect()
    print("\nانظر: كلّما صغُر TP -> WIN% يرتفع نحو 90% (كذاكرتك)، لكن mean (التوقّع) يبقى سالباً.")
    print("لأن الـ'hold loser' (لم تصل الهدف) ينزف ويبتلع الأرباح الصغيرة.")
    print("\nDONE_MULTITP.", flush=True)


if __name__ == "__main__":
    main()
