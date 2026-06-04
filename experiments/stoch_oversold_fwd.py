#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAITHFUL test of the user's exact screener filter (monthly), exit-agnostic.

Corrected from the previous test, which missed the key condition. The user's
TradingView screener (monthly) is:
   Stochastic %K(14,3,3)  >  %D   (or bullish CROSS)   AND   %D(14,3,3) <= 10
i.e. stochastic turning up FROM a DEEP oversold bottom. Proper (14,3,3) smoothing.

For every monthly signal across the alt universe we measure forward returns at
+1 / +3 / +6 months and compare to the baseline (all months). If signal >>
baseline (esp. the 'explosion' >+50% rate), the deep-oversold turn-up has edge.
Run:  python experiments/stoch_oversold_fwd.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

START = "2022-06-01"; END = "2026-06-01"
DSTART = int(pd.Timestamp(START, tz="UTC").timestamp()*1000)
DEND = int(pd.Timestamp(END, tz="UTC").timestamp()*1000)


def resample_m(df):
    g = df.set_index("dt")
    return pd.DataFrame({"h": g["high"].resample("ME").max(),
                         "l": g["low"].resample("ME").min(),
                         "c": g["close"].resample("ME").last()}).dropna()


def stoch1433(h, l, c):
    ll = pd.Series(l).rolling(14).min(); hh = pd.Series(h).rolling(14).max()
    rawk = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = rawk.rolling(3).mean()            # smoothed %K (fast line)
    d = k.rolling(3).mean()               # %D (slow line)
    return k.to_numpy(), d.to_numpy()


def main():
    print("FAITHFUL test: monthly stoch %K>%D (& cross) with %D<=10, exit-agnostic\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), DSTART, DEND, log=lambda *a: None)
    print(f"  universe: {len(raw)} symbols; building monthly...", flush=True)

    H = [1, 3, 6]                          # forward horizons in months
    bucket = {tag: {h: [] for h in H} for tag in ["above", "cross", "base"]}
    n_coins = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        mo = resample_m(df)
        if len(mo) < 20:
            continue
        n_coins += 1
        k, d = stoch1433(mo["h"].to_numpy(), mo["l"].to_numpy(), mo["c"].to_numpy())
        c = mo["c"].to_numpy(); n = len(c)
        for i in range(16, n - max(H)):
            if not np.isfinite(k[i]) or not np.isfinite(d[i]):
                continue
            fwd = {h: c[i + h] / c[i] - 1 for h in H}
            for h in H:
                bucket["base"][h].append(fwd[h])
            above = (k[i] > d[i]) and (d[i] <= 10)
            cross = above and (k[i-1] <= d[i-1])
            if above:
                for h in H: bucket["above"][h].append(fwd[h])
            if cross:
                for h in H: bucket["cross"][h].append(fwd[h])
    del raw; gc.collect()

    print(f"\n##### {n_coins} alt coins #####")
    for h in H:
        print(f"\n--- forward {h} month(s) ---")
        print(f"{'set':<20}{'n':>7}{'mean':>9}{'median':>9}{'win%':>7}{'>+50%':>8}")
        for tag in ["above", "cross", "base"]:
            a = np.array(bucket[tag][h]) * 100
            if len(a) == 0:
                print(f"{tag:<20}  (none)"); continue
            label = {"above": "%K>%D & D<=10", "cross": "cross & D<=10", "base": "baseline (all)"}[tag]
            print(f"{label:<20}{len(a):>7}{a.mean():>+8.1f}%{np.median(a):>+8.1f}%"
                  f"{(a > 0).mean()*100:>6.0f}%{(a > 50).mean()*100:>7.0f}%")
    print("\nإن كان signal ≈ baseline => الفلتر لا يرفع احتمال الانفجار فوق الصدفة.")
    print("'>+50%' = نسبة الحالات التي انفجرت +50% خلال المدة.")
    print("\nDONE_OVRSLD.", flush=True)


if __name__ == "__main__":
    main()
