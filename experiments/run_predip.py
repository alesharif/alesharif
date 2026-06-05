#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""For breakout entries that REACH +50% (the runners), how much do they dip BEFORE
launching? Max adverse excursion from entry up to the bar that first hits +50%
(4h). Calibrates the stop: if runners rarely dip beyond -X%, a stop just below X
preserves them. Full sample 2022-09 -> 2026-06. Run: python experiments/run_predip.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 90*24*3600*1000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
TARGET = 0.50


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print(f"Pre-launch dip for breakout RUNNERS (reach +{int(TARGET*100)}%), 4h\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    predips = []; n_entries = 0
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
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float)
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
            n_entries += 1
            ent = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]; Ls = L[j0:j1]
            tgt = ent*(1+TARGET)
            hit = np.argmax(Hs >= tgt) if (Hs >= tgt).any() else -1
            if hit < 0:
                continue                            # not a runner
            predip = Ls[:hit+1].min()/ent - 1       # worst dip before reaching +50%
            predips.append(predip*100)
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    p = np.array(predips)
    print(f"إجمالي إشارات الاختراق: {n_entries}")
    print(f"التي وصلت +{int(TARGET*100)}% (الرابضة): {len(p)} ({len(p)/n_entries*100:.0f}%)\n")
    print("##### كم نزلت الرابضة قبل أن تصل +50%؟ (أقصى نزول) #####")
    for lo, hi, lab in [(-2, 0, "لم تنزل تقريباً (0 إلى −2%)"), (-4, -2, "−2% إلى −4%"),
                        (-6, -4, "−4% إلى −6%"), (-10, -6, "−6% إلى −10%"),
                        (-20, -10, "−10% إلى −20%"), (-100, -20, "أعمق من −20%")]:
        share = ((p > lo) & (p <= hi)).mean()*100
        print(f"  {lab:<26} {share:>4.0f}%")
    print(f"\n  متوسّط أقصى نزول قبل الإقلاع: {p.mean():+.1f}%   |   الوسيط: {np.median(p):+.1f}%")
    for thr in [2, 3, 4.5, 5, 7, 10]:
        print(f"  وقف −{thr}% كان سيُبقي: {(p > -thr).mean()*100:>4.0f}% من الرابضة")
    print("\nDONE_PREDIP.", flush=True)


if __name__ == "__main__":
    main()
