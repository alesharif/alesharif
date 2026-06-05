#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast (cached 4h) pre-launch dip + TIMING for breakout runners (reach +50%).
Dip = min low before the +50% bar; timing = hours after entry until that bottom.
Run: python experiments/run_predip_timing.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 90*24*3600*1000; TARGET = 0.50
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Pre-launch dip + TIMING for breakout runners (4h cached), full sample\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    dips = []; th = []; nent = 0
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
            nent += 1
            ent = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]; Ls = L[j0:j1]; Ts = T[j0:j1]
            hit = np.argmax(Hs >= ent*(1+TARGET)) if (Hs >= ent*(1+TARGET)).any() else -1
            if hit < 0:
                continue
            seg = Ls[:hit+1]; botpos = int(np.argmin(seg))
            dips.append((seg[botpos]/ent - 1)*100)
            th.append((int(Ts[botpos]) - ent_t)/3600000.0)        # hours to bottom
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    p = np.array(dips); h = np.array(th)
    print(f"إشارات: {nent}   |   رابضون (+50%): {len(p)} ({len(p)/nent*100:.0f}%)\n")
    print("##### كم نزلت قبل الإقلاع؟ #####")
    for lo, hi, lab in [(-2, 0, "0 إلى −2%"), (-4, -2, "−2 إلى −4%"), (-6, -4, "−4 إلى −6%"),
                        (-10, -6, "−6 إلى −10%"), (-20, -10, "−10 إلى −20%"), (-100, -20, "أعمق من −20%")]:
        print(f"  {lab:<14} {((p>lo)&(p<=hi)).mean()*100:>4.0f}%")
    print(f"  الوسيط {np.median(p):+.1f}%")
    print("\n##### جدول القاع: التوقيت + عمق النزول #####")
    print(f"{'متى القاع':<18}{'النسبة':>8}{'وسيط النزول':>14}{'متوسط النزول':>14}")
    for lo, hi, lab in [(0, 8, "خلال 8 ساعات"), (8, 24, "8-24 ساعة"), (24, 72, "1-3 أيام"),
                        (72, 168, "3-7 أيام"), (168, 504, "1-3 أسابيع"), (504, 1e9, "أكثر من 3 أسابيع")]:
        mask = (h > lo) & (h <= hi)
        share = mask.mean()*100
        md = np.median(p[mask]) if mask.any() else float("nan")
        mn = p[mask].mean() if mask.any() else float("nan")
        print(f"{lab:<18}{share:>7.0f}%{md:>+13.1f}%{mn:>+13.1f}%")
    print(f"  متوسّط الوقت للقاع: {h.mean()/24:.1f} يوم  |  الوسيط {np.median(h)/24:.1f} يوم")
    print("\n##### تقاطع: القاع المبكّر (أوّل يومين) مقابل المتأخّر — كم العمق؟ #####")
    early = p[h <= 48]; late = p[h > 48]
    print(f"  قاع خلال يومين ({len(early)}): وسيط نزول {np.median(early):+.1f}%")
    print(f"  قاع بعد يومين ({len(late)}): وسيط نزول {np.median(late):+.1f}%")
    print("\nDONE_PDT.", flush=True)


if __name__ == "__main__":
    main()
