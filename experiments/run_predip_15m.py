#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-launch dip for breakout RUNNERS at 15m precision + TIMING. For entries that
reach +50%, load 15m from entry to the +50% bar, find the deepest dip and HOW LONG
after entry it occurs. Run: python experiments/run_predip_15m.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

LIQ_MIN = 300_000; WIN = 90*24*3600*1000; TARGET = 0.50
S, E, SF = "2022-06-01", "2026-06-01", "2024-01-01"


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Pre-launch dip + TIMING for breakout runners (15m), 2024->2026\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    # collect runner entries first (sym, ent_t, ent_px, tp_time)
    runners = []
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
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float)
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
            ent = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]
            hit = np.argmax(Hs >= ent*(1+TARGET)) if (Hs >= ent*(1+TARGET)).any() else -1
            if hit < 0:
                continue
            runners.append((sym, ent_t, ent, int(T[j0+hit])))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()
    print(f"  {len(runners)} runners; loading 15m for pre-launch windows...", flush=True)

    dips = []; times_h = []; done = 0
    for sym, ent_t, ent, tp_t in runners:
        done += 1
        if done % 200 == 0:
            print(f"   {done}/{len(runners)}...", flush=True)
        m15 = HR.load_range(sym, "15m", ent_t, tp_t)
        if m15 is None or len(m15) < 3:
            continue
        lo = m15["low"].to_numpy(float); tm = m15["time"].to_numpy()
        botpos = int(np.argmin(lo)); botlow = lo[botpos]
        dips.append((botlow/ent - 1)*100)
        times_h.append((int(tm[botpos]) - ent_t)/3600000.0)
    gc.collect()

    p = np.array(dips); th = np.array(times_h)
    print(f"\nرابضون بـ15m: {len(p)}\n")
    print("##### كم نزلت قبل الإقلاع؟ (15m) #####")
    for lo, hi, lab in [(-2, 0, "0 إلى −2%"), (-4, -2, "−2 إلى −4%"), (-6, -4, "−4 إلى −6%"),
                        (-10, -6, "−6 إلى −10%"), (-20, -10, "−10 إلى −20%"), (-100, -20, "أعمق من −20%")]:
        print(f"  {lab:<14} {((p>lo)&(p<=hi)).mean()*100:>4.0f}%")
    print(f"  المتوسّط {p.mean():+.1f}%  |  الوسيط {np.median(p):+.1f}%")
    print("\n##### متى تحدث القاع (الوقت بعد الدخول)؟ #####")
    for lo, hi, lab in [(0, 6, "خلال 6 ساعات"), (6, 24, "6-24 ساعة"), (24, 72, "1-3 أيام"),
                        (72, 168, "3-7 أيام"), (168, 1e9, "أكثر من أسبوع")]:
        print(f"  {lab:<14} {((th>lo)&(th<=hi)).mean()*100:>4.0f}%")
    print(f"  متوسّط الوقت للقاع: {th.mean():.0f} ساعة  |  الوسيط {np.median(th):.0f} ساعة")
    print("\n##### الوقف وما يُبقيه (15m) #####")
    for thr in [3, 4.5, 5, 7, 10]:
        print(f"  وقف −{thr}% يُبقي: {(p>-thr).mean()*100:>4.0f}%")
    print("\nDONE_PD15.", flush=True)


if __name__ == "__main__":
    main()
