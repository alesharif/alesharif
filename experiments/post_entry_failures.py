#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""For BREAKOUT entries that go DOWN first (the mostly-failing group), 2024->2026-03:
on 4h data and WITHOUT a stop (full 90d), measure (1) the TRUE depth they fall to
(MAE), (2) the real rise from the new bottom (max high AFTER the bottom bar), and
(3) whether any eventually recover to breakeven/+20% if held without a stop.
Run:  python experiments/post_entry_failures.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; HOLD_MS = 90*24*3600*1000
S, E = "2023-08-01", "2026-06-01"
SIG_FROM = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp()*1000)
SIG_TO = int(pd.Timestamp("2026-03-01", tz="UTC").timestamp()*1000)


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Down-first breakout entries: true fall depth + rise-from-bottom (4h, no stop)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    maes = []; rfbs = []; recov_be = 0; recov_20 = 0; ndf = 0
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
        # 4h path arrays
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n-1):
            if not (SIG_FROM <= int(t[i]) <= SIG_TO):
                continue
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            ent_t, ent = int(t[i]), c[i]
            j0 = np.searchsorted(T, ent_t, side="right")
            j1 = np.searchsorted(T, ent_t+HOLD_MS, side="right")
            Hp = H[j0:j1]; Lp = L[j0:j1]
            if len(Lp) < 5:
                continue
            # down-first? first -5% before first +5%
            dn = np.argmax(Lp <= ent*0.95) if (Lp <= ent*0.95).any() else 10**9
            up = np.argmax(Hp >= ent*1.05) if (Hp >= ent*1.05).any() else 10**9
            if not (dn < up):
                continue
            ndf += 1
            minpos = int(np.argmin(Lp)); botlow = Lp[minpos]
            maes.append(botlow/ent - 1)
            after = Hp[minpos+1:]                      # AFTER the bottom bar (real recovery)
            rfb = (after.max()/botlow - 1) if len(after) else 0.0
            rfbs.append(rfb)
            if Hp.max() >= ent:                        # ever back to breakeven (no stop)
                recov_be += 1
            if Hp.max() >= ent*1.20:                   # ever +20% (no stop)
                recov_20 += 1
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    mae = np.array(maes)*100; rfb = np.array(rfbs)*100
    print(f"عدد الصفقات النازلة أولاً: {ndf}\n")
    print("##### إلى أين تنزل فعلاً؟ (أقصى نزول بلا وقف، 4h) #####")
    for lo, hi, lab in [(-10, 0, "حتى −10%"), (-20, -10, "−10% إلى −20%"),
                        (-40, -20, "−20% إلى −40%"), (-70, -40, "−40% إلى −70%"), (-100, -70, "أعمق من −70%")]:
        share = ((mae > lo) & (mae <= hi)).mean()*100 if lab != "حتى −10%" else (mae > -10).mean()*100
        print(f"  {lab:<18} {share:>4.0f}%")
    print(f"  متوسّط أقصى نزول: {mae.mean():+.1f}%   |   الوسيط: {np.median(mae):+.1f}%")

    print("\n##### كم ترتدّ من القاع الجديد؟ (بعد شمعة القاع، 4h) #####")
    for lo, hi, lab in [(0, 5, "ضعيف (0-5%)"), (5, 15, "5-15%"), (15, 30, "15-30%"),
                        (30, 60, "30-60%"), (60, 1e9, "أكثر من 60%")]:
        share = ((rfb > lo) & (rfb <= hi)).mean()*100
        print(f"  {lab:<14} {share:>4.0f}%")
    print(f"  متوسّط الارتداد من القاع: {rfb.mean():+.1f}%   |   الوسيط: {np.median(rfb):+.1f}%")

    print(f"\n##### لو احتفظنا بلا وقف 90 يوماً: #####")
    print(f"  عادت لنقطة الدخول (breakeven): {recov_be/ndf*100:.0f}%")
    print(f"  بلغت +20% فأكثر:               {recov_20/ndf*100:.0f}%")
    print("\nDONE_FAIL.", flush=True)


if __name__ == "__main__":
    main()
