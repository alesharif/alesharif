#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detail of the 1-3 WEEKS bottom bucket (168-504h) for breakout runners (+50%).
For each runner whose bottom falls 1-3 weeks after entry: symbol, entry date,
dip%, days-to-bottom. Plus sub-distributions of dip-depth and duration within
the bucket. Run:  python experiments/dip_bucket_1_3w.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; WIN = 90*24*3600*1000; TARGET = 0.50
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
LO_H, HI_H = 168, 504        # 1-3 weeks in hours


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("تفاصيل فئة 1-3 أسابيع (القاع بين 7 و21 يوماً)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    total_runners = 0; rec = []        # (sym, ent_t, dip%, hours)
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
            ent = c[i]; ent_t = int(t[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+WIN, side="right")
            Hs = H[j0:j1]; Ls = L[j0:j1]; Ts = T[j0:j1]
            if not (Hs >= ent*(1+TARGET)).any():
                continue
            total_runners += 1
            hit = int(np.argmax(Hs >= ent*(1+TARGET)))
            seg = Ls[:hit+1]; botpos = int(np.argmin(seg))
            hours = (int(Ts[botpos]) - ent_t)/3600000.0
            if LO_H < hours <= HI_H:
                rec.append((sym, ent_t, (seg[botpos]/ent-1)*100, hours))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    nbk = len(rec)
    print(f"إجمالي الرابضين (+50%): {total_runners}")
    print(f"في فئة 1-3 أسابيع: {nbk}  ({nbk/total_runners*100:.0f}% من الرابضين)\n")
    dips = np.array([r[2] for r in rec]); hrs = np.array([r[3] for r in rec])
    print(f"وسيط النزول {np.median(dips):+.1f}%  |  متوسط النزول {dips.mean():+.1f}%")
    print(f"وسيط المدة {np.median(hrs)/24:.1f} يوم  |  متوسط المدة {hrs.mean()/24:.1f} يوم\n")

    print("##### توزيع عمق النزول داخل الفئة #####")
    for lo, hi, lab in [(-6, 0, "0 إلى −6%"), (-8, -6, "−6 إلى −8%"), (-10, -8, "−8 إلى −10%"),
                        (-13, -10, "−10 إلى −13%"), (-20, -13, "−13 إلى −20%"), (-100, -20, "أعمق من −20%")]:
        m = (dips > lo) & (dips <= hi); print(f"  {lab:<14} {m.sum():>4} صفقة  ({m.mean()*100:>3.0f}%)")
    print("\n##### توزيع المدة داخل الفئة #####")
    for lo, hi, lab in [(168, 240, "7-10 أيام"), (240, 336, "10-14 يوم"), (336, 504, "14-21 يوم")]:
        m = (hrs > lo) & (hrs <= hi); print(f"  {lab:<12} {m.sum():>4} صفقة  ({m.mean()*100:>3.0f}%)")

    print("\n##### قائمة الصفقات (مرتّبة حسب الأعمق نزولاً) #####")
    print(f"{'العملة':<14}{'التاريخ':<12}{'النزول':>8}{'الأيام':>8}")
    for sym, ent_t, dip, h in sorted(rec, key=lambda r: r[2]):
        dt = pd.Timestamp(ent_t, unit="ms").strftime("%Y-%m-%d")
        print(f"{sym:<14}{dt:<12}{dip:>+7.1f}%{h/24:>7.1f}")
    print("\nDONE_BK.", flush=True)


if __name__ == "__main__":
    main()
