#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-entry behavior of BREAKOUT signals (2024 -> 2026-03): after the entry,
does price go UP first, or DOWN first? Of those that drop, how far, and how many
recover vs never rise? Answers: % straight-up, % dip-first, % dip-then-recover,
% never-rise. Daily path to exit (TP+100/SL-10, 90d). Run: python experiments/post_entry_behavior.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; TP = 1.0; SL = 0.10
S, E = "2023-08-01", "2026-06-01"
SIG_FROM = ms_from = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp()*1000)
SIG_TO = int(pd.Timestamp("2026-03-01", tz="UTC").timestamp()*1000)


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Post-entry behavior of BREAKOUT signals (2024 -> 2026-03)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    rows = []      # (up_first, mae, won, outcome)
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
        o = d["o"].to_numpy(); h = d["h"].to_numpy(); l = d["l"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        dv = c*v
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
            ent = c[i]; tp_px = ent*(1+TP); sl_px = ent*(1-SL)
            minlow = ent; up5 = dn5 = None; outcome = None; fin = ent
            for j in range(i+1, min(i+91, n)):
                if l[j] < minlow:
                    minlow = l[j]
                if up5 is None and h[j] >= ent*1.05:
                    up5 = j
                if dn5 is None and l[j] <= ent*0.95:
                    dn5 = j
                if l[j] <= sl_px:
                    outcome = "stop"; fin = sl_px; break
                if h[j] >= tp_px:
                    outcome = "tp"; fin = tp_px; break
                fin = c[j]
            if outcome is None:
                outcome = "timeout"
            up_first = (up5 is not None) and (dn5 is None or up5 <= dn5)
            mae = minlow/ent - 1
            won = (outcome == "tp") or (outcome == "timeout" and fin > ent)
            rows.append((up_first, mae, won, outcome))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    N = len(rows)
    up = [r for r in rows if r[0]]; dn = [r for r in rows if not r[0]]
    print(f"إجمالي صفقات الاختراق: {N}\n")
    print("##### الاتجاه المباشر بعد الدخول #####")
    print(f"  تصعد مباشرة (+5% قبل −5%):   {len(up)/N*100:>4.0f}%   ({len(up)})")
    print(f"  تنزل أولاً  (−5% قبل +5%):    {len(dn)/N*100:>4.0f}%   ({len(dn)})")

    print("\n##### الصفقات التي تنزل أولاً: ماذا يحصل بعدها؟ #####")
    if dn:
        rec = [r for r in dn if r[2]]; fail = [r for r in dn if not r[2]]
        print(f"  تنزل ثم ترتدّ وتربح:   {len(rec)/len(dn)*100:>4.0f}%   ({len(rec)})")
        print(f"  تنزل ولا ترتدّ (تفشل): {len(fail)/len(dn)*100:>4.0f}%   ({len(fail)})")

    print("\n##### كم تنزل؟ (أقصى نزول MAE لكل الصفقات) #####")
    mae = np.array([r[1] for r in rows])*100
    for lo, hi, lab in [(-3, 0, "نزول طفيف (0 إلى −3%)"), (-5, -3, "−3% إلى −5%"),
                        (-10, -5, "−5% إلى −10%"), (-100, -10, "ضرب الوقف (−10%)")]:
        share = ((mae > lo) & (mae <= hi)).mean()*100 if lab != "ضرب الوقف (−10%)" else (mae <= -10).mean()*100
        print(f"  {lab:<22} {share:>4.0f}%")
    print(f"  متوسّط أقصى نزول: {mae.mean():+.1f}%   |   الوسيط: {np.median(mae):+.1f}%")

    won = [r for r in rows if r[2]]
    if won:
        wd = sum(1 for r in won if not r[0])/len(won)*100
        print(f"\n##### بين الرابحات: كم نزلت أولاً قبل أن تربح؟ {wd:.0f}% #####")
    print("\nDONE_BEHAV.", flush=True)


if __name__ == "__main__":
    main()
