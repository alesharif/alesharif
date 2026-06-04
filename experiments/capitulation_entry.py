#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capitulation-as-ENTRY (forward/precision test). Buy the falling knife at panic:
  4h bar with volume > 3x its 30-bar avg, AND RSI(14,4h) < 25, AND price fell
  >15% over the last ~5 days (30 4h bars), liquid. Cooldown 5 days per coin.
Exit = first touch TP/SL (60d). Measures PRECISION (% reaching TP) + expectancy
per regime. NOTE: heavy survivorship — coins that capitulate then DIE are absent.
Run:  python experiments/capitulation_entry.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; S, E, SF = "2023-08-01", "2026-06-01", "2024-01-01"
WIN = 60*24*3600*1000; COST = 1.0; STOP_SLIP = 1.0
TPS = [0.15, 0.30, 0.50]; SLS = [0.10, 0.15]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("Capitulation-as-entry: vol>3x + RSI<25 + fell>15%/5d (4h). Precision + expectancy\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    res = {(tp, sl): {"full": [], 2024: [], 2025: []} for tp in TPS for sl in SLS}
    nsig = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        t = df["time"].to_numpy(); o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float); c = df["close"].to_numpy(float); v = df["volume"].to_numpy(float)
        n = len(c)
        if n < 120:
            continue
        r = rsi(c, 14); vma = pd.Series(v).rolling(30).mean().to_numpy()
        dvma = pd.Series(c*v).rolling(180).mean().to_numpy()    # ~30d $vol on 4h
        last = -10**9
        for i in range(40, n-1):
            if int(t[i]) < ms(SF):
                continue
            if i - last < 30:                                   # cooldown 5 days
                continue
            if not (np.isfinite(vma[i]) and vma[i] > 0 and v[i] > 3*vma[i]
                    and r[i] < 25 and (c[i]/c[i-30]-1) < -0.15):
                continue
            if not np.isfinite(dvma[i]) or dvma[i] <= LIQ_MIN:
                continue
            last = i; nsig += 1
            ent = c[i]; yr = int(pd.Timestamp(int(t[i]), unit="ms").year)
            end = min(i+1+int(np.searchsorted(t[i:], int(t[i])+WIN)), n)
            Hs = h[i+1:end]; Ls = l[i+1:end]; lastc = c[end-1] if end > i+1 else ent
            for tp in TPS:
                hb = Hs >= ent*(1+tp); tph = int(np.argmax(hb)) if hb.any() else 10**9
                for sl in SLS:
                    lb = Ls <= ent*(1-sl); slh = int(np.argmax(lb)) if lb.any() else 10**9
                    if slh <= tph and slh < 10**9:
                        nr = -sl*100 - COST - STOP_SLIP
                    elif tph < 10**9:
                        nr = tp*100 - COST
                    else:
                        nr = (lastc/ent-1)*100 - COST
                    res[(tp, sl)]["full"].append(nr)
                    if yr in (2024, 2025):
                        res[(tp, sl)][yr].append(nr)
    del raw; gc.collect()

    print(f"  {nsig} capitulation signals\n")
    print(f"{'TP':>4}{'SL':>5}{'n':>7}{'precision(TP%)':>15}{'FULL':>8}{'2024':>8}{'2025':>8}")
    print("-" * 56)
    for tp in TPS:
        for sl in SLS:
            a = np.array(res[(tp, sl)]["full"])
            prec = (np.isclose(a, tp*100-COST)).mean()*100 if len(a) else float("nan")
            def m(k):
                x = np.array(res[(tp, sl)][k]); return x.mean() if len(x) else float("nan")
            mark = " *" if m("full") > 0 else ""
            print(f"{int(tp*100):>3}%{int(sl*100):>4}%{len(a):>7}{prec:>13.0f}%{m('full'):>+7.1f}%{m(2024):>+7.1f}%{m(2025):>+7.1f}%{mark}")
    print("\nprecision = نسبة بلوغ الهدف (هل هو القاع فعلاً؟). موجب => إشارة دخول رابحة.")
    print("⚠️ متفائل بانحياز البقاء (العملات التي استسلمت ثم ماتت غائبة).")
    print("\nDONE_CAP.", flush=True)


if __name__ == "__main__":
    main()
