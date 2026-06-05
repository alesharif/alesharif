#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Among capitulation entries (vol>3x + RSI<25 + fell>15%/5d), what distinguishes
WINNERS (bounce +15%) from LOSERS (drop -15%)? Compare median features at entry:
volume-climax size, RSI, recent drop depth, lower-wick (rejection), liquidity,
distance below EMA50. Then split each feature at its median -> win-rate lift.
Run:  python experiments/capitulation_discriminator.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; S, E, SF = "2023-08-01", "2026-06-01", "2024-01-01"
WIN = 60*24*3600*1000; TP = 0.15; SL = 0.15


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


FEATS = ["vol_ratio", "rsi", "drop5d", "drop20d", "lower_wick", "log_$vol", "ext_EMA50"]


def main():
    print("Capitulation: what distinguishes WINNERS (+15%) from LOSERS (-15%)?\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    W = []; L = []          # feature dicts for winners / losers
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        t = df["time"].to_numpy(); o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float); v = df["volume"].to_numpy(float)
        n = len(c)
        if n < 160:
            continue
        r = rsi(c, 14); vma = pd.Series(v).rolling(30).mean().to_numpy()
        dvma = pd.Series(c*v).rolling(180).mean().to_numpy(); e50 = ema(c, 50)
        last = -10**9
        for i in range(130, n-1):
            if int(t[i]) < ms(SF) or i-last < 30:
                continue
            if not (np.isfinite(vma[i]) and vma[i] > 0 and v[i] > 3*vma[i]
                    and r[i] < 25 and (c[i]/c[i-30]-1) < -0.15):
                continue
            if not np.isfinite(dvma[i]) or dvma[i] <= LIQ_MIN:
                continue
            last = i; ent = c[i]; rng = max(h[i]-lo[i], 1e-12)
            feat = {"vol_ratio": v[i]/vma[i], "rsi": r[i], "drop5d": (c[i]/c[i-30]-1)*100,
                    "drop20d": (c[i]/c[i-120]-1)*100, "lower_wick": (min(o[i], c[i])-lo[i])/rng,
                    "log_$vol": np.log10(dvma[i]), "ext_EMA50": (c[i]-e50[i])/e50[i]*100}
            # outcome TP15/SL15 first touch
            end = min(i+1+int(np.searchsorted(t[i:], int(t[i])+WIN)), n)
            Hs = h[i+1:end]; Ls = lo[i+1:end]
            tph = int(np.argmax(Hs >= ent*(1+TP))) if (Hs >= ent*(1+TP)).any() else 10**9
            slh = int(np.argmax(Ls <= ent*(1-SL))) if (Ls <= ent*(1-SL)).any() else 10**9
            if tph == 10**9 and slh == 10**9:
                continue                      # undecided -> skip for clean comparison
            (W if tph < slh else L).append(feat)
    del raw; gc.collect()

    print(f"رابحات: {len(W)}   |   خاسرات: {len(L)}\n")
    print(f"{'feature':<12}{'WINNERS(med)':>14}{'LOSERS(med)':>13}{'الفرق':>9}")
    print("-" * 50)
    for f in FEATS:
        wm = np.median([d[f] for d in W]); lm = np.median([d[f] for d in L])
        print(f"{f:<12}{wm:>13.2f}{lm:>13.2f}{wm-lm:>+9.2f}")

    print(f"\n##### win-rate حسب نصفي كل خاصية (فوق/تحت الوسيط) #####")
    alld = [(d, 1) for d in W] + [(d, 0) for d in L]
    for f in FEATS:
        med = np.median([d[f] for d, _ in alld])
        hi = [w for d, w in alld if d[f] > med]; loo = [w for d, w in alld if d[f] <= med]
        print(f"{f:<12} فوق الوسيط: {np.mean(hi)*100:>4.0f}%   تحت الوسيط: {np.mean(loo)*100:>4.0f}%")
    print("\nفرق كبير في خاصية = بصمة تميّز الرابح من الخاسر (نفلتر بها).")
    print("\nDONE_DISC.", flush=True)


if __name__ == "__main__":
    main()
