#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the ENTRY threshold above EMA7200 (the user's idea: enter later, at +20%, since
winners reach >20% MFE and losers die under 10%). Exit<EMA7200. Bear year, all band
coins, cache-only. BUF in 5/10/15/20/30%. Run: python experiments/ema7200_entrysweep.py
"""
import os, glob, sys, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT

HIRES = "data/cache/hires"
S, E = dt.date(2024, 6, 1), dt.date(2025, 6, 1)
P7 = 7200; COST = 0.20; VLO, VHI = 2e6, 2e8
BUFS = [0.05, 0.10, 0.15, 0.20, 0.30]
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP",
          "USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM = {"PAXG","XAUT","WBTC","WBETH","BETH"}


def exc(s):
    if not s.endswith("USDT"):
        return True
    b = s[:-4]
    if b in STABLE or b in COMM:
        return True
    if any(b.endswith(t) for t in ("UP", "DOWN", "BULL", "BEAR")):
        return True
    return b[-2:] in ("3L", "3S", "5L", "5S")


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)


def read_cached(coin):
    fs = []
    for f in glob.glob(os.path.join(HIRES, f"{coin}-5m-*.csv")):
        try:
            d = dt.date.fromisoformat(os.path.basename(f).split("-5m-")[1][:10])
        except Exception:
            continue
        if S <= d < E:
            fs.append((d, f))
    if len(fs) < 200:
        return None
    fs.sort(); parts = []
    for _, f in fs:
        try:
            parts.append(pd.read_csv(f, usecols=["close"]))
        except Exception:
            pass
    if not parts:
        return None
    cl = pd.concat(parts, ignore_index=True)["close"].to_numpy(float)
    return cl[np.isfinite(cl) & (cl > 0)]


def bt(cl, e7, buf):
    n = len(cl); pos = False; entry = 0.0; eq = 1.0; tr = []
    for i in range(P7, n):
        if not pos:
            if cl[i] >= e7[i]*(1+buf):
                pos = True; entry = cl[i]
        elif cl[i] < e7[i]:
            r = (cl[i]/entry-1)*100 - COST; eq *= (1+r/100); tr.append(r); pos = False
    if pos:
        r = (cl[-1]/entry-1)*100 - COST; eq *= (1+r/100); tr.append(r)
    wr = (np.array(tr) > 0).mean()*100 if tr else 0
    return (eq-1)*100, len(tr), wr


def main():
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms("2024-03-01"), ms("2025-07-01"), log=lambda *a: None)
    vol = {}
    for sym, df in raw.items():
        if exc(sym):
            continue
        g = df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"], unit="ms"))
        dv = (g["close"]*g["volume"]).resample("D").sum().dropna()
        if len(dv) < 100:
            continue
        if VLO < dv.median() < VHI:
            vol[sym] = 1
    del raw
    coins = list(vol)
    print(f"عملات النطاق: {len(coins)} | مسح عتبة الدخول فوق EMA7200\n", flush=True)
    agg = {b: [] for b in BUFS}; ntr = {b: [] for b in BUFS}; wrr = {b: [] for b in BUFS}
    for k, c in enumerate(coins):
        cl = read_cached(c)
        if cl is None or len(cl) < P7+2000:
            continue
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        for b in BUFS:
            ret, nt, wr = bt(cl, e7, b); agg[b].append(ret); ntr[b].append(nt); wrr[b].append(wr)
        if (k+1) % 60 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)
    print(f"\nعملات: {len(agg[BUFS[0]])}\n")
    print(f"{'عتبة الدخول':<14}{'متوسط':>8}{'وسيط':>8}{'رابحة%':>8}{'WR':>6}{'متوسط صفقات':>12}")
    print("-"*56)
    for b in BUFS:
        r = np.array(agg[b])
        print(f"+{int(b*100)}% فوق EMA{'':<6}{r.mean():>+7.0f}%{np.median(r):>+7.0f}%{(r>0).mean()*100:>7.0f}%{np.mean(wrr[b]):>5.0f}%{np.mean(ntr[b]):>11.0f}")
    print("-"*56)
    print("الهدف: هل الدخول المتأخّر (+20%) يصطفي الرابحات ويقلب الفترة لموجبة؟")
    print("DONE_ESWEEP.")


if __name__ == "__main__":
    main()
