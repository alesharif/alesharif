#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CONDITIONAL trailing stop matching the winner signature: never cut under +ACT% peak;
once peak >= +ACT%, exit on close<=peak*(1-TRAIL) OR close<EMA7200. Entry EMA7200+5%.
Bear year, all band coins, cache-only. Compare to baseline (exit<EMA7200 only).
Run: python experiments/ema7200_condtrail.py
"""
import os, glob, sys, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT

HIRES = "data/cache/hires"
S, E = dt.date(2024, 6, 1), dt.date(2025, 6, 1)
P7 = 7200; COST = 0.20; BUF = 0.05; VLO, VHI = 2e6, 2e8
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP",
          "USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM = {"PAXG","XAUT","WBTC","WBETH","BETH"}
CONF = [(None, None, "أساس (تحت EMA7200)"), (20, 0.10, "تفعيل+20% تريل10%"),
        (20, 0.15, "تفعيل+20% تريل15%"), (15, 0.10, "تفعيل+15% تريل10%"),
        (25, 0.15, "تفعيل+25% تريل15%"), (20, 0.20, "تفعيل+20% تريل20%")]


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


def bt(cl, e7, act, trail):
    n = len(cl); pos = False; entry = 0.0; peak = 0.0; eq = 1.0; tr = []
    for i in range(P7, n):
        if not pos:
            if cl[i] >= e7[i]*(1+BUF):
                pos = True; entry = cl[i]; peak = cl[i]
        else:
            peak = max(peak, cl[i]); ex = False
            if act is not None and (peak/entry-1)*100 >= act and cl[i] <= peak*(1-trail):
                ex = True                       # conditional trailing (only after +act%)
            elif cl[i] < e7[i]:
                ex = True                       # base exit
            if ex:
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
    print(f"عملات النطاق: {len(coins)} | وقف متحرّك مشروط بالقمّة\n", flush=True)
    agg = {i: [] for i in range(len(CONF))}; wrr = {i: [] for i in range(len(CONF))}
    for k, c in enumerate(coins):
        cl = read_cached(c)
        if cl is None or len(cl) < P7+2000:
            continue
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        for idx, (act, trail, _) in enumerate(CONF):
            ret, nt, wr = bt(cl, e7, act, trail); agg[idx].append(ret); wrr[idx].append(wr)
        if (k+1) % 60 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)
    print(f"\nعملات: {len(agg[0])}\n")
    print(f"{'الإعداد':<24}{'متوسط':>8}{'وسيط':>8}{'رابحة%':>8}{'WR':>6}")
    print("-"*54)
    for idx, (act, trail, lab) in enumerate(CONF):
        r = np.array(agg[idx])
        print(f"{lab:<24}{r.mean():>+7.0f}%{np.median(r):>+7.0f}%{(r>0).mean()*100:>7.0f}%{np.mean(wrr[idx]):>5.0f}%")
    print("-"*54)
    print("الهدف: هل الوقف المشروط بـ+20% يرفع العائد فوق الأساس؟")
    print("DONE_CT.")


if __name__ == "__main__":
    main()
