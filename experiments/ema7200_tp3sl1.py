#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA7200+5% entry, exit TP+3% / SL-1% (first touch on intrabar high/low). 3:1 R:R,
breakeven WR=25%. Bear year, all band coins, cache-only (reads high/low/close).
Compare to baseline (exit<EMA7200). Run: python experiments/ema7200_tp3sl1.py
"""
import os, glob, sys, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT

HIRES = "data/cache/hires"
S, E = dt.date(2024, 6, 1), dt.date(2025, 6, 1)
P7 = 7200; COST = 0.20; BUF = 0.05; VLO, VHI = 2e6, 2e8
TP = 0.03; SL = 0.01
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


def read_hlc(coin):
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
            parts.append(pd.read_csv(f, usecols=["high", "low", "close"]))
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    return df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)


def bt_tpsl(h, l, c, e7):
    n = len(c); pos = False; entry = 0.0; eq = 1.0; tr = []
    for i in range(P7, n):
        if not pos:
            if c[i] >= e7[i]*(1+BUF):
                pos = True; entry = c[i]
        else:
            if l[i] <= entry*(1-SL):
                r = -SL*100 - COST; eq *= (1+r/100); tr.append(r); pos = False
            elif h[i] >= entry*(1+TP):
                r = TP*100 - COST; eq *= (1+r/100); tr.append(r); pos = False
    if pos:
        r = (c[-1]/entry-1)*100 - COST; eq *= (1+r/100); tr.append(r)
    wr = (np.array(tr) > 0).mean()*100 if tr else 0
    return (eq-1)*100, len(tr), wr


def bt_base(c, e7):
    n = len(c); pos = False; entry = 0.0; eq = 1.0; tr = []
    for i in range(P7, n):
        if not pos:
            if c[i] >= e7[i]*(1+BUF):
                pos = True; entry = c[i]
        elif c[i] < e7[i]:
            r = (c[i]/entry-1)*100 - COST; eq *= (1+r/100); tr.append(r); pos = False
    if pos:
        r = (c[-1]/entry-1)*100 - COST; eq *= (1+r/100); tr.append(r)
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
    print(f"عملات النطاق: {len(coins)} | TP+3% / SL-1% مقابل الأساس\n", flush=True)
    TPL = []; TPN = []; TPW = []; BL = []
    for k, c in enumerate(coins):
        r = read_hlc(c)
        if r is None:
            continue
        h, l, cl = r
        m = np.isfinite(cl) & (cl > 0)
        if m.sum() < P7+2000:
            continue
        h, l, cl = h[m], l[m], cl[m]
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        ret, nt, wr = bt_tpsl(h, l, cl, e7); TPL.append(ret); TPN.append(nt); TPW.append(wr)
        BL.append(bt_base(cl, e7)[0])
        if (k+1) % 60 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)
    tp = np.array(TPL); bl = np.array(BL)
    print(f"\nعملات: {len(tp)}\n")
    print(f"{'الإعداد':<26}{'متوسط':>8}{'وسيط':>8}{'رابحة%':>8}{'WR':>6}{'صفقات':>8}")
    print("-"*64)
    print(f"{'TP+3% / SL-1%':<26}{tp.mean():>+7.0f}%{np.median(tp):>+7.0f}%{(tp>0).mean()*100:>7.0f}%{np.mean(TPW):>5.0f}%{np.mean(TPN):>7.0f}")
    print(f"{'الأساس (تحت EMA7200)':<26}{bl.mean():>+7.0f}%{np.median(bl):>+7.0f}%{(bl>0).mean()*100:>7.0f}%")
    print("-"*64)
    print("تعادل TP3/SL1 = WR 25% (3:1). هل WR يتجاوزه؟ والوقف -1% يُضرب بضوضاء 5m؟")
    print("DONE_TP3SL1.")


if __name__ == "__main__":
    main()
