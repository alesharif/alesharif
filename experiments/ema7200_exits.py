#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Try to make EMA7200+5% PROFITABLE in the bearish year (2024-06..2025-06) by cutting
losers faster. Same entry (+5% above EMA7200). Test EXIT variants on all band coins
(cache-only read). cost 0.2% RT.
  A exit<EMA7200 (baseline, slow)
  B exit<EMA600  (faster)
  C +hard stop -8% from entry
  D trailing 15% from peak
  E trailing 10% from peak
Run: python experiments/ema7200_exits.py
"""
import os, glob, sys, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT

HIRES = "data/cache/hires"
S, E = dt.date(2024, 6, 1), dt.date(2025, 6, 1)
P7 = 7200; P6 = 600; COST = 0.20; BUF = 0.05; VLO, VHI = 2e6, 2e8
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
    fs.sort()
    parts = []
    for _, f in fs:
        try:
            parts.append(pd.read_csv(f, usecols=["time", "close"]))
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    cl = df["close"].to_numpy(float)
    return cl[np.isfinite(cl) & (cl > 0)]


def bt(cl, e7, mode):
    n = len(cl); pos = False; entry = 0.0; peak = 0.0; eq = 1.0; tr = []
    for i in range(P7, n):
        if not pos:
            if cl[i] >= e7[i]*(1+BUF):
                pos = True; entry = cl[i]; peak = cl[i]
        else:
            peak = max(peak, cl[i]); ex = False
            if mode == "A" and cl[i] < e7[i]:
                ex = True
            elif mode == "B" and cl[i] < e6arr[i]:
                ex = True
            elif mode == "C" and (cl[i] < e7[i] or cl[i] <= entry*0.92):
                ex = True
            elif mode == "D" and (cl[i] < e7[i] or cl[i] <= peak*0.85):
                ex = True
            elif mode == "E" and (cl[i] < e7[i] or cl[i] <= peak*0.90):
                ex = True
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
        md = dv.median()
        if VLO < md < VHI:
            vol[sym] = md
    del raw
    coins = sorted(vol, key=lambda s: -vol[s])
    print(f"كل عملات النطاق: {len(coins)} | محاولة قلب الفترة الهابطة لرابحة بطرق خروج\n", flush=True)
    global e6arr
    modes = {"A": "خروج تحت EMA7200 (الأساس)", "B": "خروج تحت EMA600 (أسرع)",
             "C": "+ وقف -8% من الدخول", "D": "وقف متحرّك 15% من القمّة", "E": "وقف متحرّك 10% من القمّة"}
    agg = {m: [] for m in modes}; BH = []
    for k, c in enumerate(coins):
        cl = read_cached(c)
        if cl is None or len(cl) < P7+2000:
            continue
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        e6arr = pd.Series(cl).ewm(span=P6, adjust=False).mean().to_numpy()
        for m in modes:
            agg[m].append(bt(cl, e7, m)[0])
        BH.append((cl[-1]/cl[0]-1)*100)
        if (k+1) % 60 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)
    bh = np.array(BH)
    print(f"\nعملات: {len(bh)}\n")
    print(f"{'طريقة الخروج':<30}{'متوسط':>8}{'وسيط':>8}{'رابحة%':>8}")
    print("-"*54)
    for m, lab in modes.items():
        r = np.array(agg[m])
        print(f"{lab:<30}{r.mean():>+7.0f}%{np.median(r):>+7.0f}%{(r>0).mean()*100:>7.0f}%")
    print("-"*54)
    print(f"{'الاحتفاظ بالعملة':<30}{bh.mean():>+7.0f}%{np.median(bh):>+7.0f}%{(bh>0).mean()*100:>7.0f}%")
    print("\nالهدف: أيّ طريقة خروج تقلب الفترة الهابطة لموجبة؟")
    print("DONE_EXITS.")


if __name__ == "__main__":
    main()
