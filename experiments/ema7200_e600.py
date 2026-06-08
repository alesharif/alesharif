#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA7200 +5% entry, exit on close<EMA7200 (no TP), on ALL band coins ($2M-$200M).
TWO variants: baseline, and + price-above-EMA600 filter. Reads CACHED 5m CSVs directly
(no network -> no missing-day timeouts). cost 0.2% RT. vs buy&hold. With progress.
Run: python experiments/ema7200_e600.py
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
    """Concat cached 5m CSVs for the coin in [S,E] directly from disk (no network)."""
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


def bt(cl, e7, e6, use600):
    n = len(cl); pos = False; entry = 0.0; eq = 1.0; tr = []
    for i in range(P7, n):
        if not pos:
            if cl[i] >= e7[i]*(1+BUF) and (not use600 or cl[i] > e6[i]):
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
        md = dv.median()
        if VLO < md < VHI:
            vol[sym] = md
    del raw
    coins = sorted(vol, key=lambda s: -vol[s])
    print(f"كل عملات النطاق ($2M-$200M): {len(coins)} | قراءة من الكاش مباشرةً", flush=True)

    A = []; B = []; BH = []
    for k, c in enumerate(coins):
        cl = read_cached(c)
        if cl is None or len(cl) < P7+2000:
            continue
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        e6 = pd.Series(cl).ewm(span=P6, adjust=False).mean().to_numpy()
        A.append((c,)+bt(cl, e7, e6, False)); B.append((c,)+bt(cl, e7, e6, True)); BH.append((cl[-1]/cl[0]-1)*100)
        if (k+1) % 40 == 0:
            print(f"  {k+1}/{len(coins)} (مُختبَرة {len(A)})...", flush=True)

    def summ(R, lab):
        r = np.array([x[1] for x in R])
        print(f"{lab}: متوسط {r.mean():+.0f}% | وسيط {np.median(r):+.0f}% | رابحة {(r>0).mean()*100:.0f}% | WR {np.mean([x[3] for x in R]):.0f}%")
    bh = np.array(BH)
    print(f"\nعملات مُختبَرة (لها سنة كافية بالكاش): {len(A)}\n")
    summ(A, "(أ) أساس (EMA7200+5%، بلا EMA600)")
    summ(B, "(ب) + شرط السعر فوق EMA600")
    print(f"الاحتفاظ: متوسط {bh.mean():+.0f}% | وسيط {np.median(bh):+.0f}%")
    print("\nأفضل 12 بالنسخة (ب):")
    for i in sorted(range(len(B)), key=lambda i: -B[i][1])[:12]:
        print(f"  {B[i][0]:<11}ب {B[i][1]:+5.0f}%  أساس {A[i][1]:+5.0f}%  احتفاظ {BH[i]:+5.0f}%  ({B[i][2]}ص WR{B[i][3]:.0f}%)")
    print("DONE_E600.")


if __name__ == "__main__":
    main()
