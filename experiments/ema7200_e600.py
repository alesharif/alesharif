#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA7200 +5% entry, exit on close<EMA7200 (no TP), on ALL band coins ($2M-$200M).
TWO variants: baseline, and + price-above-EMA600 entry filter. Parallel 5m prefetch
(cached reused), per-coin progress. cost 0.2% RT. vs buy&hold.
Run: python experiments/ema7200_e600.py
"""
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT
from binance_sim import hires_data as HR

S, E = "2024-06-01", "2025-06-01"
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
    coins = sorted(vol, key=lambda s: -vol[s])     # ALL band coins
    print(f"كل عملات النطاق ($2M-$200M): {len(coins)} | EMA7200+5% خروج تحت EMA | +شرط EMA600", flush=True)

    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(S, E, freq="D")]
    print("تحميل/قراءة 5m (متوازٍ، المخزّن لا يُعاد)...", flush=True)

    def _f(cd):
        try:
            HR.load_day(cd[0], "5m", cd[1])
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(_f, [(c, d) for c in coins for d in days]))
    print("اكتمل التحميل، المعالجة...", flush=True)

    A = []; B = []; BH = []
    for k, c in enumerate(coins):
        df = HR.load_range(c, "5m", ms(S), ms(E))
        if df is None or len(df) < P7+2000:
            continue
        cl = df.drop_duplicates("time").sort_values("time")["close"].to_numpy(float)
        cl = cl[np.isfinite(cl) & (cl > 0)]
        if len(cl) < P7+2000:
            continue
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        e6 = pd.Series(cl).ewm(span=P6, adjust=False).mean().to_numpy()
        A.append((c,)+bt(cl, e7, e6, False)); B.append((c,)+bt(cl, e7, e6, True)); BH.append((cl[-1]/cl[0]-1)*100)
        if (k+1) % 40 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)

    def summ(R, lab):
        r = np.array([x[1] for x in R])
        print(f"{lab}: متوسط {r.mean():+.0f}% | وسيط {np.median(r):+.0f}% | رابحة {(r>0).mean()*100:.0f}% | WR {np.mean([x[3] for x in R]):.0f}%")
    bh = np.array(BH)
    print(f"\nعملات مُختبَرة: {len(A)}\n")
    summ(A, "(أ) أساس (EMA7200+5%، بلا EMA600)")
    summ(B, "(ب) + شرط السعر فوق EMA600")
    print(f"الاحتفاظ: متوسط {bh.mean():+.0f}% | وسيط {np.median(bh):+.0f}%")
    print("\nأفضل 12 بالنسخة (ب):")
    for i in sorted(range(len(B)), key=lambda i: -B[i][1])[:12]:
        print(f"  {B[i][0]:<11}ب {B[i][1]:+5.0f}%  أساس {A[i][1]:+5.0f}%  احتفاظ {BH[i]:+5.0f}%  ({B[i][2]}ص WR{B[i][3]:.0f}%)")
    print("DONE_E600.")


if __name__ == "__main__":
    main()
