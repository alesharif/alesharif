#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""For EMA7200+5% (exit<EMA7200) LOSING trades in the bear year: how much did each rise
(max favorable excursion, close-based) before reversing into a loss? How many never rose?
All band coins, cache-only. Run: python experiments/ema7200_loser_mfe.py
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
            parts.append(pd.read_csv(f, usecols=["time", "close"]))
        except Exception:
            pass
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    cl = df["close"].to_numpy(float)
    return cl[np.isfinite(cl) & (cl > 0)]


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
    print(f"عملات النطاق: {len(coins)} | تشريح الخاسرات (MFE قبل الخسارة)\n", flush=True)

    mfe_losers = []; loss_ret = []; nwin = nloss = 0
    for k, c in enumerate(coins):
        cl = read_cached(c)
        if cl is None or len(cl) < P7+2000:
            continue
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        n = len(cl); pos = False; entry = 0.0; peak = 0.0
        for i in range(P7, n):
            if not pos:
                if cl[i] >= e7[i]*(1+BUF):
                    pos = True; entry = cl[i]; peak = cl[i]
            else:
                peak = max(peak, cl[i])
                if cl[i] < e7[i]:
                    r = (cl[i]/entry-1)*100 - COST
                    if r > 0:
                        nwin += 1
                    else:
                        nloss += 1; mfe_losers.append((peak/entry-1)*100); loss_ret.append(r)
                    pos = False
        if (k+1) % 60 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)

    mfe = np.array(mfe_losers); lr = np.array(loss_ret)
    print(f"\nإجمالي الصفقات: {nwin+nloss} | رابحة {nwin} | خاسرة {nloss} ({nloss/(nwin+nloss)*100:.0f}%)\n")
    print("##### كم صعدت الخاسرة قبل أن تنعكس وتخسر؟ (أقصى صعود MFE) #####")
    buckets = [(-1e9, 0.0, "لم تصعد أبداً (≤0%)"), (0, 1, "0–1%"), (1, 2, "1–2%"), (2, 3, "2–3%"),
               (3, 5, "3–5%"), (5, 10, "5–10%"), (10, 20, "10–20%"), (20, 1e9, ">20%")]
    for lo, hi, lab in buckets:
        m = (mfe > lo) & (mfe <= hi)
        print(f"  {lab:<22}{m.sum():>7}  ({m.mean()*100:>4.1f}%)")
    print(f"\n  لم تصعد أبداً (≤0%): {(mfe<=0).mean()*100:.0f}% من الخاسرات")
    print(f"  وسيط أقصى صعود للخاسرة: {np.median(mfe):+.1f}%   |   المتوسّط: {mfe.mean():+.1f}%")
    print(f"  وسيط خسارة الخاسرة: {np.median(lr):+.1f}%")
    print("DONE_MFE.")


if __name__ == "__main__":
    main()
