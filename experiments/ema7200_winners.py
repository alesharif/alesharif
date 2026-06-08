#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anatomy of WINNERS vs LOSERS for EMA7200+5% (exit<EMA7200), bear year, all band coins.
For each trade record: final return, MFE (max close gain), MAE (max close drawdown before
peak). Compare the signatures. cache-only. Run: python experiments/ema7200_winners.py
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
            parts.append(pd.read_csv(f, usecols=["close"]))
        except Exception:
            pass
    if not parts:
        return None
    cl = pd.concat(parts, ignore_index=True)["close"].to_numpy(float)
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
    print(f"عملات النطاق: {len(coins)} | تشريح الرابحات مقابل الخاسرات\n", flush=True)

    W_mfe = []; W_ret = []; L_mfe = []; L_ret = []
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
                    r = (cl[i]/entry-1)*100 - COST; mfe = (peak/entry-1)*100
                    if r > 0:
                        W_ret.append(r); W_mfe.append(mfe)
                    else:
                        L_ret.append(r); L_mfe.append(mfe)
                    pos = False
        if (k+1) % 60 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)

    W_mfe = np.array(W_mfe); W_ret = np.array(W_ret); L_mfe = np.array(L_mfe); L_ret = np.array(L_ret)
    nw = len(W_ret); nl = len(L_ret); tot = nw+nl
    print(f"\nإجمالي الصفقات: {tot} | رابحة {nw} ({nw/tot*100:.0f}%) | خاسرة {nl} ({nl/tot*100:.0f}%)\n")
    print(f"{'':<22}{'الرابحات':>12}{'الخاسرات':>12}")
    print("-"*46)
    print(f"{'العدد':<22}{nw:>12}{nl:>12}")
    print(f"{'وسيط العائد':<22}{np.median(W_ret):>+11.1f}%{np.median(L_ret):>+11.1f}%")
    print(f"{'متوسط العائد':<22}{W_ret.mean():>+11.1f}%{L_ret.mean():>+11.1f}%")
    print(f"{'وسيط أقصى صعود MFE':<22}{np.median(W_mfe):>+11.1f}%{np.median(L_mfe):>+11.1f}%")
    print(f"{'متوسط أقصى صعود MFE':<22}{W_mfe.mean():>+11.1f}%{L_mfe.mean():>+11.1f}%")
    print(f"\n##### توزيع أقصى صعود (MFE): رابحات مقابل خاسرات #####")
    print(f"{'MFE':<12}{'رابحات%':>10}{'خاسرات%':>10}")
    for lo, hi, lab in [(-1e9, 0, "≤0"), (0, 3, "0–3%"), (3, 5, "3–5%"), (5, 10, "5–10%"),
                        (10, 20, "10–20%"), (20, 50, "20–50%"), (50, 1e9, ">50%")]:
        wm = ((W_mfe > lo) & (W_mfe <= hi)).mean()*100
        lm = ((L_mfe > lo) & (L_mfe <= hi)).mean()*100
        print(f"{lab:<12}{wm:>9.1f}%{lm:>9.1f}%")
    print(f"\nمجموع عائد الرابحات: {W_ret.sum():+.0f}%  |  مجموع خسارة الخاسرات: {L_ret.sum():+.0f}%  |  الصافي: {W_ret.sum()+L_ret.sum():+.0f}%")
    print("DONE_WIN.")


if __name__ == "__main__":
    main()
