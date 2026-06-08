#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA7200+5% entry, exit<EMA7200, + RELATIVE-STRENGTH filter: only enter if the coin's
return over the last L bars beats BTC's over the same window (coin leading the market).
Bear year 2024-06..2025-06, all band coins, cache-only. Variants: no-RS, RS-1day, RS-1week.
Run: python experiments/ema7200_rs.py
"""
import os, glob, sys, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT

HIRES = "data/cache/hires"
S, E = dt.date(2024, 6, 1), dt.date(2025, 6, 1)
P7 = 7200; COST = 0.20; BUF = 0.05; VLO, VHI = 2e6, 2e8
DAYBARS = 288; WEEKBARS = 2016
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
        return None, None
    fs.sort()
    parts = []
    for _, f in fs:
        try:
            parts.append(pd.read_csv(f, usecols=["time", "close"]))
        except Exception:
            pass
    if not parts:
        return None, None
    df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    t = df["time"].to_numpy(np.int64); cl = df["close"].to_numpy(float)
    m = np.isfinite(cl) & (cl > 0)
    return t[m], cl[m]


def main():
    # BTC benchmark (cached)
    bt, bc = read_cached("BTCUSDT")
    if bt is None:
        print("لا بيانات BTC للمعيار"); return
    btc_d = np.full(len(bc), np.nan); btc_w = np.full(len(bc), np.nan)
    btc_d[DAYBARS:] = bc[DAYBARS:]/bc[:-DAYBARS]-1
    btc_w[WEEKBARS:] = bc[WEEKBARS:]/bc[:-WEEKBARS]-1

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
    print(f"عملات النطاق: {len(coins)} | فلتر القوّة النسبية مقابل BTC\n", flush=True)

    def bt_run(t, cl, e7, mode):
        n = len(cl); pos = False; entry = 0.0; eq = 1.0; tr = []
        for i in range(P7, n):
            if not pos:
                if cl[i] < e7[i]*(1+BUF):
                    continue
                if mode != "none":
                    L = DAYBARS if mode == "day" else WEEKBARS
                    if i < L:
                        continue
                    cr = cl[i]/cl[i-L]-1
                    j = np.searchsorted(bt, t[i], side="right")-1
                    br = (btc_d[j] if mode == "day" else btc_w[j]) if j >= 0 else np.nan
                    if not (np.isfinite(br) and cr > br):     # require coin stronger than BTC
                        continue
                pos = True; entry = cl[i]
            elif cl[i] < e7[i]:
                r = (cl[i]/entry-1)*100 - COST; eq *= (1+r/100); tr.append(r); pos = False
        if pos:
            r = (cl[-1]/entry-1)*100 - COST; eq *= (1+r/100); tr.append(r)
        wr = (np.array(tr) > 0).mean()*100 if tr else 0
        return (eq-1)*100, len(tr), wr

    modes = {"none": "بلا فلتر (أساس)", "day": "أقوى من BTC (يوم)", "week": "أقوى من BTC (أسبوع)"}
    agg = {m: [] for m in modes}; ntr = {m: [] for m in modes}; BH = []
    for k, c in enumerate(coins):
        t, cl = read_cached(c)
        if t is None or len(cl) < P7+2000:
            continue
        e7 = pd.Series(cl).ewm(span=P7, adjust=False).mean().to_numpy()
        for m in modes:
            ret, nt, wr = bt_run(t, cl, e7, m); agg[m].append(ret); ntr[m].append(nt)
        BH.append((cl[-1]/cl[0]-1)*100)
        if (k+1) % 60 == 0:
            print(f"  {k+1}/{len(coins)}...", flush=True)
    bh = np.array(BH)
    print(f"\nعملات: {len(bh)}\n")
    print(f"{'الفلتر':<22}{'متوسط':>8}{'وسيط':>8}{'رابحة%':>8}{'متوسط صفقات':>12}")
    print("-"*58)
    for m, lab in modes.items():
        r = np.array(agg[m])
        print(f"{lab:<22}{r.mean():>+7.0f}%{np.median(r):>+7.0f}%{(r>0).mean()*100:>7.0f}%{np.mean(ntr[m]):>11.0f}")
    print("-"*58)
    print(f"{'الاحتفاظ':<22}{bh.mean():>+7.0f}%{np.median(bh):>+7.0f}%{(bh>0).mean()*100:>7.0f}")
    print("\nالهدف: هل تداول الأقوى-من-BTC يقلب الفترة الهابطة لموجبة؟")
    print("DONE_RS.")


if __name__ == "__main__":
    main()
