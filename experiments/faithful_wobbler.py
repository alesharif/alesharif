#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAITHFUL Wobbler replica + user's full system, with realistic slippage.

Per the indicator author (barnabygraham): "Uses moving averages in an RSI to
signal relative momentum/velocity." Settings: MA1=81, MA3=54, RSI=14, OB90/OS10.
So Wobbler = ribbon between SMA(RSI14, 54) and SMA(RSI14, 81); GREEN when the 54
is above the 81 (momentum up). The user's entry = the ribbon squeezes (the two
MAs converge) then EXPANDS GREEN (54 crosses above 81) from a low zone = 'buying
has started'. Then SL 10%, let winners run to +50% (the slippage-surviving config).

We test on DAILY Wobbler signals, SL 10%, TP +50% first-touch (90d window), and
report expectancy at fee 0.2% AND realistic cost (1% round-trip + 1% stop slip).
Note: user trades KuCoin (more/earlier small alts); we only have Binance/OKX data.
Run:  python experiments/faithful_wobbler.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

START = "2022-06-01"; END = "2026-06-01"
if "2024" in sys.argv:                       # out-of-sample regime
    START = "2023-09-01"; END = "2025-01-01"  # warmup from Sep'23, signals in 2024
DS = int(pd.Timestamp(START, tz="UTC").timestamp()*1000)
DE = int(pd.Timestamp(END, tz="UTC").timestamp()*1000)
SIG_FROM = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp()*1000) if "2024" in sys.argv else DS
SL = 0.10; TP = 0.50; WINDOW_MS = 90*24*3600*1000
RSI_N = 14; MA_FAST = 54; MA_SLOW = 81
LOW_ZONE = 45.0          # ribbon turning up from the lower half (oversold-ish)


def resample_d(df):
    g = df.set_index("dt")
    return pd.DataFrame({"h": g["high"].resample("D").max(),
                         "l": g["low"].resample("D").min(),
                         "c": g["close"].resample("D").last(),
                         "t": g["time"].resample("D").last()}).dropna()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def wobbler_entries(d, low_zone=True):
    c = d["c"].to_numpy(); t = d["t"].to_numpy(); n = len(c)
    if n < MA_SLOW + 20:
        return []
    r = rsi(c, RSI_N)
    maf = pd.Series(r).rolling(MA_FAST).mean().to_numpy()      # SMA(RSI,54) fast ribbon
    mas = pd.Series(r).rolling(MA_SLOW).mean().to_numpy()      # SMA(RSI,81) slow ribbon
    out = []
    for i in range(MA_SLOW + 1, n):
        cross_up = (maf[i] > mas[i]) and (maf[i-1] <= mas[i-1])    # turns green (expand)
        zone_ok = (mas[i] < LOW_ZONE) if low_zone else True       # from a low zone
        if cross_up and zone_ok and np.isfinite(mas[i]) and int(t[i]) >= SIG_FROM:
            out.append((int(t[i]), float(c[i])))
    return out


def main():
    print("FAITHFUL Wobbler (SMA of RSI 54/81) + user's system (SL10, TP50)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), DS, DE, log=lambda *a: None)

    for low_zone in (True, False):
        sigs = {}
        for sym, df in raw.items():
            df = df.sort_values("time").reset_index(drop=True)
            df["dt"] = pd.to_datetime(df["time"], unit="ms")
            e = wobbler_entries(resample_d(df), low_zone)
            if e: sigs[sym] = e
            df.drop(columns=["dt"], inplace=True, errors="ignore")
        # collect raw outcomes
        outs = []
        for sym, lst in sigs.items():
            dff = raw[sym]; t = dff["time"].to_numpy()
            H = dff["high"].to_numpy(float); L = dff["low"].to_numpy(float); C = dff["close"].to_numpy(float)
            for ent_t, ent_px in lst:
                j0 = np.searchsorted(t, ent_t)
                if j0 >= len(t): continue
                tp_px = ent_px*(1+TP); sl_px = ent_px*(1-SL); end_t = ent_t+WINDOW_MS
                tag = None; j = j0
                while j < len(t) and t[j] <= end_t:
                    if L[j] <= sl_px: tag = ("stop", -SL*100); break
                    if H[j] >= tp_px: tag = ("tp", TP*100); break
                    j += 1
                if tag is None:
                    tag = ("to", (C[min(j, len(t)-1)]/ent_px-1)*100)
                outs.append(tag)
        n = len(outs)
        tag = "from LOW zone (oversold turn)" if low_zone else "ANY cross-up"
        if n == 0:
            print(f"[{tag}] no signals\n"); continue
        tphit = sum(1 for k, _ in outs if k == "tp"); sthit = sum(1 for k, _ in outs if k == "stop")
        def exp(cost, stopslip):
            r = []
            for k, v in outs:
                if k == "stop": r.append(v - stopslip - cost)
                else: r.append(v - cost)
            return np.mean(r)
        print(f"### Wobbler entry — {tag}: {n} signals ###")
        print(f"  TP+50% hit {tphit/n*100:.0f}%, stop {sthit/n*100:.0f}%, "
              f"win {sum(1 for k,_ in outs if k=='tp')/n*100:.0f}%")
        print(f"  expectancy @fee0.2%:           {exp(0.2,0):+.1f}%/trade")
        print(f"  expectancy @realistic(1%+1%):  {exp(1.0,1.0):+.1f}%/trade")
        print(f"  expectancy @harsh(2%+1.5%):    {exp(2.0,1.5):+.1f}%/trade\n", flush=True)
    del raw; gc.collect()
    print("إن بقي موجباً عند 1%+1% => حافة Wobbler الأمينة تصمد. لاحظ: بيانات Binance/OKX لا KuCoin.")
    print("\nDONE_WOBBLER.", flush=True)


if __name__ == "__main__":
    main()
