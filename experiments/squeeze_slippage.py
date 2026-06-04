#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decider: does the user's full system survive REAL small-alt slippage?

The proxy (squeeze+confirmation + 10% SL) showed positive expectancy at FEE 0.2%.
But the stop is hit 47-74% of the time, and small alts have wide spreads/slippage.
We sweep realistic round-trip cost (0.2..2.0%) PLUS extra stop slippage (the stop
fills worse than -10% on a fast drop), for TP +10% (his comfort) and +50% (best),
to find where the edge dies. Run:  python experiments/squeeze_slippage.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

START = "2022-06-01"; END = "2026-06-01"
DS = int(pd.Timestamp(START, tz="UTC").timestamp()*1000)
DE = int(pd.Timestamp(END, tz="UTC").timestamp()*1000)
SL = 0.10; WINDOW_MS = 90*24*3600*1000
TPS = [0.10, 0.50]
COSTS = [0.2, 0.5, 1.0, 1.5, 2.0]        # round-trip % (spread+slippage+fee)
STOP_SLIP = 1.0                          # extra % worse fill when stop triggers


def resample_d(df):
    g = df.set_index("dt")
    return pd.DataFrame({"o": g["open"].resample("D").first(),
                         "h": g["high"].resample("D").max(),
                         "l": g["low"].resample("D").min(),
                         "c": g["close"].resample("D").last(),
                         "t": g["time"].resample("D").last()}).dropna()


def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def entries(d):
    o = d["o"].to_numpy(); c = d["c"].to_numpy(); t = d["t"].to_numpy()
    n = len(c)
    if n < 120:
        return []
    sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
    bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
    e20 = ema(c, 20); e50 = ema(c, 50)
    macd = ema(c, 12)-ema(c, 26); sig = ema(macd, 9); hist = macd-sig
    out = []
    for i in range(100, n):
        if (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]):
            out.append((int(t[i]), float(c[i])))
    return out


def main():
    print("SLIPPAGE decider for the user's full system (SL 10%)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), DS, DE, log=lambda *a: None)
    sigs = {}
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        e = entries(resample_d(df))
        if e: sigs[sym] = e
        df.drop(columns=["dt"], inplace=True)
    nsig = sum(len(v) for v in sigs.values())
    print(f"  {nsig} signals; sweeping cost (stop slippage +{STOP_SLIP}% extra)\n", flush=True)

    # precompute raw outcomes (gross, before cost): 'TP'|'stop'|'timeoutRet'
    def raw_outcomes(tp):
        res = []
        for sym, lst in sigs.items():
            df = raw[sym]; t = df["time"].to_numpy()
            H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
            for ent_t, ent_px in lst:
                j0 = np.searchsorted(t, ent_t)
                if j0 >= len(t):
                    continue
                tp_px = ent_px*(1+tp); sl_px = ent_px*(1-SL); end_t = ent_t+WINDOW_MS
                tag = None; j = j0
                while j < len(t) and t[j] <= end_t:
                    if L[j] <= sl_px:
                        tag = ("stop", -SL*100); break
                    if H[j] >= tp_px:
                        tag = ("tp", tp*100); break
                    j += 1
                if tag is None:
                    tag = ("to", (C[min(j, len(t)-1)]/ent_px-1)*100)
                res.append(tag)
        return res

    print(f"{'TP':<7}{'cost->':<8}" + "".join(f"{c:>8.1f}%" for c in COSTS))
    print("-" * 56)
    for tp in TPS:
        outs = raw_outcomes(tp)
        row = []
        for cost in COSTS:
            rets = []
            for kind, val in outs:
                if kind == "stop":
                    rets.append(val - STOP_SLIP - cost)     # extra stop slippage
                else:
                    rets.append(val - cost)
            row.append(np.mean(rets))
        print(f"+{int(tp*100)}%{'':<4}{'exp/trade':<8}" + "".join(f"{v:>+7.1f}%" for v in row))
    del raw; gc.collect()
    print("\nالقيمة = التوقّع/صفقة عند كل مستوى تكلفة. سالب => الانزلاق قتل الحافة.")
    print("العملات الصغيرة جداً تكلفتها الحقيقية غالباً 1-2%.")
    print("\nDONE_SLIP.", flush=True)


if __name__ == "__main__":
    main()
