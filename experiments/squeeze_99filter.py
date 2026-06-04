#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the user's survivorship fix: avoid coins in a persistent 99-day downtrend.

User's real-world rule: skip coins the exchange flags for delisting/monitoring
(can't backtest — no historical tag data) AND skip coins that have been falling
for ~99 days (the dying ones). His argument: the 'dead coins' that inflate a
survivorship-biased backtest are exactly the ones you'd avoid live anyway, so the
real result tracks the favorable backtest.

We CAN test the 99-day rule. Developed system (squeeze breakout + SL10 + TP+100%
+ liquid >$300k) across 3 regimes, comparing entry filters:
  ALL              : no trend filter
  no-99d-downtrend : require close[i] >= close[i-99]  (not net-down over 99 days)
  above-EMA200     : require close > EMA200 (stronger 'healthy trend' filter)
Net @1% RT + 1% stop slip. Run:  python experiments/squeeze_99filter.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

SL = 0.10; TP = 1.0; WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000
REGIMES = {"in-sample": ("2022-06-01", "2026-06-01", "2022-09-01"),
           "2024-bull": ("2023-06-01", "2025-01-01", "2024-01-01"),
           "bear22-23": ("2022-06-01", "2023-12-01", "2022-10-01")}


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def resample_d(df):
    g = df.set_index("dt")
    return pd.DataFrame({"o": g["open"].resample("D").first(), "h": g["high"].resample("D").max(),
                         "l": g["low"].resample("D").min(), "c": g["close"].resample("D").last(),
                         "v": g["volume"].resample("D").sum(), "t": g["time"].resample("D").last()}).dropna()


def entries(d, sig_from):
    o = d["o"].to_numpy(); c = d["c"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy(); n = len(c)
    if n < 220:
        return []
    sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
    bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
    e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
    macd = ema(c, 12)-ema(c, 26); sig = ema(macd, 9); hist = macd-sig
    dv = c*v; out = []
    for i in range(200, n):
        if (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i] and int(t[i]) >= sig_from):
            liq = np.nanmean(dv[max(0, i-30):i])
            not_dn99 = c[i] >= c[i-99]                 # not net-down over 99 days
            above200 = c[i] > e200[i]
            out.append((int(t[i]), float(c[i]), float(liq), bool(not_dn99), bool(above200)))
    return out


def main():
    print("User's 99-day-downtrend survivorship fix (developed system, TP+100, liquid)\n", flush=True)
    FILT = ["ALL", "no-99d-dn", "above-EMA200"]
    res = {rg: {f: [] for f in FILT} for rg in REGIMES}
    cnt = {rg: {f: 0 for f in FILT} for rg in REGIMES}

    for rg, (s, e, sf) in REGIMES.items():
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(s), ms(e), log=lambda *a: None)
        for sym, df in raw.items():
            df = df.sort_values("time").reset_index(drop=True)
            df["dt"] = pd.to_datetime(df["time"], unit="ms")
            sigs = entries(resample_d(df), ms(sf))
            if not sigs:
                df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
            t = df["time"].to_numpy(); H = df["high"].to_numpy(float)
            L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
            for ent_t, ent_px, liq, not_dn99, above200 in sigs:
                if liq <= LIQ_MIN:
                    continue
                j0 = np.searchsorted(t, ent_t)
                if j0 >= len(t):
                    continue
                sl = ent_px*(1-SL); tp = ent_px*(1+TP); end_t = ent_t+WINDOW_MS
                r = None; j = j0
                while j < len(t) and t[j] <= end_t:
                    if L[j] <= sl: r = (-SL*100, True); break
                    if H[j] >= tp: r = (TP*100, False); break
                    j += 1
                if r is None:
                    r = ((C[min(j, len(t)-1)]/ent_px-1)*100, False)
                nr = r[0] - 1.0 - (1.0 if r[1] else 0.0)
                res[rg]["ALL"].append(nr); cnt[rg]["ALL"] += 1
                if not_dn99: res[rg]["no-99d-dn"].append(nr); cnt[rg]["no-99d-dn"] += 1
                if above200: res[rg]["above-EMA200"].append(nr); cnt[rg]["above-EMA200"] += 1
            df.drop(columns=["dt"], inplace=True, errors="ignore")
        print(f"  {rg}: ALL {cnt[rg]['ALL']}, no-99d-dn {cnt[rg]['no-99d-dn']}, "
              f"above200 {cnt[rg]['above-EMA200']}", flush=True)
        del raw; gc.collect()

    print(f"\n{'filter':<14}" + "".join(f"{rg:>14}" for rg in REGIMES))
    print("-" * 60)
    for f in FILT:
        row = f"{f:<14}"
        for rg in REGIMES:
            a = np.array(res[rg][f])
            row += f"{(a.mean() if len(a) else float('nan')):>+13.1f}%"
        print(row, flush=True)
    print("\nإن بقي/ارتفع التوقّع مع الفلتر => قاعدتك صحيحة: تجنّب المحتضرة يحافظ على الحافة.")
    print("\nDONE_99.", flush=True)


if __name__ == "__main__":
    main()
