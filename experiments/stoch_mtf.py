#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-timeframe STOCHASTIC strategy (user's idea) — tested honestly.

Rule:
  * regime gate (MONTHLY): stochastic %K > %D (fast above slow) = bullish month.
  * entry (WEEKLY): buy when weekly %K crosses ABOVE %D, while monthly is bullish.
  * exit: weekly %K crosses below %D, OR monthly turns bearish.
Stochastic = classic %K(14) with %D = SMA(%K,3). One position at a time.
Compared to Buy&Hold per coin, on a basket of major coins (long history). Note:
testing on SURVIVORS (majors) is GENEROUS to a bottom-reversal idea — small dead
alts (the real targets) recover far less, so real-world would be worse.
Run:  python experiments/stoch_mtf.py
"""

from __future__ import annotations

import json, urllib.request
import numpy as np
import pandas as pd

FEE = 0.2
COINS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD",
         "DOGE-USD", "LTC-USD", "LINK-USD", "TRX-USD", "BCH-USD", "XLM-USD"]


def fetch_daily(tkr):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{tkr}?range=15y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    j = json.loads(urllib.request.urlopen(req, timeout=40).read())
    r = j["chart"]["result"][0]; q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"]},
                      index=pd.to_datetime(r["timestamp"], unit="s")).dropna()
    return df


def resample(df, rule):
    return pd.DataFrame({"h": df["h"].resample(rule).max(),
                         "l": df["l"].resample(rule).min(),
                         "c": df["c"].resample(rule).last()}).dropna()


def stoch(h, l, c, n=14, d=3):
    llv = pd.Series(l).rolling(n).min(); hhv = pd.Series(h).rolling(n).max()
    k = 100 * (c - llv) / (hhv - llv).replace(0, np.nan)
    k = k.fillna(50.0)
    dd = k.rolling(d).mean().fillna(50.0)
    return k.to_numpy(), dd.to_numpy()


def metrics(eq, n_weeks):
    peak = np.maximum.accumulate(eq); dd = (eq / peak - 1).min() * 100
    yrs = n_weeks / 52
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 and eq[-1] > 0 else -100
    return cagr, (eq[-1] - 1) * 100, dd


def run(tkr):
    df = fetch_daily(tkr)
    wk = resample(df, "W"); mo = resample(df, "ME")
    if len(wk) < 60 or len(mo) < 20:
        return None
    wk_k, wk_d = stoch(wk["h"].to_numpy(), wk["l"].to_numpy(), wk["c"].to_numpy())
    mo_k, mo_d = stoch(mo["h"].to_numpy(), mo["l"].to_numpy(), mo["c"].to_numpy())
    mo_bull = pd.Series(mo_k > mo_d, index=mo.index).shift(1)
    mo_bull_w = mo_bull.reindex(wk.index, method="ffill").fillna(False).to_numpy()
    wret = wk["c"].pct_change().fillna(0).to_numpy()
    n = len(wk)
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    cu[1:] = (wk_k[1:] > wk_d[1:]) & (wk_k[:-1] <= wk_d[:-1])
    cd[1:] = (wk_k[1:] < wk_d[1:]) & (wk_k[:-1] >= wk_d[:-1])

    eq = 1.0; curve = [1.0]; inmkt = False; sw = 0; inm = 0
    for t in range(1, n):
        sigup, sigdn, bull = cu[t-1], cd[t-1], mo_bull_w[t-1]
        if not inmkt and sigup and bull:
            inmkt = True; eq *= (1 - FEE/100); sw += 1
        elif inmkt and (sigdn or not bull):
            inmkt = False; eq *= (1 - FEE/100); sw += 1
        if inmkt:
            eq *= (1 + wret[t]); inm += 1
        curve.append(eq)
    curve = np.array(curve)
    bh = np.concatenate([[1.0], np.cumprod(1 + wret[1:])])
    s = metrics(curve, n); b = metrics(bh, n)
    return s, b, sw, inm * 100 // n, n


def main():
    print("MULTI-TIMEFRAME STOCHASTIC (monthly gate + weekly cross) vs Buy&Hold\n", flush=True)
    print(f"{'coin':<9}{'STR_CAGR':>9}{'STR_DD':>8}{'BH_CAGR':>9}{'BH_DD':>8}{'sw':>5}{'inMkt':>7}{'  winner'}")
    print("-" * 66)
    agg_s = []; agg_b = []
    for tkr in COINS:
        try:
            r = run(tkr)
        except Exception as e:
            print(f"{tkr:<9} err {e}"); continue
        if r is None:
            print(f"{tkr:<9} too little data"); continue
        s, b, sw, inm, n = r
        agg_s.append(s[1]); agg_b.append(b[1])
        win = "STRAT" if s[1] > b[1] else "B&H"
        print(f"{tkr.replace('-USD',''):<9}{s[0]:>+8.0f}%{s[2]:>7.0f}%{b[0]:>+8.0f}%"
              f"{b[2]:>7.0f}%{sw:>5}{inm:>6}%   {win}")
    print("-" * 66)
    if agg_s:
        import numpy as _np
        print(f"median total return  ->  STRAT {_np.median(agg_s):+.0f}%   "
              f"B&H {_np.median(agg_b):+.0f}%")
        wins = sum(1 for x, y in zip(agg_s, agg_b) if x > y)
        print(f"strategy beats B&H in {wins}/{len(agg_s)} coins")
    print("\nملاحظة: اختبار على عملات كبرى ناجية = كرم للاستراتيجية؛ الـ alts الصغيرة أسوأ.")
    print("\nDONE_STOCHMTF.", flush=True)


if __name__ == "__main__":
    main()
