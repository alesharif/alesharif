#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monthly trend-timing on crypto (BTC/ETH) vs Buy & Hold.

The user's idea: trade on the MONTHLY timeframe. The classic version (Faber) is
a long-term trend filter: hold the asset while its monthly close is above its
N-month SMA, else move to cash. It trades rarely (low fees) and its real value
is cutting the brutal -70/-80% crypto drawdowns — relevant to a -10% risk
tolerance. We measure CAGR AND max drawdown vs buy & hold, on real data.

Run:  python experiments/monthly_trend.py
"""

from __future__ import annotations

import json, urllib.request
import numpy as np
import pandas as pd

FEE = 0.2          # % per switch (enter or exit)
SMAS = [7, 10, 12]
TICKERS = ["BTC-USD", "ETH-USD"]


def fetch_monthly(tkr):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{tkr}?range=15y&interval=1mo"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    j = json.loads(urllib.request.urlopen(req, timeout=40).read())
    res = j["chart"]["result"][0]
    c = res["indicators"]["quote"][0]["close"]
    return np.array([x for x in c if x is not None], dtype=float)


def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return (equity / peak - 1).min() * 100


def cagr(equity, n_months):
    yrs = n_months / 12
    return (equity[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0


def run(tkr):
    c = fetch_monthly(tkr)
    n = len(c)
    if n < 40:
        print(f"{tkr}: too little data ({n})"); return
    ret = c[1:] / c[:-1] - 1                       # monthly returns
    # buy & hold
    bh_eq = np.cumprod(1 + ret)
    bh_eq = np.concatenate([[1.0], bh_eq])
    print(f"\n### {tkr}  ({n} months ≈ {n/12:.1f}y) ###")
    print(f"{'strategy':<14}{'CAGR':>8}{'totRet':>9}{'maxDD':>8}{'switch':>8}{'inMkt':>7}")
    print(f"{'Buy&Hold':<14}{cagr(bh_eq,n-1):>7.0f}%{(bh_eq[-1]-1)*100:>+8.0f}%"
          f"{max_dd(bh_eq):>7.0f}%{0:>8}{100:>6}%")
    for P in SMAS:
        sma = pd.Series(c).rolling(P).mean().to_numpy()
        eq = 1.0; curve = [1.0]; inmkt = False; switches = 0; months_in = 0
        for t in range(1, n):
            # decision uses info up to month t-1
            signal = (t - 1 >= P) and np.isfinite(sma[t-1]) and (c[t-1] > sma[t-1])
            if signal != inmkt:
                eq *= (1 - FEE/100); switches += 1; inmkt = signal
            if inmkt:
                eq *= (1 + ret[t-1]); months_in += 1
            curve.append(eq)
        curve = np.array(curve)
        print(f"{'SMA'+str(P)+' filter':<14}{cagr(curve,n-1):>7.0f}%{(curve[-1]-1)*100:>+8.0f}%"
              f"{max_dd(curve):>7.0f}%{switches:>8}{months_in*100//(n-1):>6}%")


def main():
    print("MONTHLY trend-timing vs BUY&HOLD — crypto, real data\n", flush=True)
    print(f"rule: hold while monthly close > N-month SMA, else cash. fee {FEE}%/switch.")
    for tkr in TICKERS:
        try:
            run(tkr)
        except Exception as e:
            print(f"{tkr}: err {e}")
    print("\nالمهمّ: هل يقلّص الفلتر maxDD (الهبوط) بوضوح مع حفاظ معقول على CAGR؟")
    print("هذا جوهر الفائدة — حماية، لا تعظيم عائد.")
    print("\nDONE_MONTHLY.", flush=True)


if __name__ == "__main__":
    main()
