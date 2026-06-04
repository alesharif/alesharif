#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-timeframe: trade WEEKLY but only while the MONTHLY trend is positive.

The user's idea: monthly = regime gate (only be active in a monthly uptrend),
weekly = finer timing (exit faster than the slow monthly signal). Hypothesis:
weekly execution cuts drawdown more than monthly alone, while the monthly gate
avoids whipsaws during bear markets. We compare, on BTC/ETH (weekly equity, so
drawdowns are measured more honestly than month-end):
  B&H | monthly-only(SMA10) | weekly-only(SMA20) | weekly gated by monthly(his)
Run:  python experiments/weekly_monthly.py
"""

from __future__ import annotations

import json, urllib.request
import numpy as np
import pandas as pd

FEE = 0.2; M_SMA = 10; W_SMA = 20
TICKERS = ["BTC-USD", "ETH-USD"]


def fetch_daily(tkr):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{tkr}?range=15y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    j = json.loads(urllib.request.urlopen(req, timeout=40).read())
    res = j["chart"]["result"][0]
    ts = res["timestamp"]; c = res["indicators"]["quote"][0]["close"]
    s = pd.Series(c, index=pd.to_datetime(ts, unit="s")).dropna()
    return s


def stats(eq, n_weeks):
    peak = np.maximum.accumulate(eq); dd = (eq / peak - 1).min() * 100
    yrs = n_weeks / 52
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    return cagr, (eq[-1] - 1) * 100, dd


def equity(ret, inmkt):
    eq = 1.0; curve = [1.0]; sw = 0; prev = False; inm = 0
    for t in range(len(ret)):
        if inmkt[t] != prev:
            eq *= (1 - FEE/100); sw += 1; prev = inmkt[t]
        if inmkt[t]:
            eq *= (1 + ret[t]); inm += 1
        curve.append(eq)
    return np.array(curve), sw, inm


def run(tkr):
    s = fetch_daily(tkr)
    wk = s.resample("W").last().dropna()
    mo = s.resample("ME").last().dropna()
    wret = wk.pct_change().fillna(0).to_numpy()
    # weekly trend (decision known at start of week -> shift 1)
    wbull = (wk > wk.rolling(W_SMA).mean()).shift(1).fillna(False)
    # monthly trend, lagged, aligned to weekly index via ffill
    mbull = (mo > mo.rolling(M_SMA).mean()).shift(1)
    mbull_w = mbull.reindex(wk.index, method="ffill").fillna(False)
    n = len(wk)

    bh, _, _ = equity(wret, np.ones(n, bool))
    res = {}
    res["Buy&Hold"] = stats(bh, n) + (0, 100)
    for name, inm in [
        ("monthly only", mbull_w.to_numpy().astype(bool)),
        ("weekly only", wbull.to_numpy().astype(bool)),
        ("weekly+monthly", (wbull.to_numpy() & mbull_w.to_numpy()).astype(bool)),
    ]:
        eq, sw, im = equity(wret, inm)
        res[name] = stats(eq, n) + (sw, im * 100 // n)

    print(f"\n### {tkr}  ({n} weeks ≈ {n/52:.1f}y) ###")
    print(f"{'strategy':<16}{'CAGR':>7}{'totRet':>10}{'maxDD':>8}{'switch':>8}{'inMkt':>7}")
    for k, v in res.items():
        print(f"{k:<16}{v[0]:>6.0f}%{v[1]:>+9.0f}%{v[2]:>7.0f}%{v[3]:>8}{v[4]:>6}%")


def main():
    print("MULTI-TIMEFRAME: weekly timing gated by monthly trend — vs alternatives\n", flush=True)
    print(f"monthly SMA{M_SMA}, weekly SMA{W_SMA}, fee {FEE}%/switch. Drawdown measured weekly.")
    for tkr in TICKERS:
        try:
            run(tkr)
        except Exception as e:
            print(f"{tkr}: err {e}")
    print("\nنبحث: هل 'أسبوعي+شهري' يخفض maxDD أكثر من الشهري وحده، مع عائد معقول؟")
    print("\nDONE_WM.", flush=True)


if __name__ == "__main__":
    main()
