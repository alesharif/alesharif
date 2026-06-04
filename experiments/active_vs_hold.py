#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Active MACD+200EMA (long-only) vs BUY&HOLD vs DCA — on SPY/QQQ/AAPL, 15y daily.

The user reads the stock results as a 'success'. They are positive-expectancy on
strong trenders — but the honest question is whether ACTIVELY trading beats
simply HOLDING. This puts real equity curves side by side, giving the active
strategy its best shot (long-only, several risk levels) and measuring how much
of the time it even sits in the market.

Run:  python experiments/active_vs_hold.py
"""

from __future__ import annotations

import json, urllib.request
import numpy as np
import pandas as pd

RR = 1.5; SWING = 10; MAXBARS = 60; FEE = 0.05
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT"]


def fetch(tkr):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{tkr}?range=15y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    j = json.loads(urllib.request.urlopen(req, timeout=40).read())
    res = j["chart"]["result"][0]; q = res["indicators"]["quote"][0]
    return pd.DataFrame({"o": q["open"], "h": q["h" if "h" in q else "high"],
                         "l": q["low"], "c": q["close"]}).dropna().reset_index(drop=True)


def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()
def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def active(df, risk):
    """Long-only MACD+200EMA. Return (final_mult, n_trades, days_in_mkt, total_days)."""
    o = df["o"].to_numpy(float); h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float); c = df["c"].to_numpy(float)
    n = len(c)
    e2 = ema(c, 200); macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    cu = np.concatenate([[False], (macd[:-1] <= sig[:-1]) & (macd[1:] > sig[1:])])
    eq = 1.0; i = 210; ntr = 0; dim = 0
    while i < n - 1:
        if not ((c[i] > e2[i]) and cu[i] and (macd[i] < 0)):
            i += 1; continue
        entry = c[i]; sl = l[i-SWING:i+1].min(); risk_px = entry - sl
        if risk_px <= 0:
            i += 1; continue
        tp = entry + RR * risk_px; feeR = 2*FEE/(risk_px/entry*100)
        outcome, exit_i = None, min(i+MAXBARS, n-1)
        for j in range(i+1, min(i+1+MAXBARS, n)):
            if l[j] <= sl: outcome, exit_i = -1.0, j; break
            if h[j] >= tp: outcome, exit_i = RR, j; break
        if outcome is None:
            outcome = (c[exit_i]-entry)/risk_px
        eq *= (1 + risk * (outcome - feeR)); ntr += 1; dim += exit_i - i
        i = exit_i + 1
    return eq, ntr, dim, n - 210


def main():
    print("ACTIVE (MACD+200EMA, long-only) vs BUY&HOLD vs DCA — 15y daily\n", flush=True)
    print(f"{'ticker':<7}{'B&H':>9}{'DCA*':>9}{'act@5%':>9}{'act@10%':>9}{'act@20%':>10}{'  inMkt'}")
    print("-" * 64)
    for tkr in TICKERS:
        try:
            df = fetch(tkr)
        except Exception as e:
            print(f"{tkr:<7}  err {e}"); continue
        c = df["c"].to_numpy(float); n = len(c)
        years = (n - 210) / 252
        bh = c[-1] / c[210] - 1
        # DCA: invest $1 every 21 trading days from bar 210
        idxs = list(range(210, n, 21)); shares = sum(1.0 / c[k] for k in idxs)
        dca = shares * c[-1] / len(idxs) - 1
        a5 = active(df, 0.05); a10 = active(df, 0.10); a20 = active(df, 0.20)
        inmkt = a5[2] / a5[3] * 100
        print(f"{tkr:<7}{bh*100:>+8.0f}%{dca*100:>+8.0f}%{(a5[0]-1)*100:>+8.0f}%"
              f"{(a10[0]-1)*100:>+8.0f}%{(a20[0]-1)*100:>+9.0f}%{inmkt:>6.0f}%")
    print("-" * 64)
    print("*DCA = شراء دوري ثابت (العائد على متوسّط المُستثمَر). act@X% = نشط بمخاطرة X%/صفقة.")
    print("inMkt = نسبة الوقت داخل السوق فعلياً (الباقي كاش = يفوّت الصعود).")
    print("\nلاحظ: act@20% مخاطرة عالية جداً (قد يُفلس)؛ موضوع فقط ليأخذ النشط أفضل فرصة.")
    print("\nDONE_ACTIVE_HOLD.", flush=True)


if __name__ == "__main__":
    main()
