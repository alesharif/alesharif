#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test MACD+200EMA on STOCKS (incl. Apple) vs Buy & Hold — the user's point.

The video creator backtested on Apple and it 'worked'. But Apple is a single,
survivor, strong-trending instrument — exactly where trend-following flatters.
The honest checks:
  1) run the SAME strategy on a BASKET (trenders + choppy names), not one stock;
  2) for each, compare to BUY & HOLD (the real benchmark on a rising stock);
  3) include the user's 3-candle micro-divergence filter.

Daily candles from Yahoo. Long: close>EMA200 & MACD cross up below zero.
Short: mirror. SL=swing low/high(10), TP=1.5R, one position at a time, fees.
Run:  python experiments/stock_macd_test.py
"""

from __future__ import annotations

import json, sys, urllib.request
import numpy as np
import pandas as pd

RR = 1.5; SWING = 10; MAXBARS = 60; FEE = 0.05      # %/side
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA",   # strong trenders
           "SPY",                                      # the market
           "INTC", "PFE", "KO", "XOM", "F", "T", "CSCO", "WBA", "BABA"]  # mixed/choppy


def fetch(tkr):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{tkr}?range=15y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    j = json.loads(urllib.request.urlopen(req, timeout=40).read())
    res = j["chart"]["result"][0]; q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"o": q["open"], "h": q["high"], "l": q["low"],
                       "c": q["close"]}).dropna().reset_index(drop=True)
    return df


def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()
def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def run(df, use_div3=False, long_only=False):
    o = df["o"].to_numpy(float); h = df["h"].to_numpy(float)
    l = df["l"].to_numpy(float); c = df["c"].to_numpy(float)
    n = len(c)
    if n < 260:
        return None
    e2 = ema(c, 200); macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9); rs = rsi(c)
    cu = np.concatenate([[False], (macd[:-1] <= sig[:-1]) & (macd[1:] > sig[1:])])
    cd = np.concatenate([[False], (macd[:-1] >= sig[:-1]) & (macd[1:] < sig[1:])])
    Rs = []; i = 210
    while i < n - 1:
        long = (c[i] > e2[i]) and cu[i] and (macd[i] < 0)
        short = (not long_only) and (c[i] < e2[i]) and cd[i] and (macd[i] > 0)
        if not (long or short):
            i += 1; continue
        if use_div3 and i - 3 >= 0:
            pchg = c[i-1]/c[i-3] - 1; rchg = rs[i-1] - rs[i-3]
            ok = (pchg < 0 and rchg > 0) if long else (pchg > 0 and rchg < 0)
            if not ok:
                i += 1; continue
        entry = c[i]
        sl = l[i-SWING:i+1].min() if long else h[i-SWING:i+1].max()
        risk = abs(entry - sl)
        if risk <= 0:
            i += 1; continue
        tp = entry + RR*risk if long else entry - RR*risk
        feeR = 2*FEE/(risk/entry*100)
        outcome, exit_i = None, min(i+MAXBARS, n-1)
        for j in range(i+1, min(i+1+MAXBARS, n)):
            hit_sl = (l[j] <= sl) if long else (h[j] >= sl)
            hit_tp = (h[j] >= tp) if long else (l[j] <= tp)
            if hit_sl: outcome, exit_i = -1.0, j; break
            if hit_tp: outcome, exit_i = RR, j; break
        if outcome is None:
            px = c[exit_i]; outcome = ((px-entry) if long else (entry-px))/risk
        Rs.append(outcome - feeR)
        i = exit_i + 1
    bh = c[-1]/c[210] - 1                       # buy & hold over the traded span
    return np.array(Rs), bh


def main():
    print("MACD+200EMA on STOCKS vs BUY & HOLD (daily, 15y)\n", flush=True)
    print(f"{'ticker':<7}{'nTr':>5}{'WR':>6}{'expR':>7}{'totR':>7}{'B&H':>8}{'  verdict'}")
    print("-" * 60)
    agg = {"totR": 0.0, "n": 0}
    for tkr in TICKERS:
        try:
            res = run(fetch(tkr))
        except Exception as e:
            print(f"{tkr:<7}  fetch/err: {e}"); continue
        if res is None:
            print(f"{tkr:<7}  too little data"); continue
        R, bh = res
        if len(R) == 0:
            print(f"{tkr:<7}    0   — no signals"); continue
        wr = (R > 0).mean()*100; exp = R.mean(); tot = R.sum()
        agg["totR"] += tot; agg["n"] += len(R)
        # rough active return if risking 5% equity/trade, compounding
        verdict = "beats B&H" if tot*0.05 > bh else "LOSES to B&H"
        print(f"{tkr:<7}{len(R):>5}{wr:>5.0f}%{exp:>+6.2f}R{tot:>+6.0f}R{bh*100:>+7.0f}%   {verdict}")
    print("-" * 60)
    print(f"{'ALL':<7}{agg['n']:>5}{'':>6}{'':>7}{agg['totR']:>+6.0f}R")
    print(f"\nB&H = شراء واحتفاظ على نفس المدة. totR = مجموع R للاستراتيجية.")
    print(f"لو totR صغير/سالب بينما B&H ضخم => الاستراتيجية تخسر أمام مجرّد الاحتفاظ.")
    print(f"\nDONE_STOCKS.", flush=True)


if __name__ == "__main__":
    main()
