#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PUMP CATCHER built from the user's proven manual pattern (KuCoin success).

The user manually caught pumps (+50% in 15min) by watching, on 15m candles:
  1) a coin that recently pumped hard (e.g. +60%) then came back / accumulated
  2) MACD histogram turning green and staying green for ~10+ 15m bars
  3) the buyer/seller range tightening (a squeeze) -> readiness
  4) a sudden velocity burst (price/orders changing fast) -> ignition
  5) volume confirming

This encodes that as a 15m-entry catcher (orderbook part is live-only). Entry on
15m close; exit via hybrid 30s (5m scan + 1s zoom). Tested on April 2025 (rich)
first; we'll validate on other months next.

Variants tested (to see which conditions matter):
  FULL   : all of (recent-pump + macd-green-run + squeeze + velocity + volume)
  NOSQZ  : drop the squeeze condition
  NOVEL  : drop the velocity condition
  NOHIST : drop the recent-pump history condition

Run:  python experiments/pump_catcher_v3.py
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
SL_ATR = 1.5          # exit on 15m ATR
TRAIL = 1.5
POS_USD = 250.0
MAX_CONC = 8
FIFTEEN = 15 * 60 * 1000
OUT = "results_v3"
_SEC = {}

MONTHS = {"2025-04": ("2025-04-01", "2025-05-01")}
VARIANTS = ["FULL", "NOSQZ", "NOVEL", "NOHIST"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def atr15(h, l, c, period=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def add_signals(df):
    """df is 15m OHLCV. Build the user's pattern conditions."""
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float); v = df["volume"].to_numpy(float)
    n = len(c)
    cs = pd.Series(c)
    atr = atr15(h, l, c)
    # MACD (12,26,9) histogram on 15m
    macd = cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = (macd - signal).to_numpy()
    # (2) MACD histogram green for last ~8 bars (turning/accumulating up)
    green = (hist > 0).astype(float)
    green_run = pd.Series(green).rolling(8).sum().shift(1).to_numpy()  # how many of last 8 green
    macd_green = green_run >= 5      # majority green recently
    # (3) squeeze: Bollinger bandwidth contracted (<= its own recent low band)
    ma20 = cs.rolling(20).mean(); sd20 = cs.rolling(20).std()
    bbw = ((4 * sd20) / ma20 * 100).to_numpy()
    bbw_q = pd.Series(bbw).rolling(50).quantile(0.30).shift(1).to_numpy()  # quiet zone
    squeeze = bbw <= bbw_q          # currently coiled
    # (4) velocity burst: price up fast over last 4 bars (1h) suddenly
    vel4 = np.full(n, 0.0); vel4[4:] = (c[4:] / c[:-4] - 1) * 100
    velocity = vel4 >= 4.0          # +4% in last hour = igniting
    # (5) volume surge
    vavg = pd.Series(v).rolling(40).mean().shift(1).to_numpy()
    vol_surge = v > 2.0 * vavg
    # (1) recently pumped: max high over last ~7 days (672 bars of 15m) >= +50%
    LB = 672
    roll_hi = pd.Series(h).rolling(LB).max().shift(1).to_numpy()
    roll_lo = pd.Series(l).rolling(LB).min().shift(1).to_numpy()
    recent_pump = np.nan_to_num((roll_hi / np.where(roll_lo > 0, roll_lo, np.nan) - 1) * 100) >= 50

    base_quality = vol_surge & (atr > 0)
    df["FULL"] = recent_pump & macd_green & squeeze & velocity & base_quality
    df["NOSQZ"] = recent_pump & macd_green & velocity & base_quality
    df["NOVEL"] = recent_pump & macd_green & squeeze & base_quality
    df["NOHIST"] = macd_green & squeeze & velocity & base_quality
    df["atr15"] = atr
    for col in VARIANTS:
        a = np.nan_to_num(df[col]).astype(bool); a[:LB] = False; df[col] = a
    return df


def hybrid_exit(symbol, et, ep, atr, end):
    peak = ep; sl = ep - SL_ATR * atr; trailing = False
    five = HR.load_range(symbol, "5m", et + 1, end)
    if five is None or not len(five):
        return None, None, "OPEN"
    margin = 1.0 * atr
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if not ((lo <= sl + margin) or (hi >= peak)):
            if hi > peak:
                peak = hi
                if peak - ep >= TRAIL * atr:
                    trailing = True; sl = max(sl, peak - TRAIL * atr)
            continue
        day = pd.Timestamp(int(c["time"]), unit="ms").strftime("%Y-%m-%d")
        s = sec_day(symbol, day)
        if s is not None and len(s):
            seg = s[(s["time"] >= c["time"]) & (s["time"] <= c["close_time"])]
            if len(seg):
                for _, s1 in seg.iloc[::30].iterrows():
                    ph, pl = float(s1["high"]), float(s1["low"])
                    if ph > peak:
                        peak = ph
                        if peak - ep >= TRAIL * atr:
                            trailing = True; sl = max(sl, peak - TRAIL * atr)
                    if pl <= sl:
                        return int(s1["time"]), sl, "X"
                continue
        if lo <= sl:
            return int(c["close_time"]), sl, "X"
        if hi > peak:
            peak = hi
            if peak - ep >= TRAIL * atr:
                trailing = True; sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, col):
    open_until = {}; nets = []
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
        if len(open_until) >= MAX_CONC:
            continue
        cands = []
        for sym, df in ps.items():
            if sym in open_until or t not in df.index:
                continue
            row = df.loc[t]
            if not bool(row[col]):
                continue
            cands.append((sym, float(row["close"]), float(row["atr15"]), float(row["volume"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            if atr <= 0:
                continue
            xt, xp, oc = hybrid_exit(sym, t, price, atr, end)
            if oc == "OPEN":
                open_until[sym] = None
            else:
                nets.append((xp / price - 1) * 100 - FEE)
                open_until[sym] = xt
    return nets


def stats(nets):
    wins = [n for n in nets if n > 0]; losses = [n for n in nets if n <= 0]
    gw = sum(wins); gl = -sum(losses)
    pf = gw / gl if gl else float("inf")
    wr = len(wins) / len(nets) * 100 if nets else 0
    ret = sum(POS_USD * n / 100 for n in nets) / 2000 * 100
    return dict(n=len(nets), wr=round(wr, 0), pf=round(pf, 2),
                worst=round(min(nets, default=0), 1), ret=round(ret, 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("PUMP CATCHER v3 (user's manual pattern) — 15m entry, hybrid 30s exit\n")
    print(f"{'variant':<8}{'month':<10}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 57)
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - 8 * 24 * 3600 * 1000   # 8 days warm-up (for 7-day history)
        syms = PIT.list_all_usdt_symbols()
        ps = {}
        for sym in syms:
            d = HR.load_range(sym, "15m", ff, end)
            if d is None or len(d) < 700:
                continue
            ps[sym] = add_signals(d).set_index("close_time")
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        print(f"  ({mname}: {len(ps)} symbols, {len(times)} bars)", flush=True)
        for col in VARIANTS:
            nets = simulate(ps, times, end, col)
            st = stats(nets)
            json.dump({"month": mname, "variant": col, **st}, open(f"{OUT}/{mname}__{col}.json", "w"))
            print(f"{col:<8}{mname:<10}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
            _SEC.clear(); gc.collect()
        del ps; gc.collect()
    print("\nDONE_V3.", flush=True)


if __name__ == "__main__":
    main()
