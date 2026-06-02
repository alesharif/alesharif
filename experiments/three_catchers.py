#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three pump-catchers (from the anatomy of pumps), hybrid 30s backtest.

Anatomy revealed 3 pump types by their pre-pump 7-bar shape:
  type2 BREAKOUT  : already running up -> close > HH10
  type3 SQUEEZE   : flat/quiet then explodes (85% of pumps!) -> ATR was low
                    (<= its own median) then jumps + volume surge
  type1 REBOUND   : dropped >=15% over 7 bars then a strong green bar + volume
Shared quality gate (what separates real pumps from fakes, from the fingerprint):
  ATR/price in a healthy band, ADX>=25, volume>=2x avg, close>EMA50.

Tests each catcher ALONE and the UNION (any catcher fires) on April 2025 (bull)
and December 2025 (hard). SL/TRAIL 1.5, ATR cap 12%. Hybrid 30s exits.

Run:  python experiments/three_catchers.py
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
SL_ATR = 1.5
TRAIL = 1.5
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
ATR_CAP = 0.12
OUT = "results_catchers"
_SEC = {}

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
}
CATCHERS = ["BREAKOUT", "SQUEEZE", "REBOUND", "UNION"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def add_catchers(df):
    df = S.compute_features(df)
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float); v = df["quote_av"].to_numpy(float)
    atr = df["atr"].to_numpy(float); adx = df["adx"].to_numpy(float)
    n = len(c)
    e50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    ar = atr / np.where(c > 0, c, np.nan)
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    atr_med = pd.Series(atr).rolling(20).median().shift(1).to_numpy()  # own ATR baseline
    hh10 = pd.Series(h).rolling(10).max().shift(1).to_numpy()
    ret7 = np.full(n, 0.0)
    ret7[7:] = (c[7:] / c[:-7] - 1) * 100
    ret1 = np.full(n, 0.0)
    ret1[1:] = (c[1:] / c[:-1] - 1) * 100
    rng = np.where((h - l) > 0, h - l, np.nan)
    close_pos = np.nan_to_num((c - l) / rng)

    # shared quality gate (from fingerprint: ATR healthy, ADX, volume, trend)
    okatr = np.isfinite(ar) & (ar <= ATR_CAP) & (atr > 0)
    quality = okatr & (adx >= 25) & (v > 2.0 * vavg) & (c > e50)

    # type2 BREAKOUT: close breaks above prior HH10
    bo = np.nan_to_num(c > hh10).astype(bool)
    # type3 SQUEEZE: prior ATR was quiet (<= its 20-median) then THIS bar pops
    #   (current atr notably above the baseline) + strong green close
    squeeze = np.nan_to_num((atr_med > 0) & (atr > 1.3 * atr_med) &
                            (ret1 > 2.0) & (close_pos >= 0.6)).astype(bool)
    # type1 REBOUND: dropped >=15% over last 7 bars, now a strong green bar
    rebound = np.nan_to_num((ret7 <= -15) & (ret1 > 3.0) & (close_pos >= 0.6)).astype(bool)

    df["BREAKOUT"] = bo & quality
    df["SQUEEZE"] = squeeze & quality
    df["REBOUND"] = rebound & quality
    df["UNION"] = (bo | squeeze | rebound) & quality
    for col in CATCHERS:
        a = df[col].to_numpy(); a[:30] = False; df[col] = a
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
            cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
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
    print("THREE CATCHERS (hybrid 30s)  SL/TRAIL 1.5, ATR<=12%, quality: ADX25+vol2x+>EMA50\n")
    print(f"{'catcher':<10}{'month':<10}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 59)
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: add_catchers(df.copy()).set_index("close_time") for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        for col in CATCHERS:
            cf = f"{OUT}/{mname}__{col}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); st = {k: d[k] for k in ("n","wr","pf","worst","ret")}
            else:
                nets = simulate(ps, times, end, col)
                st = stats(nets)
                json.dump({"month": mname, "catcher": col, **st}, open(cf, "w"))
                _SEC.clear(); gc.collect()
            print(f"{col:<10}{mname:<10}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        print("-" * 59, flush=True)
        del ps, raw; gc.collect()
    print("\nDONE_CATCHERS.", flush=True)


if __name__ == "__main__":
    main()
