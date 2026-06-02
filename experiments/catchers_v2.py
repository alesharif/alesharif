#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Catchers v2 — richer toolset (Bollinger squeeze, Keltner, BB width) to catch
the SQUEEZE-type pumps (74% of pumps, currently caught ~1%).

New tools added:
  * Bollinger Bands (20, 2sd) + bandwidth
  * Keltner Channels (20, 1.5*ATR)
  * Bollinger Squeeze = BB inside Keltner (classic low-vol coil), then release
  * Donchian breakout (HH15)

Catchers (entry on 4h close, then hybrid 30s exit):
  SQ_REL  : was in a squeeze in the last few bars, now releasing UP
            (BB width expanding + close breaks the upper band) + volume
  BO15    : Donchian HH15 breakout + close>upper BB + volume
  REBND   : RSI was <35 then turns up + strong green bar + volume (no >EMA50)
  UNION3  : any of the above
Quality gate kept light (ATR cap + volume) so we don't re-block the quiet base.

Tested April 2025 + December 2025. Run:  python experiments/catchers_v2.py
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
OUT = "results_v2"
_SEC = {}

MONTHS = {"2025-04": ("2025-04-01", "2025-05-01"),
          "2025-12": ("2025-12-01", "2026-01-01")}
CATCHERS = ["SQ_REL", "BO15", "REBND", "UNION3"]


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
    atr = df["atr"].to_numpy(float); rsi = df["rsi"].to_numpy(float)
    n = len(c)
    cs = pd.Series(c)
    # Bollinger Bands (20, 2sd)
    ma20 = cs.rolling(20).mean()
    sd20 = cs.rolling(20).std()
    bb_up = (ma20 + 2 * sd20).to_numpy()
    bb_lo = (ma20 - 2 * sd20).to_numpy()
    bb_w = ((bb_up - bb_lo) / ma20.to_numpy()) * 100      # bandwidth %
    bb_w_prev = pd.Series(bb_w).shift(1).to_numpy()
    bb_w_med = pd.Series(bb_w).rolling(50).median().shift(1).to_numpy()
    # Keltner (20 ema, 1.5 ATR)
    ke = cs.ewm(span=20, adjust=False).mean().to_numpy()
    k_up = ke + 1.5 * atr
    k_lo = ke - 1.5 * atr
    # Bollinger squeeze ON when BB is inside Keltner (low volatility coil)
    in_sqz = (bb_up < k_up) & (bb_lo > k_lo)
    sqz_recent = pd.Series(in_sqz.astype(float)).rolling(6).max().shift(1).to_numpy()  # squeezed in last 6
    # Donchian
    hh15 = cs.rolling(15).max().shift(1).to_numpy()       # note: uses close highs proxy
    hhraw = pd.Series(h).rolling(15).max().shift(1).to_numpy()
    ret1 = np.full(n, 0.0); ret1[1:] = (c[1:] / c[:-1] - 1) * 100
    rng = np.where((h - l) > 0, h - l, np.nan)
    cpos = np.nan_to_num((c - l) / rng)
    ar = atr / np.where(c > 0, c, np.nan)
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    rsi_prev = pd.Series(rsi).shift(1).to_numpy()

    okatr = np.isfinite(ar) & (ar <= ATR_CAP) & (atr > 0)
    okvol = v > 2.0 * vavg

    # SQ_REL: was coiled recently, now band expands & breaks upper BB, with volume
    sq_rel = np.nan_to_num((sqz_recent > 0) & (bb_w > bb_w_prev) &
                           (c > bb_up) & okvol & okatr).astype(bool)
    # BO15: Donchian HH15 breakout + above upper BB + volume
    bo15 = np.nan_to_num((c > hhraw) & (c > bb_up) & okvol & okatr).astype(bool)
    # REBND: RSI rose out of oversold + strong green bar + volume
    rebnd = np.nan_to_num((rsi_prev < 35) & (rsi > rsi_prev) & (ret1 > 3) &
                          (cpos >= 0.6) & okvol & okatr).astype(bool)

    df["SQ_REL"] = sq_rel
    df["BO15"] = bo15
    df["REBND"] = rebnd
    df["UNION3"] = sq_rel | bo15 | rebnd
    for col in CATCHERS:
        a = df[col].to_numpy().copy(); a[:60] = False; df[col] = a
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
    print("CATCHERS v2 (Bollinger squeeze + Keltner + Donchian)  hybrid 30s\n")
    print(f"{'catcher':<9}{'month':<10}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 58)
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
            print(f"{col:<9}{mname:<10}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        print("-" * 58, flush=True)
        del ps, raw; gc.collect()
    print("\nDONE_V2.", flush=True)


if __name__ == "__main__":
    main()
