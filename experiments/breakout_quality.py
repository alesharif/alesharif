#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BREAKOUT + QUALITY FILTERS: separate the real breakouts from the fakes.

BO10 caught 94% of movers but WR was only ~45% (half are false breakouts).
This adds quality gates to keep early entry while filtering fakes, testing
each gate's effect on WR / PF / return. Trail fixed at 1.5 (proven best).

Base entry: close > HH10 & vol>1.5x & ATR<=12%  (BO10)
Quality variants (each ANDed onto BO10):
  BO10            : base (reference)
  +ADX25          : ADX >= 25 (trend strength, from your original strategy)
  +VOL3           : volume > 3x its 20-avg (real explosion, not 1.5x)
  +EMA200         : close > EMA200 (overall uptrend)
  +STRONG_CLOSE   : close in top 30% of the bar's range (not a wick)
  +ALL            : ADX25 & VOL3 & EMA200 & strong-close combined

Four months, hybrid 30s exits, SL/TRAIL 1.5. Saves per-cell, resumable.
Run:  python experiments/breakout_quality.py
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
OUT = "results_quality"
_SEC = {}

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
    "2026-04": ("2026-04-01", "2026-05-01"),
    "2026-05": ("2026-05-01", "2026-06-01"),
}

VARIANTS = ["BO10", "ADX25", "VOL3", "EMA200", "STRONG", "ALL"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def add_signals(df):
    df = S.compute_features(df)
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float); v = df["quote_av"].to_numpy(float)
    atr = df["atr"].to_numpy(float); adx = df["adx"].to_numpy(float)
    ab = df["above_ema200"].to_numpy(float)
    ar = atr / np.where(c > 0, c, np.nan)
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    hh = pd.Series(h).rolling(10).max().shift(1).to_numpy()
    okatr = np.isfinite(ar) & (ar <= ATR_CAP) & (atr > 0)
    base = np.nan_to_num((c > hh) & (v > 1.5 * vavg) & okatr).astype(bool)
    base[:30] = False
    # quality components
    q_adx = np.nan_to_num(adx >= 25)
    q_vol3 = np.nan_to_num(v > 3.0 * vavg)
    q_ema = ab > 0.5
    rng = np.where((h - l) > 0, h - l, np.nan)
    strong = np.nan_to_num((c - l) / rng >= 0.70)   # close in top 30%
    df["BO10"] = base
    df["ADX25"] = base & q_adx
    df["VOL3"] = base & q_vol3
    df["EMA200"] = base & q_ema
    df["STRONG"] = base & strong
    df["ALL"] = base & q_adx & q_vol3 & q_ema & strong
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


def load(s_str, e_str):
    start, end = parse(s_str), parse(e_str)
    ff = start - PB.WARMUP_BARS * FOUR_H
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: add_signals(df.copy()).set_index("close_time") for s, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    return ps, times, end


def main():
    os.makedirs(OUT, exist_ok=True)
    for mname, (s, e) in MONTHS.items():
        cells = [f"{OUT}/{mname}__{v}.json" for v in VARIANTS]
        if all(os.path.exists(c) for c in cells):
            print(f"{mname}: all cached, skip", flush=True); continue
        print(f"\n##### {mname} #####", flush=True)
        ps, times, end = load(s, e)
        print(f"  loaded {len(ps)} symbols", flush=True)
        for v in VARIANTS:
            cf = f"{OUT}/{mname}__{v}.json"
            if os.path.exists(cf):
                continue
            nets = simulate(ps, times, end, v)
            st = stats(nets)
            json.dump({"month": mname, "variant": v, **st,
                       "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
            print(f"  {mname} {v:<8} n{st['n']:<4} WR{st['wr']:.0f}% "
                  f"PF{st['pf']} worst{st['worst']}% ret{st['ret']}%", flush=True)
            _SEC.clear(); gc.collect()
        del ps; gc.collect()

    # aggregate ranking
    print("\n##### RANKING (4-month aggregate) #####", flush=True)
    print(f"{'variant':<9}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'ret_ALL':>9}   per-month")
    print("-" * 80)
    agg = {}
    for v in VARIANTS:
        an = []; pm = {}
        for m in MONTHS:
            cf = f"{OUT}/{m}__{v}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); an += d["nets"]; pm[m] = d["ret"]
        if an:
            agg[v] = (stats(an), pm)
    for v, (st, pm) in sorted(agg.items(), key=lambda kv: kv[1][0]["ret"], reverse=True):
        pms = " ".join(f"{m.split('-')[0][2:]}{m.split('-')[1]}:{pm.get(m,0):+.0f}" for m in MONTHS)
        print(f"{v:<9}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
              f"{st['worst']:>8.1f}%{st['ret']:>8.1f}%   {pms}", flush=True)
    print("\nDONE_QUALITY. results in results_quality/", flush=True)


if __name__ == "__main__":
    main()
