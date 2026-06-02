#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune the ALL strategy: one-at-a-time sweep of the 3 key entry params.

Baseline ALL = HH10 + vol>=1.5x + ADX>=25 + vol>=3x + close>EMA200 +
strong-close(>=70%) + ATR<=12% + SL/TRAIL 1.5.

Sweeps (each varied alone, rest fixed at baseline), all on the FOUR months,
hybrid 30s exits, aggregate + per-month:
  break_n : 8, 10, 15, 20
  adx_min : 20, 25, 30
  vol_x   : 2.0, 3.0, 4.0

Saves per-cell (resumable). Run:  python experiments/tune_all.py
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
OUT = "results_tune"
_SEC = {}

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
    "2026-04": ("2026-04-01", "2026-05-01"),
    "2026-05": ("2026-05-01", "2026-06-01"),
}

# baseline params
BASE = dict(break_n=10, adx_min=25, vol_x=3.0, strong=0.70)

# variations to test (label -> param override)
VARIANTS = [
    ("baseline", {}),
    ("HH8",  {"break_n": 8}),
    ("HH15", {"break_n": 15}),
    ("HH20", {"break_n": 20}),
    ("ADX20", {"adx_min": 20}),
    ("ADX30", {"adx_min": 30}),
    ("VOL2", {"vol_x": 2.0}),
    ("VOL4", {"vol_x": 4.0}),
]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def add_signal(df, p):
    """Attach 'sig' for the ALL strategy with params p."""
    df = S.compute_features(df)
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float); v = df["quote_av"].to_numpy(float)
    atr = df["atr"].to_numpy(float); adx = df["adx"].to_numpy(float)
    ab = df["above_ema200"].to_numpy(float)
    ar = atr / np.where(c > 0, c, np.nan)
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    hh = pd.Series(h).rolling(p["break_n"]).max().shift(1).to_numpy()
    rng = np.where((h - l) > 0, h - l, np.nan)
    strong = (c - l) / rng
    sig = ((c > hh) & (v > 1.5 * vavg) & np.isfinite(ar) & (ar <= ATR_CAP) & (atr > 0) &
           (adx >= p["adx_min"]) & (v > p["vol_x"] * vavg) & (ab > 0.5) &
           np.nan_to_num(strong >= p["strong"]))
    sig = np.nan_to_num(sig).astype(bool)
    sig[:max(30, p["break_n"])] = False
    df["sig"] = sig
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


def simulate(ps, times, end):
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
            if not bool(row["sig"]):
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
    # process month by month: for each month, build signals per variant, simulate
    for mname, (s, e) in MONTHS.items():
        cells = [f"{OUT}/{mname}__{lbl}.json" for lbl, _ in VARIANTS]
        if all(os.path.exists(c) for c in cells):
            print(f"{mname}: cached, skip", flush=True); continue
        print(f"\n##### {mname} #####", flush=True)
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        print(f"  loaded {len(raw)} symbols", flush=True)
        for lbl, ov in VARIANTS:
            cf = f"{OUT}/{mname}__{lbl}.json"
            if os.path.exists(cf):
                continue
            p = dict(BASE, **ov)
            ps = {sym: add_signal(df.copy(), p).set_index("close_time") for sym, df in raw.items()}
            times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
            nets = simulate(ps, times, end)
            st = stats(nets)
            json.dump({"month": mname, "variant": lbl, **st,
                       "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
            print(f"  {mname} {lbl:<9} n{st['n']:<4} WR{st['wr']:.0f}% "
                  f"PF{st['pf']} worst{st['worst']}% ret{st['ret']}%", flush=True)
            del ps; _SEC.clear(); gc.collect()
        del raw; gc.collect()

    # ranking
    print("\n##### TUNE RANKING (4-month aggregate) #####", flush=True)
    print(f"{'variant':<10}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'ret_ALL':>9}   per-month", flush=True)
    print("-" * 82)
    agg = {}
    for lbl, _ in VARIANTS:
        an = []; pm = {}
        for m in MONTHS:
            cf = f"{OUT}/{m}__{lbl}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); an += d["nets"]; pm[m] = d["ret"]
        if an:
            agg[lbl] = (stats(an), pm)
    for lbl, (st, pm) in sorted(agg.items(), key=lambda kv: kv[1][0]["ret"], reverse=True):
        pms = " ".join(f"{m[2:]}:{pm.get(m,0):+.0f}" for m in MONTHS)
        print(f"{lbl:<10}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
              f"{st['worst']:>8.1f}%{st['ret']:>8.1f}%   {pms}", flush=True)
    print("\nDONE_TUNE. results in results_tune/", flush=True)


if __name__ == "__main__":
    main()
