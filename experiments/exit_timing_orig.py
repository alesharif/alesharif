#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exit timing: instant (30s) vs wait-for-candle-close (15m/1h/2h/4h).

The user's live bot checks every 30s. Question: would WAITING for a candle
close before acting on the stop/target perform better (less whipsaw) than the
instant 30s check? We compare, on the ORIGINAL strategy entries (4h signal,
base settings), several exit-checking frequencies:

  INSTANT_30s : hybrid engine, exits the moment price touches the stop (1s data)
  CLOSE_15m   : only at each 15m close, act if close<=stop or close>=target-ish
  CLOSE_1h    : only at each 1h close
  CLOSE_2h    : only at each 2h close
  CLOSE_4h    : only at each 4h close

Exit rule for CLOSE_* (user's idea): at the candle close, look at the price NOW:
  - if close <= trailing stop  -> exit
  - else raise trailing stop from this close, keep holding
(No intrabar high/low peeking -> no look-ahead; decision uses the close only.)

ORIGINAL geometry: SL 1.5*ATR, TRAIL 1.5*ATR, ATR cap 12%. Four months.
Run:  python experiments/exit_timing.py
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
TRAIL = 0.2
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
ATR_CAP = 99.0
OUT = "results_exit_timing_orig"
_SEC = {}

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
    "2026-04": ("2026-04-01", "2026-05-01"),
    "2026-05": ("2026-05-01", "2026-06-01"),
}
MODES = ["INSTANT_30s", "CLOSE_15m", "CLOSE_1h", "CLOSE_2h", "CLOSE_4h"]
CLOSE_TF = {"CLOSE_15m": "15m", "CLOSE_1h": "1h", "CLOSE_2h": "2h", "CLOSE_4h": "4h"}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def exit_instant(symbol, et, ep, atr, end):
    """Hybrid 30s: exit the moment price touches the stop."""
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


def exit_on_close(symbol, et, ep, atr, end, tf):
    """Wait-for-close: act only on each tf candle CLOSE (price now)."""
    df = HR.load_range(symbol, tf, et + 1, end)
    if df is None or not len(df):
        return None, None, "OPEN"
    peak = ep; sl = ep - SL_ATR * atr
    for _, c in df.iterrows():
        close = float(c["close"]); ct = int(c["close_time"]) if "close_time" in c else int(c["time"])
        if close <= sl:
            return ct, close, "X"          # exit at the close price
        if close > peak:
            peak = close
            if peak - ep >= TRAIL * atr:
                sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, mode):
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
            if not bool(row["entry_signal"]):
                continue
            ar = float(row["atr"]) / float(row["close"])
            if ar > ATR_CAP:
                continue
            cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            if mode == "INSTANT_30s":
                xt, xp, oc = exit_instant(sym, t, price, atr, end)
            else:
                xt, xp, oc = exit_on_close(sym, t, price, atr, end, CLOSE_TF[mode])
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
    print("EXIT TIMING: instant 30s vs wait-for-close (ORIGINAL strategy, 4 months)\n")
    print(f"{'mode':<13}{'month':<10}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 62)
    for mname, (s, e) in MONTHS.items():
        cells = [f"{OUT}/{mname}__{m}.json" for m in MODES]
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        for mode in MODES:
            cf = f"{OUT}/{mname}__{mode}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); st = {k: d[k] for k in ("n","wr","pf","worst","ret")}
            else:
                nets = simulate(ps, times, end, mode)
                st = stats(nets)
                json.dump({"month": mname, "mode": mode, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
                _SEC.clear(); gc.collect()
            print(f"{mode:<13}{mname:<10}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        print("-" * 62, flush=True)
        del ps, raw; gc.collect()

    # aggregate ranking
    print("\n##### AGGREGATE (4 months) #####")
    print(f"{'mode':<13}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'ret_ALL':>9}   per-month")
    print("-" * 78)
    for mode in MODES:
        an = []; pm = {}
        for m in MONTHS:
            cf = f"{OUT}/{m}__{mode}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); an += d["nets"]; pm[m] = d["ret"]
        if an:
            st = stats(an)
            pms = " ".join(f"{x[2:]}:{pm.get(x,0):+.0f}" for x in MONTHS)
            print(f"{mode:<13}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.1f}%   {pms}", flush=True)
    print("\nDONE_EXIT_TIMING.", flush=True)


if __name__ == "__main__":
    main()
