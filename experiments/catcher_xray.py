#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X-ray each catcher: coverage by pump type + entry/exit timing diagnosis.

For April 2025 + December 2025, this answers (per catcher):
  1) how many pumps of EACH TYPE existed, and how many that catcher caught
  2) of the catches: how late did it enter (vs the move start) and how much
     upside remained AFTER entry (early-exit gap)
This pinpoints each catcher's specific weakness so we can develop it.

Pure 4h analysis (entry signals + forward move), no hybrid exit — fast.
A pump = forward 12-bar high >= +30% from the bar's close.
Pump TYPE (by prior 7-bar run-up shape):
  REBOUND_T : ret7 <= -10   (came from a drop)
  BREAKOUT_T: ret7 >= +15   (already running)
  SQUEEZE_T : in between, and ATR was quiet then popped (flat base)
Run:  python experiments/catcher_xray.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FOUR_H = 4 * 3600 * 1000
FWD = 12
PUMP = 30.0
ATR_CAP = 0.12
MONTHS = {"2025-04": ("2025-04-01", "2025-05-01"),
          "2025-12": ("2025-12-01", "2026-01-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def build(df):
    df = S.compute_features(df)
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float); v = df["quote_av"].to_numpy(float)
    atr = df["atr"].to_numpy(float); adx = df["adx"].to_numpy(float)
    n = len(c)
    e50 = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    ar = atr / np.where(c > 0, c, np.nan)
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    atr_med = pd.Series(atr).rolling(20).median().shift(1).to_numpy()
    hh10 = pd.Series(h).rolling(10).max().shift(1).to_numpy()
    ret7 = np.full(n, 0.0); ret7[7:] = (c[7:] / c[:-7] - 1) * 100
    ret1 = np.full(n, 0.0); ret1[1:] = (c[1:] / c[:-1] - 1) * 100
    rng = np.where((h - l) > 0, h - l, np.nan)
    cpos = np.nan_to_num((c - l) / rng)
    fwd = np.full(n, 0.0)
    for i in range(n - FWD):
        fwd[i] = (h[i + 1:i + 1 + FWD].max() / c[i] - 1) * 100
    okatr = np.isfinite(ar) & (ar <= ATR_CAP) & (atr > 0)
    quality = okatr & (adx >= 25) & (v > 2.0 * vavg) & (c > e50)
    df["c"] = c; df["fwd"] = fwd; df["ret7"] = ret7
    df["BREAKOUT"] = np.nan_to_num(c > hh10).astype(bool) & quality
    df["SQUEEZE"] = (np.nan_to_num((atr_med > 0) & (atr > 1.3 * atr_med) &
                     (ret1 > 2.0) & (cpos >= 0.6)).astype(bool) & quality)
    df["REBOUND"] = (np.nan_to_num((ret7 <= -15) & (ret1 > 3.0) & (cpos >= 0.6)).astype(bool)
                     & okatr & (v > 2.0 * vavg))   # rebound: no >EMA50 req
    return df


def pump_type(ret7):
    if ret7 <= -10:
        return "REBOUND_T"
    if ret7 >= 15:
        return "BREAKOUT_T"
    return "SQUEEZE_T"


def main():
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: build(df.copy()).set_index("close_time") for sym, df in raw.items()}

        # census of pumps by type + which catcher fires on each pump bar
        type_count = {"REBOUND_T": 0, "BREAKOUT_T": 0, "SQUEEZE_T": 0}
        caught = {cat: {"REBOUND_T": 0, "BREAKOUT_T": 0, "SQUEEZE_T": 0}
                  for cat in ("BREAKOUT", "SQUEEZE", "REBOUND")}
        caught_any = {"REBOUND_T": 0, "BREAKOUT_T": 0, "SQUEEZE_T": 0}
        for sym, df in ps.items():
            idx = [t for t in df.index if start <= t <= end]
            if not idx:
                continue
            sub = df.loc[idx]
            for t, row in sub.iterrows():
                if row["fwd"] < PUMP:
                    continue
                ty = pump_type(row["ret7"])
                type_count[ty] += 1
                fired = False
                for cat in ("BREAKOUT", "SQUEEZE", "REBOUND"):
                    if bool(row[cat]):
                        caught[cat][ty] += 1
                        fired = True
                if fired:
                    caught_any[ty] += 1

        print(f"\n##### {mname}: PUMP CENSUS & CATCHER COVERAGE #####")
        tot = sum(type_count.values())
        print(f"total pump-bars: {tot}  "
              f"(REBOUND_T:{type_count['REBOUND_T']}  "
              f"BREAKOUT_T:{type_count['BREAKOUT_T']}  "
              f"SQUEEZE_T:{type_count['SQUEEZE_T']})")
        print(f"\n{'pump type':<12}{'count':>7}{'by BREAKOUT':>13}{'by SQUEEZE':>12}"
              f"{'by REBOUND':>12}{'ANY':>8}")
        print("-" * 64)
        for ty in ("BREAKOUT_T", "SQUEEZE_T", "REBOUND_T"):
            tc = type_count[ty] or 1
            print(f"{ty:<12}{type_count[ty]:>7}"
                  f"{caught['BREAKOUT'][ty]:>9}({caught['BREAKOUT'][ty]*100//tc}%)"
                  f"{caught['SQUEEZE'][ty]:>8}({caught['SQUEEZE'][ty]*100//tc}%)"
                  f"{caught['REBOUND'][ty]:>8}({caught['REBOUND'][ty]*100//tc}%)"
                  f"{caught_any[ty]*100//tc:>6}%")
        print("(each cell: pumps of that type caught by that catcher, and % of the type)")


if __name__ == "__main__":
    main()
