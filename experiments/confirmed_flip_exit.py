#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CONFIRMED-FLIP exit — drop X% from running peak + persist Y min -> exit.

Built on the post-peak finding (tops are FINAL 78%, the flip persists ~7h while
mid-pump pullbacks recover fast). So a percentage trailing stop with a
CONFIRMATION DELAY should catch the real top (which stays down) while ignoring
brief pullbacks (which recover before the delay elapses) — possibly capturing
closer to the peak than the slow EMA60_20.

Entry base = our best: 4h signal + fear 1.15 + atr_ratio>=5% & adx>=40.
On each trade's real 1m path we compare:
  EMA60_20           : benchmark (1h close < EMA20) + 2ATR stop
  FLIP_d%_cYY        : exit when close < peak*(1-d) for YY consecutive minutes,
                       reset on a new high; + 2ATR hard stop.
Run:  python experiments/confirmed_flip_exit.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
MAX_CONC = 8
FOUR = 4 * 3600 * 1000
HOLD_H = 48
HARD_SL_ATR = 2.0
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}

# confirmed-flip variants: (label, drop_frac, confirm_minutes)
FLIPS = [("FLIP_3%_c15", 0.03, 15), ("FLIP_3%_c30", 0.03, 30),
         ("FLIP_5%_c15", 0.05, 15), ("FLIP_5%_c30", 0.05, 30),
         ("FLIP_5%_c60", 0.05, 60), ("FLIP_8%_c30", 0.08, 30)]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def exit_ema60(tmin, hi, lo, cl, ep, atr):
    sl = ep - HARD_SL_ATR * atr
    sl_idx = None
    for i in range(len(lo)):
        if lo[i] <= sl:
            sl_idx = i; break
    n = len(cl); grp = (tmin - tmin[0]) // (60 * 60 * 1000)
    C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            C.append(cl[m][-1]); T.append(tmin[m][-1])
    C = np.array(C); T = np.array(T)
    et = epx = None
    if len(C) >= 21:
        e = ema(C, 20)
        for j in range(21, len(C)):
            if C[j] < e[j]:
                et = T[j]; epx = C[j]; break
    sl_t = tmin[sl_idx] if sl_idx is not None else None
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_idx]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def exit_flip(hi, lo, cl, ep, atr, drop, confirm):
    """Drop `drop` from running peak + stay below for `confirm` minutes -> exit."""
    peak = ep; sl = ep - HARD_SL_ATR * atr
    trigger = peak * (1 - drop); below = 0
    for i in range(len(cl)):
        if lo[i] <= sl:
            return (min(sl, cl[i]) / ep - 1) * 100 - FEE
        if hi[i] > peak:
            peak = hi[i]; trigger = peak * (1 - drop); below = 0
        if cl[i] < trigger:
            below += 1
            if below >= confirm:
                return (cl[i] / ep - 1) * 100 - FEE
        else:
            below = 0
    return (cl[-1] / ep - 1) * 100 - FEE


def main():
    cols = ["EMA60_20"] + [f[0] for f in FLIPS]
    acc = {c: {m: [] for m in MONTHS} for c in cols}
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e); ff = start - PB.WARMUP_BARS * FOUR
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        sr = PB.build_stable_ratio(ff, end)
        sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
        open_until = {}
        for t in times:
            open_until = {sy: u for sy, u in open_until.items() if u > t}
            if sblock.get(t, False) or len(open_until) >= MAX_CONC:
                continue
            cands = []
            for sym, df in ps.items():
                if sym in open_until or t not in df.index:
                    continue
                row = df.loc[t]
                if not bool(row["entry_signal"]):
                    continue
                price = float(row["close"]); atr = float(row["atr"]); adx = float(row["adx"])
                if atr / price < ATR_MIN or adx < ADX_MIN:
                    continue
                cands.append((sym, price, atr, float(row["vol_pit"])))
            cands.sort(key=lambda x: x[3], reverse=True)
            for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 60:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                acc["EMA60_20"][mname].append(exit_ema60(tmin, hi, lo, cl, price, atr))
                for lbl, dr, cf in FLIPS:
                    acc[lbl][mname].append(exit_flip(hi, lo, cl, price, atr, dr, cf))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### CONFIRMED-FLIP vs EMA60_20 (base +7.5% entries) #####")
    print(f"{'exit':<14}{'trades':>7}{'WR':>6}{'PF':>7}{'avg%':>8}{'ret_ALL':>9}   per-month")
    print("-" * 82)
    for c in cols:
        alln = [x for m in MONTHS for x in acc[c][m]]
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        pm = " ".join(f"{m[2:]}:{np.array(acc[c][m]).sum()*0.125:+.0f}" for m in MONTHS)
        print(f"{c:<14}{len(a):>7}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{a.mean():>7.2f}%"
              f"{a.sum()*0.125:>8.1f}%   {pm}", flush=True)
    print("\nDONE_FLIP_EXIT.", flush=True)


if __name__ == "__main__":
    main()
