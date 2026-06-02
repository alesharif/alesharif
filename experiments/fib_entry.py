#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fibonacci-retracement ENTRY vs immediate entry — does it cut the -6% heat?

For each taken trade (4h signal + fear gate 1.15 + atr>=5%+adx>=40), instead of
entering at the 4h close we place a limit BUY at a Fib retracement of the recent
swing and wait up to WAIT_H hours for a fill. Same exit for all (EMA60_20 on 1h
+ 2*ATR protective stop). We compare entry methods on identical trade slots:

  IMMEDIATE : enter at the 4h close (current behaviour)
  FIB_50    : limit buy at 50.0% retracement of the last SWING_BARS swing
  FIB_618   : limit buy at 61.8% retracement
  FIB_382   : limit buy at 38.2% retracement

Metrics: fill rate, avg net%, WR, PF, mean MAE (heat), per-month return. If Fib
entry lowers MAE and raises PF, entering on the pullback beats chasing.
Run:  python experiments/fib_entry.py
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
FOUR_H = 4 * 3600 * 1000
HOLD_H = 48
WAIT_H = 12              # how long to wait for the retracement fill
SWING_BARS = 12         # 4h bars (~2 days) to define the recent swing
HARD_SL_ATR = 2.0
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
FIBS = {"IMMEDIATE": None, "FIB_382": 0.382, "FIB_50": 0.5, "FIB_618": 0.618}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def resample_close(tmin, cl, tf_min):
    n = len(cl); grp = np.arange(n) // tf_min
    C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            C.append(cl[m][-1]); T.append(tmin[m][-1])
    return np.array(C), np.array(T)


def exit_ema60(tmin, hi, lo, cl, ep, atr):
    """EMA60_20 exit: 2*ATR hard stop (1m) + first 1h close below EMA20."""
    sl = ep - HARD_SL_ATR * atr
    sl_idx = None
    for i in range(len(lo)):
        if lo[i] <= sl:
            sl_idx = i; break
    C, T = resample_close(tmin, cl, 60)
    ema_t = ema_px = None
    if len(C) >= 21:
        e = ema(C, 20)
        for j in range(21, len(C)):
            if C[j] < e[j]:
                ema_t = T[j]; ema_px = C[j]; break
    sl_t = tmin[sl_idx] if sl_idx is not None else None
    if sl_t is not None and (ema_t is None or sl_t <= ema_t):
        return (min(sl, cl[sl_idx]) / ep - 1) * 100 - FEE
    if ema_t is not None:
        return (ema_px / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def main():
    acc = {k: [] for k in FIBS}     # (net, month)
    nsig = 0
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
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
                df = ps[sym]
                # recent swing from the 4h history up to t
                hist = df[df.index <= t].tail(SWING_BARS)
                if len(hist) < 4:
                    continue
                sw_hi = float(hist["high"].max()); sw_lo = float(hist["low"].min())
                move = sw_hi - sw_lo
                if move <= 0:
                    continue
                d = HR.load_range(sym, "1m", t + 1, t + (WAIT_H + HOLD_H) * 3600 * 1000)
                if d is None or len(d) < 60:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                nsig += 1
                wait_end = t + WAIT_H * 3600 * 1000
                for name, retr in FIBS.items():
                    if retr is None:
                        # immediate entry at the 4h close; exit over the hold window
                        m = tmin <= t + HOLD_H * 3600 * 1000
                        net = exit_ema60(tmin[m], hi[m], lo[m], cl[m], price, atr)
                        acc[name].append((net, mname))
                    else:
                        target = sw_hi - retr * move
                        # find fill: first 1m within wait window where low <= target
                        fill_i = None
                        for i in range(len(tmin)):
                            if tmin[i] > wait_end:
                                break
                            if lo[i] <= target:
                                fill_i = i; break
                        if fill_i is None:
                            continue   # no fill -> no trade (skipped)
                        ft = tmin[fill_i]
                        m = (tmin >= ft) & (tmin <= ft + HOLD_H * 3600 * 1000)
                        if m.sum() < 60:
                            continue
                        net = exit_ema60(tmin[m], hi[m], lo[m], cl[m], target, atr)
                        acc[name].append((net, mname))
        print(f"  {mname} done ({nsig} signals)", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### FIB-RETRACEMENT ENTRY vs IMMEDIATE ({nsig} signals) #####")
    print(f"{'entry':<11}{'fills':>7}{'fill%':>7}{'avg%':>8}{'WR':>6}{'PF':>7}   per-month return")
    print("-" * 80)
    for name in FIBS:
        rows = acc[name]; n = np.array([x[0] for x in rows])
        if len(n) == 0:
            continue
        wins = n[n > 0]; losses = n[n <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        wr = (n > 0).mean() * 100
        fillp = len(n) / nsig * 100
        pm = {mm: 0.0 for mm in MONTHS}
        for net_i, mm in rows:
            pm[mm] += net_i * 0.125
        pms = " ".join(f"{mm[2:]}:{pm[mm]:+.0f}" for mm in MONTHS)
        print(f"{name:<11}{len(n):>7}{fillp:>6.0f}%{n.mean():>7.2f}%{wr:>5.0f}%{pf:>7.2f}   {pms}", flush=True)
    print("\nDONE_FIB_ENTRY.", flush=True)


if __name__ == "__main__":
    main()
