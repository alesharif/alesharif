#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Indicator-based exits (user's idea): exit when price closes below an EMA, or
on a fast/slow EMA crossover — vs the wide 2*ATR trailing stop.

Entry set: 4h entry_signal + stable-ratio fear gate (1.15) + per-coin quality
filter (atr_ratio>=5% & adx>=40)  [the best base so far]. Exit choices, each
with a protective hard stop (2*ATR), tested on the real 1m path (48h):

  TRAIL_2ATR : benchmark — wide 2*ATR trailing stop, no indicator
  EMA15_20   : exit when a 15m candle CLOSES below EMA20(15m)
  EMA15_50   : ... below EMA50(15m)
  EMA60_10   : exit when a 1h candle CLOSES below EMA10(1h)
  EMA60_20   : ... below EMA20(1h)
  XEMA15     : exit on EMA9<EMA21 crossover (15m)

Realistic fills: indicator exit at the candle close we observe; hard stop at the
level. Run:  python experiments/indicator_exit.py
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
HARD_SL_ATR = 2.0
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def resample(tmin, hi, lo, cl, tf_min):
    """Resample 1m arrays to tf_min-minute candles. Returns (close, close_time)."""
    n = len(cl)
    grp = (np.arange(n) // tf_min)
    closes = []; cts = []
    for g in range(grp[-1] + 1):
        m = grp == g
        if not m.any():
            continue
        closes.append(cl[m][-1]); cts.append(tmin[m][-1])
    return np.array(closes), np.array(cts)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def exit_trail(hi, lo, cl, ep, atr):
    sl = ep - HARD_SL_ATR * atr; peak = ep; trailing = False
    for i in range(len(cl)):
        if lo[i] <= sl:
            return min(sl, cl[i]) / ep - 1
        if hi[i] > peak:
            peak = hi[i]
            if peak - ep >= 1.0 * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - 2.0 * atr)
    return cl[-1] / ep - 1


def exit_ema(tmin, hi, lo, cl, ep, atr, tf_min, span, span2=None):
    """Hard stop (2ATR) on 1m + exit when a tf candle closes below EMA(span)
    (or EMA(span) < EMA(span2) crossover if span2 given). Fill at that close."""
    sl = ep - HARD_SL_ATR * atr
    # find hard-stop time
    sl_idx = None
    for i in range(len(cl)):
        if lo[i] <= sl:
            sl_idx = i; break
    closes, cts = resample(tmin, hi, lo, cl, tf_min)
    if len(closes) < (span2 or span) + 1:
        # not enough tf candles; fall back to stop or end
        if sl_idx is not None:
            return min(sl, cl[sl_idx]) / ep - 1
        return cl[-1] / ep - 1
    e1 = ema(closes, span)
    if span2:
        e2 = ema(closes, span2)
        sig = e1 < e2
    else:
        sig = closes < e1
    ema_ct = None; ema_px = None
    warm = (span2 or span)
    for j in range(warm, len(closes)):
        if sig[j]:
            ema_ct = cts[j]; ema_px = closes[j]; break
    sl_ct = tmin[sl_idx] if sl_idx is not None else None
    # whichever triggers first in time
    if sl_ct is not None and (ema_ct is None or sl_ct <= ema_ct):
        return min(sl, cl[sl_idx]) / ep - 1
    if ema_ct is not None:
        return ema_px / ep - 1
    return cl[-1] / ep - 1


CONFIGS = ["TRAIL_2ATR", "EMA15_20", "EMA15_50", "EMA60_10", "EMA60_20", "XEMA15"]


def run_exit(name, tmin, hi, lo, cl, ep, atr):
    if name == "TRAIL_2ATR":
        return exit_trail(hi, lo, cl, ep, atr)
    if name == "EMA15_20":
        return exit_ema(tmin, hi, lo, cl, ep, atr, 15, 20)
    if name == "EMA15_50":
        return exit_ema(tmin, hi, lo, cl, ep, atr, 15, 50)
    if name == "EMA60_10":
        return exit_ema(tmin, hi, lo, cl, ep, atr, 60, 10)
    if name == "EMA60_20":
        return exit_ema(tmin, hi, lo, cl, ep, atr, 60, 20)
    if name == "XEMA15":
        return exit_ema(tmin, hi, lo, cl, ep, atr, 15, 9, span2=21)
    return cl[-1] / ep - 1


def main():
    acc = {c: [] for c in CONFIGS}
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
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 30:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                for c in CONFIGS:
                    r = run_exit(c, tmin, hi, lo, cl, price, atr)
                    acc[c].append((r * 100 - FEE, mname))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### INDICATOR EXITS (fear<={FEAR}, atr>={ATR_MIN:.0%}+adx>={ADX_MIN:.0f}) #####")
    print(f"{'exit':<12}{'trades':>7}{'avg%':>8}{'WR':>6}{'PF':>7}   per-month return ($2000,12.5%)")
    print("-" * 80)
    for c in CONFIGS:
        rows = acc[c]; n = np.array([x[0] for x in rows])
        if len(n) == 0:
            continue
        wins = n[n > 0]; losses = n[n <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        wr = (n > 0).mean() * 100
        pm = {mm: 0.0 for mm in MONTHS}
        for net_i, mm in rows:
            pm[mm] += net_i * 0.125
        pms = " ".join(f"{mm[2:]}:{pm[mm]:+.0f}" for mm in MONTHS)
        print(f"{c:<12}{len(n):>7}{n.mean():>7.2f}%{wr:>5.0f}%{pf:>7.2f}   {pms}", flush=True)
    print("\nDONE_INDICATOR_EXIT.", flush=True)


if __name__ == "__main__":
    main()
