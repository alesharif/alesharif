#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backtest the video strategy: MACD(12,26,9) + 200 EMA trend-pullback.

Rule (long; short is mirror):
  * close > EMA200                       (trend filter)
  * MACD line crosses ABOVE signal line  (entry trigger)
  * crossover happens BELOW zero         (i.e. on a pullback, not chasing)
  * enter at the close of the cross candle
  * SL = recent swing low (lookback L);  TP = entry + 1.5 * (entry - SL)
Variants:  +2*ATR breathing-room stop;  +skip-flat-EMA200 (regime filter).
We test the video's claimed ~70% win rate, on our USDT universe, 4h candles,
both our dev window (2025-12..2026-06) and 2024 (out-of-sample regime).

One open trade per symbol at a time. Outcomes in R (risk multiples), net of fees.
Run:  python experiments/macd_200ema.py dev    |    ... 2024
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FOUR = 4 * 3600 * 1000
FEE = 0.05          # % per side (futures taker); round-trip ~0.10%
RR = 1.5            # take-profit risk:reward
SWING = 10          # swing-low/high lookback (bars)
ATR_MULT = 2.0
SLOPE_K = 10        # bars for EMA200 slope (flat filter)
SLOPE_MIN = 0.003   # min |slope| over K bars to call the trend non-flat
MAXBARS = 120       # give up after this many bars -> close at market

PERIODS = {"dev": ("2025-12-01", "2026-06-01"),
           "2024": ("2024-01-01", "2025-01-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def atr(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def simulate_symbol(df, t0, t1):
    """Yield trades for one symbol as dicts {dir, R_base, R_atr, ok_slope, t}."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    tm = df["time"].to_numpy()
    n = len(c)
    if n < 220:
        return
    e200 = ema(c, 200)
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    a = atr(h, l, c, 14)
    cup = (macd[:-1] <= sig[:-1]) & (macd[1:] > sig[1:])     # cross up at i (index+1)
    cdn = (macd[:-1] >= sig[:-1]) & (macd[1:] < sig[1:])
    cross_up = np.concatenate([[False], cup])
    cross_dn = np.concatenate([[False], cdn])

    open_until = -1
    for i in range(210, n - 1):
        if tm[i] < t0 or tm[i] > t1:
            continue
        if i <= open_until:
            continue
        long = (c[i] > e200[i]) and cross_up[i] and (macd[i] < 0)
        short = (c[i] < e200[i]) and cross_dn[i] and (macd[i] > 0)
        if not (long or short):
            continue
        slope = (e200[i] - e200[i - SLOPE_K]) / e200[i] if e200[i] > 0 else 0.0
        if long:
            entry = c[i]; swing = l[i - SWING:i + 1].min()
            sl_b = swing; sl_a = swing - ATR_MULT * a[i]
            ok_slope = slope > SLOPE_MIN
        else:
            entry = c[i]; swing = h[i - SWING:i + 1].max()
            sl_b = swing; sl_a = swing + ATR_MULT * a[i]
            ok_slope = slope < -SLOPE_MIN
        risk_b = abs(entry - sl_b); risk_a = abs(entry - sl_a)
        if risk_b <= 0 or risk_a <= 0:
            continue
        tp_b = entry + RR * (entry - sl_b) if long else entry - RR * (sl_b - entry)
        tp_a = entry + RR * (entry - sl_a) if long else entry - RR * (sl_a - entry)

        def outcome(sl, tp, risk):
            for j in range(i + 1, min(i + 1 + MAXBARS, n)):
                hit_sl = (l[j] <= sl) if long else (h[j] >= sl)
                hit_tp = (h[j] >= tp) if long else (l[j] <= tp)
                if hit_sl and hit_tp:
                    return -1.0       # same bar -> assume stop hit first (conservative)
                if hit_sl:
                    return -1.0
                if hit_tp:
                    return RR
            # timeout: close at last close
            px = c[min(i + MAXBARS, n - 1)]
            return ((px - entry) if long else (entry - px)) / risk
        feeR = 2 * FEE / (risk_b / entry * 100)       # round-trip fee in R units
        feeR_a = 2 * FEE / (risk_a / entry * 100)
        Rb = outcome(sl_b, tp_b, risk_b) - feeR
        Ra = outcome(sl_a, tp_a, risk_a) - feeR_a
        open_until = i + 1   # block re-entry next bar; real exit may be later (approx)
        yield {"dir": "L" if long else "S", "Rb": Rb, "Ra": Ra, "ok_slope": ok_slope}


def main():
    pkey = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PERIODS else "dev"
    s, e = PERIODS[pkey]; t0, t1 = parse(s), parse(e)
    ff = t0 - 220 * FOUR
    print(f"MACD+200EMA backtest — period={pkey} ({s}..{e}), 4h candles\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, t1, log=lambda *a: None)
    print(f"  universe: {len(raw)} symbols loaded; simulating...", flush=True)

    trades = []
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        for tr in simulate_symbol(df, t0, t1):
            trades.append(tr)
    del raw; gc.collect()

    if not trades:
        print("no trades."); return
    Rb = np.array([t["Rb"] for t in trades])
    Ra = np.array([t["Ra"] for t in trades])
    slope_ok = np.array([t["ok_slope"] for t in trades])

    def rep(name, R, mask=None):
        r = R if mask is None else R[mask]
        if len(r) == 0:
            print(f"{name:<26}  (no trades)"); return
        wr = (r > 0).mean() * 100
        exp = r.mean()
        tot = r.sum()
        print(f"{name:<26}{len(r):>6}{wr:>7.0f}%{exp:>+8.2f}R{tot:>+9.0f}R")

    print(f"{'variant':<26}{'n':>6}{'WR':>8}{'exp/trade':>9}{'totalR':>10}")
    print("-" * 60)
    rep("base (swing SL)", Rb)
    rep("base + 2xATR SL", Ra)
    rep("base + skip-flat", Rb, slope_ok)
    rep("base+ATR + skip-flat", Ra, slope_ok)
    print(f"\nنقطة التعادل عند 1.5:1 = WR 40%. الفيديو يدّعي ~70%.")
    print(f"exp>0 => توقّع موجب. قارن WR الفعلي بالـ70% المُدّعى.")
    print(f"\nDONE_MACD200.", flush=True)


if __name__ == "__main__":
    main()
