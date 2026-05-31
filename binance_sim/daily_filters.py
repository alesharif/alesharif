#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily-timeframe trend-filter candidates, computed from 4h candles.

Resamples each symbol's 4h series to true daily OHLC, computes daily EMAs /
MACD / Stochastic, derives a set of candidate boolean filters, and maps each
back to the 4h bars using only the *previous completed* daily value (shift by
one day) so there is zero look-ahead.

Each ``d_*`` column is 1.0 when that daily condition holds, else 0.0. The
backtester can then gate entries on any one of them via --daily-filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_MS = 24 * 60 * 60 * 1000

# the candidate filters this module produces (name -> short description)
CANDIDATES = {
    "d_above_ema20":  "price > daily EMA20",
    "d_above_ema50":  "price > daily EMA50",
    "d_above_ema100": "price > daily EMA100",
    "d_above_ema200": "price > daily EMA200",
    "d_ema5_9":       "daily EMA5 > EMA9",
    "d_ema9_15":      "daily EMA9 > EMA15",
    "d_ema15_20":     "daily EMA15 > EMA20",
    "d_ema20_50":     "daily EMA20 > EMA50",
    "d_ema50_100":    "daily EMA50 > EMA100",
    "d_ema50_200":    "daily EMA50 > EMA200 (golden)",
    "d_ema100_200":   "daily EMA100 > EMA200",
    "d_ema_stack":    "daily 5>9>15>20>50>100>200 fully stacked",
    "d_macd_up":      "daily MACD line curling up (slope>0, any side)",
    "d_stoch_up":     "daily Stochastic %K curling up (slope>0, any side)",
}


def add_daily_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all ``d_*`` daily-filter columns to a 4h dataframe.

    Expects columns: high, low, close and a 'close_time' column (ms).
    """
    df = df.copy()
    ct = df["close_time"].to_numpy()
    day = (ct // DAY_MS).astype(np.int64)

    d = pd.DataFrame({"day": day, "high": df["high"].to_numpy(float),
                      "low": df["low"].to_numpy(float), "close": df["close"].to_numpy(float)})
    daily = d.groupby("day").agg(high=("high", "max"), low=("low", "min"),
                                 close=("close", "last"))
    dc = daily["close"]

    spans = [5, 9, 15, 20, 50, 100, 200]
    e = {n: dc.ewm(span=n, adjust=False).mean() for n in spans}
    macd = dc.ewm(span=12, adjust=False).mean() - dc.ewm(span=26, adjust=False).mean()
    ll = daily["low"].rolling(14).min()
    hh = daily["high"].rolling(14).max()
    rng = (hh - ll).replace(0, np.nan)
    stoch_k = 100.0 * (dc - ll) / rng

    cond = pd.DataFrame(index=daily.index)
    cond["d_above_ema20"] = (dc > e[20]).astype(float)
    cond["d_above_ema50"] = (dc > e[50]).astype(float)
    cond["d_above_ema100"] = (dc > e[100]).astype(float)
    cond["d_above_ema200"] = (dc > e[200]).astype(float)
    cond["d_ema5_9"] = (e[5] > e[9]).astype(float)
    cond["d_ema9_15"] = (e[9] > e[15]).astype(float)
    cond["d_ema15_20"] = (e[15] > e[20]).astype(float)
    cond["d_ema20_50"] = (e[20] > e[50]).astype(float)
    cond["d_ema50_100"] = (e[50] > e[100]).astype(float)
    cond["d_ema50_200"] = (e[50] > e[200]).astype(float)
    cond["d_ema100_200"] = (e[100] > e[200]).astype(float)
    cond["d_ema_stack"] = ((e[5] > e[9]) & (e[9] > e[15]) & (e[15] > e[20]) &
                           (e[20] > e[50]) & (e[50] > e[100]) & (e[100] > e[200])).astype(float)
    cond["d_macd_up"] = (macd > macd.shift(1)).astype(float)
    cond["d_stoch_up"] = (stoch_k > stoch_k.shift(1)).astype(float)

    # use the PREVIOUS completed day's value (no look-ahead), then map to 4h bars
    cond = cond.shift(1)
    mapped = cond.reindex(day)
    for c in CANDIDATES:
        vals = mapped[c].to_numpy()
        df[c] = np.where(np.isfinite(vals), vals, 0.0)
    return df
