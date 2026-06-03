#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the strategy's trend SIGNAL on a higher timeframe (daily / weekly).

Question: on a big frame, do we catch only big moves, or does it fail? We
compute the core trend signal (ema_order>=0.75, rsi<75, calm>80pctl, price>EMA200,
MACD>0, ADX>=25) on TF candles, enter at the TF close, and exit when the TF
closes below its EMA20 (trend break) or hits a 2*ATR_TF stop. 8 concurrent,
12.5% each. (Volume band dropped here — its scaling is TF-specific.)

Honest caveat: EMA200 on weekly needs ~4y history -> tiny universe (old coins).
Run:  python experiments/higher_tf_test.py
"""

from __future__ import annotations

import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FEE = 0.2
MAX_CONC = 8
POS_W = 0.125
SL_ATR = 2.0
MIN_HIST = 210

TESTS = {"1d": ("2023-06-01", "2026-06-01"),
         "1w": ("2021-06-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def fetch_tf(sym, interval, start_ms, end_ms):
    rows = []; cur = start_ms
    while cur < end_ms:
        data = PB._http_get("/klines", {"symbol": sym, "interval": interval,
                                        "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cur = int(data[-1][6]) + 1
        time.sleep(0.02)
    if len(rows) < MIN_HIST:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_av", "n", "tb", "tq", "ig"])
    for c in ["open", "high", "low", "close", "volume", "quote_av"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = df["time"].astype("int64"); df["close_time"] = df["close_time"].astype("int64")
    return df.drop_duplicates("time").reset_index(drop=True)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def signal_and_exit(df):
    """Compute core signal + EMA20 exit levels on the TF df."""
    df = S.compute_features(df)
    cl = df["calm_long"].to_numpy()
    thr = pd.Series(cl).shift(1).rolling(S.CALM_LOOKBACK, min_periods=S.CALM_MIN_HISTORY)\
            .quantile(S.CALM_PCTL / 100.0).to_numpy()
    eo = df["ema_order"].to_numpy(); rv = df["rsi"].to_numpy(); ab = df["above_ema200"].to_numpy()
    mp = df["macd_pos"].to_numpy(); adx = df["adx"].to_numpy(); atr = df["atr"].to_numpy()
    sig = ((eo >= 0.75) & (rv < 75) & np.isfinite(cl) & np.isfinite(thr) & (cl > thr) &
           (ab > 0.5) & (mp > 0.5) & np.isfinite(adx) & (adx >= 25) & np.isfinite(atr) & (atr > 0))
    if len(sig) >= MIN_HIST:
        sig[:MIN_HIST - 1] = False
    df["sig"] = sig
    df["ema20"] = ema(df["close"].to_numpy(float), 20)
    return df


def backtest(per, interval, start_ms, end_ms):
    # timeline = sorted union of close_times in window
    times = sorted({int(t) for df in per.values() for t in df["close_time"]
                    if start_ms <= t <= end_ms})
    idx = {sym: {int(ct): i for i, ct in enumerate(df["close_time"])} for sym, df in per.items()}
    open_pos = {}; trades = []
    for t in times:
        # exits
        for sym in list(open_pos.keys()):
            pos = open_pos[sym]; df = per[sym]
            if t not in idx[sym]:
                continue
            i = idx[sym][t]
            if i <= pos["i0"]:
                continue
            c = float(df["close"].iloc[i]); lo = float(df["low"].iloc[i]); e20 = float(df["ema20"].iloc[i])
            xp = None
            if lo <= pos["sl"]:
                xp = pos["sl"]
            elif c < e20:
                xp = c
            if xp is not None:
                trades.append((xp / pos["ep"] - 1) * 100 - FEE)
                del open_pos[sym]
        # entries
        if len(open_pos) >= MAX_CONC:
            continue
        cands = []
        for sym, df in per.items():
            if sym in open_pos or t not in idx[sym]:
                continue
            i = idx[sym][t]
            if bool(df["sig"].iloc[i]):
                cands.append((sym, i, float(df["close"].iloc[i]), float(df["atr"].iloc[i]),
                              float(df["quote_av"].iloc[i])))
        cands.sort(key=lambda x: x[4], reverse=True)
        for sym, i, price, atr, _ in cands[:MAX_CONC - len(open_pos)]:
            open_pos[sym] = dict(i0=i, ep=price, sl=price - SL_ATR * atr)
    return trades


def main():
    syms = PIT.list_all_usdt_symbols()
    print("HIGHER-TIMEFRAME TEST (core trend signal + EMA20-break exit)\n")
    print(f"{'TF':<5}{'coins':>7}{'trades':>8}{'WR':>6}{'PF':>7}{'avgWin':>9}{'ret(eq-wt)':>12}")
    print("-" * 56)
    for interval, (s, e) in TESTS.items():
        start, end = parse(s), parse(e)
        warm = start - (MIN_HIST + 10) * (86400000 if interval == "1d" else 7 * 86400000)
        per = {}
        for i, sym in enumerate(syms):
            df = fetch_tf(sym, interval, warm, end)
            if df is None or len(df) < MIN_HIST:
                continue
            per[sym] = signal_and_exit(df)
            if (i + 1) % 120 == 0:
                print(f"   ...scanned {i+1}/{len(syms)} ({len(per)} usable)", flush=True)
        tr = backtest(per, interval, start, end)
        if tr:
            a = np.array(tr); wins = a[a > 0]; losses = a[a <= 0]
            pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
            ret = a.sum() * POS_W   # equal-weight sum (rough portfolio %)
            aw = wins.mean() if len(wins) else 0
            print(f"{interval:<5}{len(per):>7}{len(a):>8}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}"
                  f"{aw:>8.1f}%{ret:>11.0f}%", flush=True)
        else:
            print(f"{interval:<5}{len(per):>7}{0:>8}  (no trades)", flush=True)
        del per
    print("\nDONE_HIGHER_TF.", flush=True)


if __name__ == "__main__":
    main()
