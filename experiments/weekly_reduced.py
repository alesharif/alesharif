#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weekly strategy with a REDUCED trend EMA (not EMA200) — user's request.

EMA200 on weekly needs ~4y history (excludes coins). Test the trend filter with
shorter EMAs (30/50/100 weeks) so more coins qualify, with scaled-down lookbacks.
Daily showed big avg wins (+16.6%) but big losses; so we ALSO test a faster exit
to cut the give-back: weekly close<EMA10 with a tighter 1.5*ATR stop.

Signal (weekly, causal): ema_order(5,10,20,50)>=0.75, rsi<75, calm>80pctl(50,min20),
close>EMA_N, MACD>0, ADX>=25. Entry at weekly close; exit weekly close<EMA_exit
or hard SL*ATR. Run:  python experiments/weekly_reduced.py
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
START, END = "2022-01-01", "2026-06-01"
# variants: (trend_EMA_N, exit_EMA, SL_ATR)
VARIANTS = [(30, 20, 2.0), (50, 20, 2.0), (100, 20, 2.0),
            (50, 10, 1.5), (50, 20, 1.5)]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def fetch_w(sym, start_ms, end_ms):
    rows = []; cur = start_ms
    while cur < end_ms:
        data = PB._http_get("/klines", {"symbol": sym, "interval": "1w",
                                        "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cur = int(data[-1][6]) + 1
        time.sleep(0.02)
    if len(rows) < 35:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_av", "n", "tb", "tq", "ig"])
    for c in ["open", "high", "low", "close", "volume", "quote_av"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close_time"] = df["close_time"].astype("int64")
    return df.reset_index(drop=True)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def build_signal(df, N, exit_ema):
    c = df["close"].to_numpy(float); h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    emas = {s: ema(c, s) for s in (5, 10, 20, 50)}
    order = sum((emas[a] > emas[b]).astype(float) for a, b in [(5, 10), (10, 20), (20, 50)]) / 3.0
    rsi = S.rsi_series(c, 14); atr = S.atr_series(h, l, c, 14); adx = S.compute_adx(h, l, c, 14)
    macd = ema(c, 12) - ema(c, 26)
    ret = np.concatenate([[np.nan], np.diff(c) / np.where(c[:-1] > 0, c[:-1], np.nan)])
    calm = pd.Series(np.abs(ret)).rolling(20).mean().shift(1).to_numpy() * 100
    thr = pd.Series(calm).shift(1).rolling(50, min_periods=20).quantile(0.80).to_numpy()
    above = c > ema(c, N)
    sig = ((order >= 0.75) & (rsi < 75) & np.isfinite(calm) & np.isfinite(thr) & (calm > thr) &
           above & (macd > 0) & np.isfinite(adx) & (adx >= 25) & (atr > 0))
    mh = max(N, 50) + 5
    if len(sig) >= mh:
        sig[:mh - 1] = False
    return sig, c, h, l, atr, ema(c, exit_ema), df["close_time"].to_numpy(), df["quote_av"].to_numpy(float)


def backtest(per, sl_atr, start_ms, end_ms):
    times = sorted({int(t) for d in per.values() for t in d["ct"] if start_ms <= t <= end_ms})
    idx = {s: {int(t): i for i, t in enumerate(d["ct"])} for s, d in per.items()}
    openp = {}; trades = []
    for t in times:
        for sym in list(openp.keys()):
            d = per[sym]; p = openp[sym]
            if t not in idx[sym]:
                continue
            i = idx[sym][t]
            if i <= p["i0"]:
                continue
            xp = None
            if d["l"][i] <= p["sl"]:
                xp = p["sl"]
            elif d["c"][i] < d["ee"][i]:
                xp = d["c"][i]
            if xp is not None:
                trades.append((xp / p["ep"] - 1) * 100 - FEE); del openp[sym]
        if len(openp) >= MAX_CONC:
            continue
        cands = []
        for sym, d in per.items():
            if sym in openp or t not in idx[sym]:
                continue
            i = idx[sym][t]
            if d["sig"][i]:
                cands.append((sym, i, d["c"][i], d["atr"][i], d["qv"][i]))
        cands.sort(key=lambda x: x[4], reverse=True)
        for sym, i, price, atr, _ in cands[:MAX_CONC - len(openp)]:
            openp[sym] = dict(i0=i, ep=price, sl=price - sl_atr * atr)
    return trades


def main():
    syms = PIT.list_all_usdt_symbols()
    start, end = parse(START), parse(END)
    warm = start - 120 * 7 * 86400000
    # fetch weekly once per coin (reused across variants)
    rawcache = {}
    print(f"Fetching weekly for {len(syms)} coins ...", flush=True)
    for i, sym in enumerate(syms):
        df = fetch_w(sym, warm, end)
        if df is not None and len(df) >= 35:
            rawcache[sym] = df
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(syms)} ({len(rawcache)} usable)", flush=True)
    print(f"usable weekly coins: {len(rawcache)}\n", flush=True)

    print(f"{'EMA_N':<6}{'exitEMA':>8}{'SL':>5}{'coins':>7}{'trades':>8}{'WR':>6}{'PF':>7}{'avgWin':>9}{'ret':>9}")
    print("-" * 65)
    for N, ee, sl in VARIANTS:
        per = {}
        for sym, df in rawcache.items():
            sig, c, h, l, atr, eema, ct, qv = build_signal(df, N, ee)
            if sig.sum() == 0:
                continue
            per[sym] = dict(sig=sig, c=c, h=h, l=l, atr=atr, ee=eema, ct=ct, qv=qv)
        tr = backtest(per, sl, start, end)
        if tr:
            a = np.array(tr); wins = a[a > 0]; losses = a[a <= 0]
            pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
            print(f"{N:<6}{ee:>8}{sl:>5.1f}{len(per):>7}{len(a):>8}{(a>0).mean()*100:>5.0f}%"
                  f"{pf:>7.2f}{(wins.mean() if len(wins) else 0):>8.1f}%{a.sum()*POS_W:>8.0f}%", flush=True)
        else:
            print(f"{N:<6}{ee:>8}{sl:>5.1f}{len(per):>7}{0:>8}  no trades", flush=True)
    print("\nDONE_WEEKLY_RED.", flush=True)


if __name__ == "__main__":
    main()
