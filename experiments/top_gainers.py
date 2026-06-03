#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TOP-GAINERS MOMENTUM — a genuinely different strategy targeting live pumps.

Instead of the selective 4h trend signal, we scan for coins ALREADY exploding
(top % gainers over the last LB hours) and enter on momentum CONTINUATION, with
a fast exit. This directly targets the big live pumps the user observes.

On 1h candles, for each coin: gain_LB = close/close[LB]-1. A coin is a "gainer"
if gain_LB >= G and it is still rising (close>prev). At each hour we rank gainers
by gain and enter the top (8 concurrent, stable-fear gate, liquidity filter).
Exit: 1h close < EMA20(1h) OR 1h low <= entry-2*ATR1h.

Tests several (LB, G). Months: 2025-12 .. 2026-05. data-api 1h (has today).
Run:  python experiments/top_gainers.py
"""

from __future__ import annotations

import sys, time, socket
socket.setdefaulttimeout(30)
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FEE = 0.2
MAX_CONC = 8
POS_W = 0.125
FEAR = 1.15
SL_ATR = 2.0
FOUR = 4 * 3600 * 1000
START, END = "2025-12-01", "2026-06-01"
MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
# (label, lookback_hours, min_gain%)
VARIANTS = [("LB4_G10", 4, 10.0), ("LB4_G20", 4, 20.0),
            ("LB12_G20", 12, 20.0), ("LB12_G30", 12, 30.0)]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def atr_w(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def fetch_1h(sym, start_ms, end_ms):
    rows = []; cur = start_ms
    while cur < end_ms:
        data = PB._http_get("/klines", {"symbol": sym, "interval": "1h",
                                        "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cur = int(data[-1][6]) + 1
        time.sleep(0.02)
    if len(rows) < 60:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_av", "n", "tb", "tq", "ig"])
    for c in ["high", "low", "close", "quote_av"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close_time"] = df["close_time"].astype("int64")
    return df


def main():
    s, e = parse(START), parse(END)
    warm = s - 20 * 3600 * 1000
    # liquid universe via 4h cache
    ff4 = s - PB.WARMUP_BARS * FOUR
    raw4 = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff4, e, log=lambda *a: None)
    syms = [sym for sym, df in raw4.items()
            if len(df) >= 30 and float(df["quote_av"].tail(180).mean()) * 6 >= 2_000_000]
    del raw4
    sr = PB.build_stable_ratio(ff4, e)
    sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
    print(f"Liquid coins: {len(syms)}. Fetching 1h ...", flush=True)
    per = {}
    for i, sym in enumerate(syms):
        df = fetch_1h(sym, warm, e)
        if df is None:
            continue
        h = df["high"].to_numpy(float); l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
        per[sym] = dict(h=h, l=l, c=c, ct=df["close_time"].to_numpy(),
                        qv=df["quote_av"].to_numpy(float), atr=atr_w(h, l, c, 14),
                        e20=ema(c, 20), qvavg=pd.Series(df["quote_av"].to_numpy(float)).rolling(20).mean().shift(1).to_numpy())
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(syms)} ({len(per)} usable)", flush=True)
    print(f"usable: {len(per)}\n", flush=True)

    # global 1h timeline
    times = sorted({int(t) for d in per.values() for t in d["ct"] if s <= t <= e})
    idx = {sym: {int(t): i for i, t in enumerate(d["ct"])} for sym, d in per.items()}

    print(f"{'variant':<10}{'trades':>8}{'WR':>6}{'PF':>7}{'avgWin':>9}{'ret':>9}   per-month")
    print("-" * 78)
    for lbl, LB, G in VARIANTS:
        openp = {}; trades = []; pm = defaultdict(list)
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
                elif d["c"][i] < d["e20"][i]:
                    xp = d["c"][i]
                if xp is not None:
                    net = (xp / p["ep"] - 1) * 100 - FEE
                    trades.append(net); pm[p["mon"]].append(net); del openp[sym]
            if sblock.get((t // FOUR) * FOUR + FOUR - 1, False) or len(openp) >= MAX_CONC:
                continue
            cands = []
            for sym, d in per.items():
                if sym in openp or t not in idx[sym]:
                    continue
                i = idx[sym][t]
                if i < LB + 1:
                    continue
                gain = (d["c"][i] / d["c"][i - LB] - 1) * 100
                if (gain >= G and d["c"][i] > d["c"][i - 1] and d["atr"][i] > 0 and
                        np.isfinite(d["qvavg"][i]) and d["qvavg"][i] >= 1_000_000):
                    cands.append((sym, i, d["c"][i], d["atr"][i], gain))
            cands.sort(key=lambda x: x[4], reverse=True)   # biggest gainers first
            mon = pd.Timestamp(t, unit="ms", tz="UTC").strftime("%Y-%m")
            for sym, i, price, atr, _ in cands[:MAX_CONC - len(openp)]:
                openp[sym] = dict(i0=i, ep=price, sl=price - SL_ATR * atr, mon=mon)
        if trades:
            a = np.array(trades); wins = a[a > 0]; losses = a[a <= 0]
            pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
            pms = " ".join(f"{m[2:]}:{np.array(pm[m]).sum()*POS_W:+.0f}" if pm[m] else f"{m[2:]}:0" for m in MONTHS)
            print(f"{lbl:<10}{len(a):>8}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}"
                  f"{(wins.mean() if len(wins) else 0):>8.1f}%{a.sum()*POS_W:>8.0f}%   {pms}", flush=True)
        else:
            print(f"{lbl:<10}{0:>8}  no trades", flush=True)
    print("\nDONE_TOPGAINERS.", flush=True)


if __name__ == "__main__":
    main()
