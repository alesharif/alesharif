#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STANDALONE SQUEEZE catcher — scans ALL coins on 15m for the launch moment.

Independent of the 4h signal. The classic TTM squeeze: Bollinger Bands coil
INSIDE the Keltner Channels (low-volatility accumulation), then RELEASE with an
upward break on volume = ignition. We enter on that breakout bar and ride it
with the best exit found (1h close below EMA20 + 2*ATR stop).

Signal (15m, causal):
  BB(20,2), KC(20, 1.5*ATR15). squeeze_on = BB inside KC.
  fire = was squeezed in the last 6 bars  AND  not squeezed now (released)
         AND close > MA20  AND  close > prev close   AND  volume > 1.5*avg20
  liquidity: 20-bar avg quote-volume >= MIN_QV
Portfolio: 8 concurrent, 12.5% each, stable-ratio fear gate 1.15.
Exit: EMA60_20 (1h close < EMA20) + 2*ATR15 hard stop, on the real 1m path.

Run:  python experiments/squeeze_catcher.py
"""

from __future__ import annotations

import gc, sys, socket
socket.setdefaulttimeout(30)        # safety net against hung requests
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
MAX_CONC = 8
POS_USD = 250.0
HOLD_H = 48
HARD_SL_ATR = 2.0
FEAR = 1.15
MIN_QV = 1_000_000          # 20-bar avg quote volume (liquidity floor)
SQZ_LOOKBACK = 6
VOL_MULT = 1.5
FIFTEEN = 15 * 60 * 1000

MONTHS = {"2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def atr_w(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def resample_close(tmin, cl, tf_min):
    n = len(cl); grp = np.arange(n) // tf_min
    C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            C.append(cl[m][-1]); T.append(tmin[m][-1])
    return np.array(C), np.array(T)


def exit_ema60(tmin, hi, lo, cl, ep, atr):
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


def fire_events(sym, df15, start, end):
    """Return list of (time, price, atr15, qv) squeeze-breakout fires in [start,end]."""
    c = df15["close"].to_numpy(float); h = df15["high"].to_numpy(float)
    l = df15["low"].to_numpy(float); v = df15["volume"].to_numpy(float)
    qv = df15["quote_av"].to_numpy(float) if "quote_av" in df15 else v * c
    ct = df15["close_time"].to_numpy()
    n = len(c)
    if n < 60:
        return []
    cs = pd.Series(c)
    ma = cs.rolling(20).mean().to_numpy(); sd = cs.rolling(20).std().to_numpy()
    bb_up = ma + 2 * sd; bb_lo = ma - 2 * sd
    atr = atr_w(h, l, c, 14)
    kc = ema(c, 20); kc_up = kc + 1.5 * atr; kc_lo = kc - 1.5 * atr
    in_sqz = (bb_up < kc_up) & (bb_lo > kc_lo)
    sqz_recent = pd.Series(in_sqz.astype(float)).rolling(SQZ_LOOKBACK).max().shift(1).to_numpy()
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    qvavg = pd.Series(qv).rolling(20).mean().shift(1).to_numpy()
    out = []
    for i in range(40, n):
        if not (start <= ct[i] <= end):
            continue
        if (sqz_recent[i] >= 1 and not in_sqz[i] and c[i] > ma[i] and
                c[i] > c[i - 1] and v[i] > VOL_MULT * vavg[i] and
                np.isfinite(qvavg[i]) and qvavg[i] >= MIN_QV and atr[i] > 0):
            out.append((int(ct[i]), float(c[i]), float(atr[i]), float(qvavg[i])))
    return out


def main():
    print("STANDALONE SQUEEZE CATCHER (15m breakout) + EMA60_20 exit\n")
    print(f"{'month':<10}{'fires':>7}{'trades':>8}{'WR':>6}{'PF':>7}{'avg%':>8}{'return':>9}")
    print("-" * 55)
    allnets = []
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        warm = start - 3 * 24 * 3600 * 1000
        # pre-filter to LIQUID coins using cached 4h data (avoids scanning 592 on 15m)
        ff4 = start - PB.WARMUP_BARS * 4 * 3600 * 1000
        raw4 = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff4, end, log=lambda *a: None)
        syms = [sym for sym, df in raw4.items()
                if len(df) >= 30 and (float(df["quote_av"].tail(180).mean()) * 6) >= 2_000_000]
        del raw4; gc.collect()
        print(f"  ({mname}: {len(syms)} liquid coins to scan on 15m)", flush=True)
        sr = PB.build_stable_ratio(start - 40 * 4 * 3600 * 1000, end)
        sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
        # collect all fires across coins
        fires = []
        for sym in syms:
            d15 = HR.load_range(sym, "15m", warm, end)
            if d15 is None or len(d15) < 60:
                continue
            for ev in fire_events(sym, d15, start, end):
                fires.append((ev[0], sym, ev[1], ev[2], ev[3]))   # (t, sym, price, atr, qv)
        fires.sort()
        # portfolio sim: 8 concurrent, stable gate (nearest 4h bar), rank by qv at each ts
        open_until = {}
        nets = []
        # group fires by timestamp
        from collections import defaultdict
        byt = defaultdict(list)
        for t, sym, price, atr, qv in fires:
            byt[t].append((sym, price, atr, qv))
        for t in sorted(byt):
            open_until = {sy: u for sy, u in open_until.items() if u > t}
            if len(open_until) >= MAX_CONC:
                continue
            # stable fear gate: nearest 4h close <= t
            fourh = (t // (4 * 3600 * 1000)) * (4 * 3600 * 1000) + 4 * 3600 * 1000 - 1
            if sblock.get(fourh, False):
                continue
            cands = [x for x in byt[t] if x[0] not in open_until]
            cands.sort(key=lambda x: x[3], reverse=True)
            for sym, price, atr, qv in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 60:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                net = exit_ema60(tmin, hi, lo, cl, price, atr)
                nets.append(net)
        st_n = len(nets)
        if st_n:
            a = np.array(nets); wins = a[a > 0]; losses = a[a <= 0]
            pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
            wr = (a > 0).mean() * 100
            ret = a.sum() * 0.125
        else:
            pf = wr = ret = 0; a = np.array([])
        allnets.append((mname, nets))
        print(f"{mname:<10}{len(fires):>7}{st_n:>8}{wr:>5.0f}%{pf:>7.2f}"
              f"{(a.mean() if st_n else 0):>7.2f}%{ret:>8.1f}%", flush=True)
        gc.collect()

    print("-" * 55)
    flat = [x for _, ns in allnets for x in ns]
    if flat:
        a = np.array(flat); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        tot = a.sum() * 0.125
        print(f"{'TOTAL':<10}{'':>7}{len(flat):>8}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}"
              f"{a.mean():>7.2f}%{tot:>8.1f}%   (~{tot/4:.1f}%/mo)")
    print("\nDONE_SQUEEZE.", flush=True)


if __name__ == "__main__":
    main()
