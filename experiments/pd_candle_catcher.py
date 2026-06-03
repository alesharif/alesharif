#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STANDALONE Pump_Doctor catcher — directional-volume ignition (15m).

Port of the user's "Pump_Doctor / 1337_Volume_Trend" Pine indicator as an entry
catcher. Over the last 14 x 15m bars we split volume into UP (close>open) and
DOWN volume:
  avg_up = sum(up_volume)/count(up_bars)
  flow   = sum(up_volume) - sum(down_volume)        (net buying pressure)
  sigup  = current volume >= avg_up  AND  close>open
  FIRE   = sigup AND flow>0           (ignition with net accumulation)
Scan liquid coins, enter on the fire bar, ride with EMA60_20 + 2*ATR stop.
Same harness as squeeze_catcher.py so results are directly comparable.

Run:  python experiments/pump_doctor_catcher.py
"""

from __future__ import annotations

import gc, sys, socket
socket.setdefaulttimeout(30)
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
MAX_CONC = 8
HOLD_H = 48
HARD_SL_ATR = 2.0
FEAR = 1.15
MIN_QV = 1_000_000
LOOKBACK = 14
FOUR = 4 * 3600 * 1000

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
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
    """Large-candle ignition: range >= THRESH*ATR AND green (close>close[1]), liquid."""
    o = df15["open"].to_numpy(float); c = df15["close"].to_numpy(float)
    h = df15["high"].to_numpy(float); l = df15["low"].to_numpy(float)
    v = df15["volume"].to_numpy(float); ct = df15["close_time"].to_numpy()
    qv = v * c
    n = len(c)
    if n < 60:
        return []
    atr = atr_w(h, l, c, 14)
    THRESH = 1.9
    candle = np.abs(h - l)
    large = candle >= THRESH * atr
    qvavg = pd.Series(qv).rolling(20).mean().shift(1).to_numpy()
    out = []
    for i in range(40, n):
        if not (start <= ct[i] <= end):
            continue
        if (large[i] and c[i] > c[i - 1] and
                np.isfinite(qvavg[i]) and qvavg[i] >= MIN_QV and atr[i] > 0):
            out.append((int(ct[i]), float(c[i]), float(atr[i]), float(qvavg[i])))
    return out


def main():
    print("STANDALONE Pump&Dump-Candle CATCHER (15m large-candle ignition) + EMA60_20 exit\n")
    print(f"{'month':<10}{'fires':>7}{'trades':>8}{'WR':>6}{'PF':>7}{'avg%':>8}{'return':>9}")
    print("-" * 55)
    allnets = []
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        warm = start - 3 * 24 * 3600 * 1000
        ff4 = start - PB.WARMUP_BARS * FOUR
        raw4 = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff4, end, log=lambda *a: None)
        syms = [sym for sym, df in raw4.items()
                if len(df) >= 30 and (float(df["quote_av"].tail(180).mean()) * 6) >= 2_000_000]
        del raw4; gc.collect()
        print(f"  ({mname}: {len(syms)} liquid coins)", flush=True)
        sr = PB.build_stable_ratio(start - 40 * FOUR, end)
        sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
        fires = []
        for sym in syms:
            d15 = HR.load_range(sym, "15m", warm, end)
            if d15 is None or len(d15) < 60:
                continue
            for ev in fire_events(sym, d15, start, end):
                fires.append((ev[0], sym, ev[1], ev[2], ev[3]))
        fires.sort()
        from collections import defaultdict
        byt = defaultdict(list)
        for t, sym, price, atr, qv in fires:
            byt[t].append((sym, price, atr, qv))
        open_until = {}; nets = []
        for t in sorted(byt):
            open_until = {sy: u for sy, u in open_until.items() if u > t}
            if len(open_until) >= MAX_CONC:
                continue
            fourh = (t // FOUR) * FOUR + FOUR - 1
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
                nets.append(exit_ema60(tmin, hi, lo, cl, price, atr))
        a = np.array(nets) if nets else np.array([])
        if len(a):
            wins = a[a > 0]; losses = a[a <= 0]
            pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
            print(f"{mname:<10}{len(fires):>7}{len(a):>8}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}"
                  f"{a.mean():>7.2f}%{a.sum()*0.125:>8.1f}%", flush=True)
        else:
            print(f"{mname:<10}{len(fires):>7}{0:>8}{0:>5}%{0:>7}{0:>7}%{0:>8}%", flush=True)
        allnets.append((mname, nets)); gc.collect()

    print("-" * 55)
    flat = [x for _, ns in allnets for x in ns]
    if flat:
        a = np.array(flat); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        tot = a.sum() * 0.125
        print(f"{'TOTAL':<10}{'':>7}{len(flat):>8}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}"
              f"{a.mean():>7.2f}%{tot:>8.1f}%   (~{tot/len(MONTHS):.1f}%/mo)")
    print("\nDONE_PD_CANDLE.", flush=True)


if __name__ == "__main__":
    main()
