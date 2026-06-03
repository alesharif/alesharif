#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CATCHER SHOOTOUT — 4 ignition entries × FAST 15m exit, one pass, no 1m.

Scans liquid coins on 15m, computes FOUR entry signals per coin, and exits all
of them with a FAST trend exit on the same 15m data (close < EMA20(15m) + a
2*ATR15 hard stop). No 1m loading -> fast. This tests the entries with an exit
that MATCHES the 15m timeframe (the slow 1h-EMA exit may have been the problem).

Catchers:
  SQUEEZE   : TTM squeeze (BB inside KC) release up on volume
  PUMPDOC   : directional-volume ignition (vol>=avg_up & green & flow>0)
  BIGCANDLE : range >= 1.9*ATR & green (large-candle ignition)
  VROC      : normalized volume rate-of-change > 40

Same harness: liquid prefilter, 8 concurrent, stable fear gate 1.15.
Run:  python experiments/catcher_shootout.py
"""

from __future__ import annotations

import gc, sys, socket
socket.setdefaulttimeout(25)
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
MAX_CONC = 8
HOLD_BARS = 192            # 48h on 15m (exit forced if still open)
HARD_SL_ATR = 2.0
EXIT_EMA = 20
FEAR = 1.15
MIN_QV = 1_000_000
FOUR = 4 * 3600 * 1000

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
CATCHERS = ["SQUEEZE", "PUMPDOC", "BIGCANDLE", "VROC"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def atr_w(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def signals(o, h, l, c, v, qv):
    """Return dict of boolean fire arrays for the 4 catchers + atr + ema_exit."""
    n = len(c); cs = pd.Series(c)
    atr = atr_w(h, l, c, 14)
    ema_exit = ema(c, EXIT_EMA)
    qvavg = pd.Series(qv).rolling(20).mean().shift(1).to_numpy()
    liq = np.isfinite(qvavg) & (qvavg >= MIN_QV) & (atr > 0)
    green = c > o
    up1 = np.concatenate([[False], c[1:] > c[:-1]])
    # SQUEEZE
    ma = cs.rolling(20).mean().to_numpy(); sd = cs.rolling(20).std().to_numpy()
    bb_up = ma + 2 * sd; bb_lo = ma - 2 * sd
    kc = ema(c, 20); in_sqz = (bb_up < kc + 1.5 * atr) & (bb_lo > kc - 1.5 * atr)
    sqz_recent = pd.Series(in_sqz.astype(float)).rolling(6).max().shift(1).to_numpy()
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    sig_sqz = (sqz_recent >= 1) & (~in_sqz) & (c > ma) & up1 & (v > 1.5 * vavg) & liq
    # PUMPDOC
    upv = np.where(green, v, 0.0); dnv = np.where(~green, v, 0.0)
    sum_up = pd.Series(upv).rolling(14).sum().to_numpy()
    sum_dn = pd.Series(dnv).rolling(14).sum().to_numpy()
    cnt_up = pd.Series(green.astype(float)).rolling(14).sum().to_numpy()
    avg_up = sum_up / np.where(cnt_up > 0, cnt_up, np.nan)
    flow = sum_up - sum_dn
    sig_pd = green & np.isfinite(avg_up) & (v >= avg_up) & (flow > 0) & liq
    # BIGCANDLE
    sig_bc = (np.abs(h - l) >= 1.9 * atr) & up1 & liq
    # VROC
    mav = pd.Series(v).rolling(96).mean().to_numpy()
    mav1 = np.concatenate([[np.nan], mav[:-1]]); diff = mav - mav1
    with np.errstate(invalid="ignore", divide="ignore"):
        vroc = np.where((diff > 0) & up1 & (mav1 > 0), diff * 100.0 / mav1, 0.0)
    vroc = np.nan_to_num(vroc)
    hmax = np.maximum.accumulate(np.maximum(vroc, 10.0))
    sig_vroc = (vroc / hmax * 100.0 > 40.0) & liq
    return dict(SQUEEZE=np.nan_to_num(sig_sqz).astype(bool),
                PUMPDOC=np.nan_to_num(sig_pd).astype(bool),
                BIGCANDLE=np.nan_to_num(sig_bc).astype(bool),
                VROC=np.nan_to_num(sig_vroc).astype(bool)), atr, ema_exit


def exit_15m(c, h, l, ema_exit, i, atr):
    """Fast 15m exit from bar i: 2ATR stop + close<EMA20(15m). Returns (exit_ct_idx, net%)."""
    ep = c[i]; sl = ep - HARD_SL_ATR * atr[i]
    end = min(len(c), i + 1 + HOLD_BARS)
    for j in range(i + 1, end):
        if l[j] <= sl:
            return j, (min(sl, c[j]) / ep - 1) * 100 - FEE
        if c[j] < ema_exit[j]:
            return j, (c[j] / ep - 1) * 100 - FEE
    return end - 1, (c[end - 1] / ep - 1) * 100 - FEE


def main():
    print("CATCHER SHOOTOUT — 4 entries x fast 15m exit (close<EMA20 + 2ATR)\n")
    results = {cat: defaultdict(list) for cat in CATCHERS}   # cat -> month -> [net]
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        warm = start - 3 * 24 * 3600 * 1000
        ff4 = start - PB.WARMUP_BARS * FOUR
        raw4 = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff4, end, log=lambda *a: None)
        syms = [sym for sym, df in raw4.items()
                if len(df) >= 30 and (float(df["quote_av"].tail(180).mean()) * 6) >= 2_000_000]
        del raw4; gc.collect()
        sr = PB.build_stable_ratio(start - 40 * FOUR, end)
        sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
        # collect fires (with precomputed exit) per catcher
        fires = {cat: [] for cat in CATCHERS}    # (ct, sym, net, exit_ct)
        for sym in syms:
            d = HR.load_range(sym, "15m", warm, end)
            if d is None or len(d) < 130:
                continue
            o = d["open"].to_numpy(float); h = d["high"].to_numpy(float)
            l = d["low"].to_numpy(float); c = d["close"].to_numpy(float)
            v = d["volume"].to_numpy(float); ct = d["close_time"].to_numpy()
            qv = v * c
            sig, atr, ema_exit = signals(o, h, l, c, v, qv)
            for cat in CATCHERS:
                idxs = np.where(sig[cat])[0]
                for i in idxs:
                    if not (start <= ct[i] <= end) or i < 100:
                        continue
                    xj, net = exit_15m(c, h, l, ema_exit, i, atr)
                    fires[cat].append((int(ct[i]), sym, net, int(ct[xj])))
        # portfolio sim per catcher (8 concurrent, stable gate)
        for cat in CATCHERS:
            evs = sorted(fires[cat]); byt = defaultdict(list)
            for ctf, sym, net, xct in evs:
                byt[ctf].append((sym, net, xct))
            open_until = {}
            for t in sorted(byt):
                open_until = {sy: u for sy, u in open_until.items() if u > t}
                if len(open_until) >= MAX_CONC:
                    continue
                fourh = (t // FOUR) * FOUR + FOUR - 1
                if sblock.get(fourh, False):
                    continue
                for sym, net, xct in byt[t]:
                    if sym in open_until or len(open_until) >= MAX_CONC:
                        continue
                    open_until[sym] = xct
                    results[cat][mname].append(net)
        print(f"  {mname} done ({len(syms)} coins)", flush=True)
        gc.collect()

    print(f"\n{'catcher':<11}{'trades':>7}{'WR':>6}{'PF':>7}{'avg%':>8}{'ret_ALL':>9}   per-month")
    print("-" * 78)
    for cat in CATCHERS:
        alln = [x for m in MONTHS for x in results[cat][m]]
        if not alln:
            print(f"{cat:<11}   (no trades)"); continue
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        pm = " ".join(f"{m[2:]}:{np.array(results[cat][m]).sum()*0.125:+.0f}" if results[cat][m]
                      else f"{m[2:]}:0" for m in MONTHS)
        print(f"{cat:<11}{len(a):>7}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{a.mean():>7.2f}%"
              f"{a.sum()*0.125:>8.1f}%   {pm}", flush=True)
    print("\nDONE_SHOOTOUT.", flush=True)


if __name__ == "__main__":
    main()
