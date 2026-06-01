#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare entry signals: original vs EARLIER variants, on April 2025.

Diagnosis showed the original signal enters LATE (after +100..300%) and misses
35% of big movers. This tests alternative entry triggers that aim to catch the
START of a move, measuring for each:
  * how many of the month's >=30% movers it caught (coverage)
  * median "how far up we bought" vs the move start (lateness; lower = better)
  * median forward run still available after entry (opportunity captured)

Signals compared (all long-only, 4h bars):
  ORIG    : the full original signal (ema_order>=.75, rsi<75, adx>=25, >ema200,
            macd>0, calm>thr, vol band)
  BREAKOUT: close breaks above the highest high of the last N bars + volume
            surge (catches the start of a breakout, few lagging filters)
  EMA_CROSS: fast EMA crosses above mid EMA while price>ema50 (early trend flip)
  RELAXED : original but drop ADX + calm (the two laggiest gates)

No exits here — pure entry-quality diagnosis. Fast (4h only).
Run:  python experiments/entry_signals_compare.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FOUR_H = 4 * 3600 * 1000
HOLD = 12          # forward horizon (~2 days) to measure available run
BIG = 30.0         # "big mover" threshold (%)


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def signals_for(df, kind):
    """Return a boolean array (per bar) where this entry kind fires."""
    c = df["close"].to_numpy(float); h = df["high"].to_numpy(float)
    v = df["quote_av"].to_numpy(float)
    n = len(c)
    if kind == "ORIG":
        return df["entry_signal"].to_numpy().astype(bool)
    if kind == "RELAXED":
        # original minus ADX and calm gates
        eo = df["ema_order"].to_numpy(); rv = df["rsi"].to_numpy()
        ab = df["above_ema200"].to_numpy(); mp = df["macd_pos"].to_numpy()
        vol = df["vol_pit"].to_numpy(); atr = df["atr"].to_numpy()
        sig = ((eo >= 0.75) & (rv < 75) & (ab > 0.5) & (mp > 0.5) &
               np.isfinite(vol) & (vol >= 2e6) & (vol <= 2e8) & (atr > 0))
        sig[:S.MIN_HISTORY] = False
        return sig
    if kind == "BREAKOUT":
        # close > highest high of prior N bars, and volume > 1.5x its 20-avg
        N = 10
        hh = pd.Series(h).rolling(N).max().shift(1).to_numpy()
        vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        sig = (c > hh) & (v > 1.5 * vavg)
        sig[:30] = False
        return np.nan_to_num(sig).astype(bool)
    if kind == "EMA_CROSS":
        e5 = ema(c, 5); e20 = ema(c, 20); e50 = ema(c, 50)
        cross = (e5[1:] > e20[1:]) & (e5[:-1] <= e20[:-1])
        sig = np.zeros(n, dtype=bool)
        sig[1:] = cross & (c[1:] > e50[1:])
        sig[:50] = False
        return sig
    raise ValueError(kind)


def main():
    start, end = parse("2025-04-01"), parse("2025-05-01")
    ff = start - PB.WARMUP_BARS * FOUR_H
    print("Loading + featurising April 2025 ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for s, df in raw.items()}
    print(f"  {len(ps)} symbols\n")

    # best forward run per coin (opportunity) + where it started
    best = {}; best_start = {}
    for sym, df in ps.items():
        idx = [t for t in df.index if start <= t <= end]
        if not idx:
            continue
        a = df.loc[idx]; cl = a["close"].to_numpy(); hi = a["high"].to_numpy()
        bp = 0.0; bs = None
        for i in range(len(cl)):
            j = min(i + HOLD, len(cl))
            run = (hi[i:j].max() / cl[i] - 1) * 100
            if run > bp:
                bp = run; bs = cl[i]
        best[sym] = bp; best_start[sym] = bs
    runs = pd.Series(best)
    big_movers = set(runs[runs >= BIG].index)
    print(f"Big movers (>= {BIG:.0f}% run available): {len(big_movers)}\n")

    print(f"{'signal':<11}{'fires':>7}{'caught':>8}{'cover%':>8}"
          f"{'late%(med)':>12}{'availAfter%(med)':>17}")
    print("-" * 63)
    for kind in ["ORIG", "RELAXED", "BREAKOUT", "EMA_CROSS"]:
        total_fires = 0
        caught = set()
        late = []; avail = []
        for sym, df in ps.items():
            idx = [t for t in df.index if start <= t <= end]
            if not idx:
                continue
            sub = df.loc[idx]
            sig = signals_for(sub, kind)
            fires = np.where(sig)[0]
            total_fires += len(fires)
            if len(fires) == 0:
                continue
            cl = sub["close"].to_numpy(); hi = sub["high"].to_numpy()
            # first fire on this coin
            i0 = fires[0]
            if sym in big_movers:
                caught.add(sym)
                # lateness: how far above the run-start price we entered
                if best_start.get(sym):
                    late.append((cl[i0] / best_start[sym] - 1) * 100)
                # forward run still available after our entry
                j = min(i0 + HOLD, len(cl))
                avail.append((hi[i0:j].max() / cl[i0] - 1) * 100)
        cover = len(caught & big_movers) / len(big_movers) * 100 if big_movers else 0
        lm = np.median(late) if late else float("nan")
        am = np.median(avail) if avail else float("nan")
        print(f"{kind:<11}{total_fires:>7}{len(caught & big_movers):>8}{cover:>7.0f}%"
              f"{lm:>11.1f}%{am:>16.1f}%")
    print("\n(cover% = of big movers, how many this signal caught;")
    print(" late% = how far up we bought vs move start (lower=earlier=better);")
    print(" availAfter% = run still available after entry (higher=more upside left))")


if __name__ == "__main__":
    main()
