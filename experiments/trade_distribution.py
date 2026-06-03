#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trade-distribution report for our strategy (4h + atr>=5%+adx>=40 + EMA60_20).

Per-month trade count, and how many WINNERS reach >=3/5/7/9/10/12/15/20/25/30%,
and how many LOSERS reach <=-3/.../-30%. Realistic net (EMA60_20 exit on 1m).
6 months. Run:  python experiments/trade_distribution.py
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
FOUR = 4 * 3600 * 1000
HOLD_H = 48
HARD_SL_ATR = 2.0
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-01": ("2026-01-01", "2026-02-01"),
          "2026-02": ("2026-02-01", "2026-03-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
WIN_BK = [3, 5, 7, 9, 10, 12, 15, 20, 25, 30]
LOSS_BK = [3, 5, 7, 9, 10, 12, 15, 20, 25, 30]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def exit_ema60(tmin, hi, lo, cl, ep, atr):
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    grp = (tmin - tmin[0]) // (3600 * 1000)
    C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            C.append(cl[m][-1]); T.append(tmin[m][-1])
    C = np.array(C); T = np.array(T)
    et = epx = None
    if len(C) >= 22:
        e = ema(C, 20)
        for j in range(21, len(C)):
            if C[j] < e[j]:
                et = T[j]; epx = C[j]; break
    sl_t = tmin[sl_i] if sl_i is not None else None
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_i]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def main():
    permonth = {m: [] for m in MONTHS}
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e); ff = start - PB.WARMUP_BARS * FOUR
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
                if d is None or len(d) < 60:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                permonth[mname].append(exit_ema60(tmin, hi, lo, cl, price, atr))
        print(f"  {mname}: {len(permonth[mname])} trades", flush=True)
        del ps, raw; gc.collect()

    alln = np.array([x for m in MONTHS for x in permonth[m]])
    nmonths = len(MONTHS)
    wins = alln[alln > 0]; losses = alln[alln <= 0]
    print(f"\n##### TRADE DISTRIBUTION ({len(alln)} trades, {nmonths} months) #####")
    print(f"total trades: {len(alln)}  |  per-month avg: {len(alln)/nmonths:.0f}")
    print(f"winners: {len(wins)} ({len(wins)/len(alln)*100:.0f}%)   losers: {len(losses)} ({len(losses)/len(alln)*100:.0f}%)")
    print(f"avg win: {wins.mean():+.1f}%   avg loss: {losses.mean():+.1f}%   "
          f"PF: {wins.sum()/-losses.sum():.2f}")
    print(f"\n{'WINNERS >= X%':<16}{'count':>7}{'/month':>8}{'% of all':>9}")
    print("-" * 41)
    for b in WIN_BK:
        c = int((wins >= b).sum())
        print(f"{('win >= +'+str(b)+'%'):<16}{c:>7}{c/nmonths:>8.1f}{c/len(alln)*100:>8.0f}%")
    print(f"\n{'LOSERS <= -X%':<16}{'count':>7}{'/month':>8}{'% of all':>9}")
    print("-" * 41)
    for b in LOSS_BK:
        c = int((losses <= -b).sum())
        print(f"{('loss <= -'+str(b)+'%'):<16}{c:>7}{c/nmonths:>8.1f}{c/len(alln)*100:>8.0f}%")
    print(f"\nper-month trade counts: " + " ".join(f"{m[2:]}:{len(permonth[m])}" for m in MONTHS))
    print("\nDONE_DIST.", flush=True)


if __name__ == "__main__":
    main()
