#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA-exit deep sweep — TF x span grid, screen on ONE month then validate.

Exit = TF close < EMA(span) + 2*ATR hard stop, on the real 1m path. Sweep
TF in {15,30,60,120,240}m x span in {10,15,20,30,50}. Screen month = 2026-04
(richest). Reuses each trade's 1m once (resample per TF once) -> fast.
Run:  python experiments/ema_sweep.py [YYYY-MM]   (default 2026-04)
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
POS_W = 0.125
FOUR = 4 * 3600 * 1000
HOLD_H = 48
HARD_SL_ATR = 2.0
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0
TFS = [15, 30, 60, 120, 240]
SPANS = [10, 15, 20, 30, 50]

MWIN = {"2025-12": ("2025-12-01", "2026-01-01"), "2026-01": ("2026-01-01", "2026-02-01"),
        "2026-02": ("2026-02-01", "2026-03-01"), "2026-03": ("2026-03-01", "2026-04-01"),
        "2026-04": ("2026-04-01", "2026-05-01"), "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def resample(tmin, hi, lo, cl, tf_min):
    grp = (tmin - tmin[0]) // (tf_min * 60 * 1000)
    H = []; L = []; C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            H.append(hi[m].max()); L.append(lo[m].min()); C.append(cl[m][-1]); T.append(tmin[m][-1])
    return np.array(H), np.array(L), np.array(C), np.array(T)


def exit_from_resample(H, L, C, T, sl_t, sl, sl_px, ep, span):
    et = epx = None
    if len(C) >= span + 2:
        e = ema(C, span)
        for j in range(span + 1, len(C)):
            if C[j] < e[j]:
                et = T[j]; epx = C[j]; break
    if sl_t is not None and (et is None or sl_t <= et):
        return (sl_px / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (C[-1] / ep - 1) * 100 - FEE


def trades_for(mname):
    s, e = MWIN[mname]; start, end = parse(s), parse(e); ff = start - PB.WARMUP_BARS * FOUR
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for sym, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    sr = PB.build_stable_ratio(ff, end)
    sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
    out = []; open_until = {}
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
            out.append((d["time"].to_numpy(), d["high"].to_numpy(float),
                        d["low"].to_numpy(float), d["close"].to_numpy(float), price, atr))
    del ps, raw; gc.collect()
    return out


def evaluate(paths):
    acc = {(tf, sp): [] for tf in TFS for sp in SPANS}
    for tmin, hi, lo, cl, ep, atr in paths:
        sl = ep - HARD_SL_ATR * atr
        sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
        sl_t = tmin[sl_i] if sl_i is not None else None
        sl_px = min(sl, cl[sl_i]) if sl_i is not None else None
        for tf in TFS:
            H, L, C, T = resample(tmin, hi, lo, cl, tf)
            for sp in SPANS:
                acc[(tf, sp)].append(exit_from_resample(H, L, C, T, sl_t, sl, sl_px, ep, sp))
    return acc


def main():
    mscreen = sys.argv[1] if len(sys.argv) > 1 else "2026-04"
    print(f"EMA-EXIT SWEEP — screen month {mscreen}\n", flush=True)
    paths = trades_for(mscreen)
    print(f"{len(paths)} trades\n", flush=True)
    acc = evaluate(paths)
    res = []
    for (tf, sp), nets in acc.items():
        a = np.array(nets); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        res.append((tf, sp, (a > 0).mean() * 100, pf, a.mean(), a.sum() * POS_W))
    res.sort(key=lambda x: x[5], reverse=True)
    print(f"{'TF':>5}{'EMA':>5}{'WR':>6}{'PF':>7}{'avg%':>8}{'ret(month)':>12}")
    print("-" * 43)
    for tf, sp, wr, pf, av, ret in res:
        tag = "  <= best" if (tf, sp) == (res[0][0], res[0][1]) else ""
        print(f"{tf:>5}{sp:>5}{wr:>5.0f}%{pf:>7.2f}{av:>7.2f}%{ret:>11.1f}%{tag}")
    print(f"\nTop-3 to validate on all months: {[(r[0],r[1]) for r in res[:3]]}")
    print("\nDONE_EMA_SWEEP.", flush=True)


if __name__ == "__main__":
    main()
