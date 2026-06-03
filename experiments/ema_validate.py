#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the top EMA-exit configs (from the 2026-04 screen) on ALL 6 months.

Configs: (15,20),(15,30),(30,10) [top-3 from screen] + (60,20) [robust baseline].
Exit = TF close < EMA(span) + 2*ATR stop. Reports each config's per-month return
+ total + PF, to expose over-fit (great on the bull screen month but failing on
choppy/hard months). Run:  python experiments/ema_validate.py
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

FEE = 0.2; MAX_CONC = 8; POS_W = 0.125; FOUR = 4 * 3600 * 1000
HOLD_H = 48; HARD_SL_ATR = 2.0; FEAR = 1.15; ATR_MIN = 0.05; ADX_MIN = 40.0

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"), "2026-01": ("2026-01-01", "2026-02-01"),
          "2026-02": ("2026-02-01", "2026-03-01"), "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"), "2026-05": ("2026-05-01", "2026-06-01")}
CONFIGS = [(15, 20), (15, 30), (30, 10), (60, 20)]


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


def exit_cfg(tmin, hi, lo, cl, ep, atr, tf, span):
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    sl_t = tmin[sl_i] if sl_i is not None else None
    H, L, C, T = resample(tmin, hi, lo, cl, tf)
    et = epx = None
    if len(C) >= span + 2:
        e = ema(C, span)
        for j in range(span + 1, len(C)):
            if C[j] < e[j]:
                et = T[j]; epx = C[j]; break
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_i]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def main():
    acc = {c: {m: [] for m in MONTHS} for c in CONFIGS}
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
                for tf, sp in CONFIGS:
                    acc[(tf, sp)][mname].append(exit_cfg(tmin, hi, lo, cl, price, atr, tf, sp))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### EMA-EXIT VALIDATION on 6 months #####")
    print(f"{'config':<10}{'WR':>6}{'PF':>7}{'ret_ALL':>9}{'~/mo':>7}   per-month")
    print("-" * 82)
    for tf, sp in CONFIGS:
        alln = [x for m in MONTHS for x in acc[(tf, sp)][m]]
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        ret = a.sum() * POS_W
        pm = " ".join(f"{m[2:]}:{np.array(acc[(tf,sp)][m]).sum()*POS_W:+.0f}" for m in MONTHS)
        print(f"{f'{tf}m/E{sp}':<10}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{ret:>8.0f}%{ret/6:>6.1f}%   {pm}", flush=True)
    print("\nDONE_EMA_VAL.", flush=True)


if __name__ == "__main__":
    main()
