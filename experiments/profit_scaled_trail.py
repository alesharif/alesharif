#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PROFIT-SCALED trail — tighten the exit once the gain is big (lock fast pumps).

Root-cause analysis: the oracle's edge = avoiding GIVE-BACK on fast/big pumps
(the slow 1h exit lags a sharp reversal) while not whipsawing slow grinds (the
fast 15m exit is too tight on small moves). Velocity failed as the classifier
(it's high during healthy launches too). The cleaner CAUSAL classifier is the
UNREALIZED GAIN itself: fast pumps produce a big gain then reverse, so tighten
the trail once we are up big; stay loose while the gain is small.

Per 15m bar (causal): if peak-gain >= TIGHTEN_AT -> TIGHT exit (15m close<EMA20);
else LOOSE exit (15m close<EMA80 ~ 1h). + 2*ATR hard stop. Once tightened, stay
tight. Test several TIGHTEN_AT. vs FIXED_1h (+30%), oracle (+168%).
Run:  python experiments/profit_scaled_trail.py
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
TIGHTENS = [0.05, 0.08, 0.12, 0.20]   # tighten the trail once up this much

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


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


def exit_fixed_1h(tmin, hi, lo, cl, ep, atr):
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    H, L, C, T = resample(tmin, hi, lo, cl, 60)
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


def exit_profit_scaled(tmin, hi, lo, cl, ep, atr, tighten_at):
    """Loose (EMA80/15m ~1h) until gain>=tighten_at, then tight (EMA20/15m)."""
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    sl_t = tmin[sl_i] if sl_i is not None else None
    H, L, C, T = resample(tmin, hi, lo, cl, 15)
    if len(C) < 82:
        return exit_fixed_1h(tmin, hi, lo, cl, ep, atr)
    e20 = ema(C, 20); e80 = ema(C, 80)
    peak = ep; tight = False; et = epx = None
    for j in range(81, len(C)):
        if H[j] > peak:
            peak = H[j]
        if (peak / ep - 1) >= tighten_at:
            tight = True
        brk = (C[j] < e20[j]) if tight else (C[j] < e80[j])
        if brk:
            et = T[j]; epx = C[j]; break
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_i]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def main():
    cols = ["FIXED_1h"] + [f"PS_{int(t*100)}%" for t in TIGHTENS]
    acc = {c: {m: [] for m in MONTHS} for c in cols}
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
                if d is None or len(d) < 90:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                acc["FIXED_1h"][mname].append(exit_fixed_1h(tmin, hi, lo, cl, price, atr))
                for tt in TIGHTENS:
                    acc[f"PS_{int(tt*100)}%"][mname].append(
                        exit_profit_scaled(tmin, hi, lo, cl, price, atr, tt))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### PROFIT-SCALED TRAIL (base entries) #####")
    print(f"{'exit':<11}{'trades':>7}{'WR':>6}{'PF':>7}{'avg%':>8}{'ret_ALL':>9}   per-month")
    print("-" * 80)
    for c in cols:
        alln = [x for m in MONTHS for x in acc[c][m]]
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        pm = " ".join(f"{m[2:]}:{np.array(acc[c][m]).sum()*0.125:+.0f}" for m in MONTHS)
        print(f"{c:<11}{len(a):>7}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{a.mean():>7.2f}%"
              f"{a.sum()*0.125:>8.1f}%   {pm}", flush=True)
    print("\nDONE_PROFIT_SCALED.", flush=True)


if __name__ == "__main__":
    main()
