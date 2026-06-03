#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CONVICTION SIZING — bet bigger on the pumps our data says will be biggest.

Best exit = EC_3h_g12 (early-speed classification, PF 1.20, ~+9%/mo). Now the
bigger lever: instead of equal 12.5% per trade, size by CONVICTION using the
features that predict pump size (atr_ratio, adx). High-conviction (high atr/adx)
pumps reach +29% vs +1.5% for low — so weighting toward them should lift returns.

We compute per-trade net (EC_3h_g12 exit) + atr_ratio + adx, then compare the
total return under sizing schemes (all normalized to the SAME average weight
0.125, so deployed capital is comparable):
  EQUAL        : 0.125 each (baseline)
  ATR_LIN      : weight ∝ atr_ratio
  ADX_LIN      : weight ∝ adx
  ATRxADX      : weight ∝ atr_ratio*adx
  TOPQ_2x      : 2x on top-quartile conviction (atr*adx), 0.66x on the rest
Also reports realized variance/worst-trade-weight to flag added risk.
Run:  python experiments/conviction_sizing.py
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
AVG_W = 0.125

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


def _break(tmin, hi, lo, cl, ep, atr, tf, span, start_ms):
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    sl_t = tmin[sl_i] if sl_i is not None else None
    H, L, C, T = resample(tmin, hi, lo, cl, tf)
    et = epx = None
    if len(C) >= span + 2:
        e = ema(C, span)
        for j in range(span + 1, len(C)):
            if T[j] >= start_ms and C[j] < e[j]:
                et = T[j]; epx = C[j]; break
    if sl_t is not None and (et is None or sl_t <= et):
        return (min(sl, cl[sl_i]) / ep - 1) * 100 - FEE
    if et is not None:
        return (epx / ep - 1) * 100 - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def exit_ec(tmin, hi, lo, cl, ep, atr, early_h=3, gthr=12.0):
    cutoff = tmin[0] + early_h * 3600 * 1000
    m = tmin <= cutoff
    early_gain = (hi[m].max() / ep - 1) * 100 if m.any() else 0.0
    tf, span = (15, 20) if early_gain >= gthr else (60, 20)
    return _break(tmin, hi, lo, cl, ep, atr, tf, span, cutoff)


def main():
    nets = []; ar = []; ad = []; mon = []
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
                cands.append((sym, price, atr, adx, float(row["vol_pit"])))
            cands.sort(key=lambda x: x[4], reverse=True)
            for sym, price, atr, adx, _ in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 90:
                    continue
                tmin = d["time"].to_numpy(); hi = d["high"].to_numpy(float)
                lo = d["low"].to_numpy(float); cl = d["close"].to_numpy(float)
                nets.append(exit_ec(tmin, hi, lo, cl, price, atr))
                ar.append(atr / price); ad.append(adx); mon.append(mname)
        print(f"  {mname} done ({len(nets)})", flush=True)
        del ps, raw; gc.collect()

    nets = np.array(nets); ar = np.array(ar); ad = np.array(ad)
    mon = np.array(mon)

    def norm_w(raw_w):
        w = raw_w / raw_w.mean() * AVG_W       # normalize so mean weight = 0.125
        return np.clip(w, 0, 0.30)             # cap any single position at 30%

    schemes = {
        "EQUAL":   np.full(len(nets), AVG_W),
        "ATR_LIN": norm_w(ar),
        "ADX_LIN": norm_w(ad),
        "ATRxADX": norm_w(ar * ad),
        "TOPQ_2x": None,
    }
    q = np.quantile(ar * ad, 0.75)
    topq = np.where(ar * ad >= q, 2.0, 0.667)
    schemes["TOPQ_2x"] = norm_w(topq)

    print(f"\n##### CONVICTION SIZING on EC_3h_g12 ({len(nets)} trades) #####")
    print(f"{'scheme':<10}{'ret_ALL':>9}{'~/mo':>8}{'maxW':>7}   per-month")
    print("-" * 62)
    for name, w in schemes.items():
        contrib = nets * w / 100.0            # fraction of $2000 per trade
        tot = contrib.sum() / (2000 / 2000) * 100  # % on 2000 base (w already fraction)
        tot = (nets * w).sum()                # sum(net% * weight) = portfolio %
        pm = {}
        for m in MONTHS:
            mask = mon == m
            pm[m] = (nets[mask] * w[mask]).sum()
        pms = " ".join(f"{m[2:]}:{pm[m]:+.0f}" for m in MONTHS)
        print(f"{name:<10}{tot:>8.1f}%{tot/4:>7.1f}%{w.max():>7.2f}   {pms}", flush=True)
    print("\n(EQUAL = our +30% baseline; does conviction weighting lift it?)")
    print("\nDONE_CONVICTION.", flush=True)


if __name__ == "__main__":
    main()
