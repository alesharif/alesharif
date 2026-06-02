#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which indicator gives the best PUMP EXIT? Compare on real 1m paths.

Entry base (best so far): 4h entry_signal + stable-ratio fear gate 1.15 +
per-coin quality (atr_ratio>=5% & adx>=40). Each exit = a protective 2*ATR hard
stop (monitored on 1m) PLUS an indicator trigger evaluated at each candle close
of its timeframe; we exit at whichever fires first.

Exits tested:
  TRAIL_2ATR : benchmark, wide 2*ATR trailing stop (no indicator)
  EMA15_7    : 15m close below EMA7
  EMA15_20   : 15m close below EMA20
  EMA60_20   : 1h  close below EMA20
  STOCHRSI15 : 15m StochRSI %K crosses below %D (bearish)
  MACD15     : 15m MACD line below signal (DIF<DEA)
  KDJ15      : 15m KDJ  K below D (bearish)

Realistic fills: indicator exit at the observed close; hard stop at its level.
Run:  python experiments/indicator_exit.py
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
FOUR_H = 4 * 3600 * 1000
HOLD_H = 48
HARD_SL_ATR = 2.0
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
CONFIGS = ["TRAIL_2ATR", "EMA15_7", "EMA15_20", "EMA60_20", "STOCHRSI15", "MACD15", "KDJ15"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def resample_ohlc(tmin, hi, lo, cl, tf_min):
    n = len(cl); grp = np.arange(n) // tf_min
    H = []; L = []; C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if not m.any():
            continue
        H.append(hi[m].max()); L.append(lo[m].min()); C.append(cl[m][-1]); T.append(tmin[m][-1])
    return np.array(H), np.array(L), np.array(C), np.array(T)


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rs = ru / np.where(rd > 1e-12, rd, np.nan)
    return 100 - 100 / (1 + rs)


def sig_stochrsi(H, L, C):
    r = rsi(C, 14); rs = pd.Series(r)
    lo = rs.rolling(14).min(); hi = rs.rolling(14).max()
    st = ((rs - lo) / (hi - lo).replace(0, np.nan) * 100)
    k = st.rolling(3).mean(); d = k.rolling(3).mean()
    return (k < d).to_numpy(), 17


def sig_macd(H, L, C):
    dif = ema(C, 12) - ema(C, 26); dea = ema(dif, 9)
    return (dif < dea), 26


def sig_kdj(H, L, C):
    n = 9
    lown = pd.Series(L).rolling(n).min().to_numpy()
    highn = pd.Series(H).rolling(n).max().to_numpy()
    rsv = (C - lown) / np.where((highn - lown) > 1e-12, highn - lown, np.nan) * 100
    k = np.full(len(C), 50.0); d = np.full(len(C), 50.0)
    for i in range(1, len(C)):
        rv = rsv[i] if np.isfinite(rsv[i]) else 50.0
        k[i] = 2 / 3 * k[i - 1] + 1 / 3 * rv
        d[i] = 2 / 3 * d[i - 1] + 1 / 3 * k[i]
    return (k < d), n + 1


def hard_stop_idx(lo, ep, atr):
    sl = ep - HARD_SL_ATR * atr
    for i in range(len(lo)):
        if lo[i] <= sl:
            return i, sl
    return None, sl


def exit_trail(hi, lo, cl, ep, atr):
    sl = ep - HARD_SL_ATR * atr; peak = ep; trailing = False
    for i in range(len(cl)):
        if lo[i] <= sl:
            return min(sl, cl[i]) / ep - 1
        if hi[i] > peak:
            peak = hi[i]
            if peak - ep >= 1.0 * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - 2.0 * atr)
    return cl[-1] / ep - 1


def exit_indicator(tmin, hi, lo, cl, ep, atr, tf_min, sigfn):
    sl_idx, sl = hard_stop_idx(lo, ep, atr)
    sl_t = tmin[sl_idx] if sl_idx is not None else None
    H, L, C, T = resample_ohlc(tmin, hi, lo, cl, tf_min)
    sig, warm = sigfn(H, L, C)
    ind_t = ind_px = None
    for j in range(warm, len(C)):
        if bool(sig[j]):
            ind_t = T[j]; ind_px = C[j]; break
    if sl_t is not None and (ind_t is None or sl_t <= ind_t):
        return min(sl, cl[sl_idx]) / ep - 1
    if ind_t is not None:
        return ind_px / ep - 1
    return cl[-1] / ep - 1


def run_exit(name, tmin, hi, lo, cl, ep, atr):
    if name == "TRAIL_2ATR":
        return exit_trail(hi, lo, cl, ep, atr)
    if name == "EMA15_7":
        return exit_indicator(tmin, hi, lo, cl, ep, atr, 15, lambda H, L, C: (C < ema(C, 7), 8))
    if name == "EMA15_20":
        return exit_indicator(tmin, hi, lo, cl, ep, atr, 15, lambda H, L, C: (C < ema(C, 20), 21))
    if name == "EMA60_20":
        return exit_indicator(tmin, hi, lo, cl, ep, atr, 60, lambda H, L, C: (C < ema(C, 20), 21))
    if name == "STOCHRSI15":
        return exit_indicator(tmin, hi, lo, cl, ep, atr, 15, sig_stochrsi)
    if name == "MACD15":
        return exit_indicator(tmin, hi, lo, cl, ep, atr, 15, sig_macd)
    if name == "KDJ15":
        return exit_indicator(tmin, hi, lo, cl, ep, atr, 15, sig_kdj)
    return cl[-1] / ep - 1


def main():
    acc = {c: [] for c in CONFIGS}
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
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
                for c in CONFIGS:
                    r = run_exit(c, tmin, hi, lo, cl, price, atr)
                    acc[c].append((r * 100 - FEE, mname))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### INDICATOR EXITS (fear<={FEAR}, atr>={ATR_MIN:.0%}+adx>={ADX_MIN:.0f}) #####")
    print(f"{'exit':<12}{'trades':>7}{'avg%':>8}{'WR':>6}{'PF':>7}   per-month return ($2000,12.5%)")
    print("-" * 80)
    for c in CONFIGS:
        rows = acc[c]; n = np.array([x[0] for x in rows])
        if len(n) == 0:
            continue
        wins = n[n > 0]; losses = n[n <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        wr = (n > 0).mean() * 100
        pm = {mm: 0.0 for mm in MONTHS}
        for net_i, mm in rows:
            pm[mm] += net_i * 0.125
        pms = " ".join(f"{mm[2:]}:{pm[mm]:+.0f}" for mm in MONTHS)
        print(f"{c:<12}{len(n):>7}{n.mean():>7.2f}%{wr:>5.0f}%{pf:>7.2f}   {pms}", flush=True)
    print("\nDONE_INDICATOR_EXIT.", flush=True)


if __name__ == "__main__":
    main()
