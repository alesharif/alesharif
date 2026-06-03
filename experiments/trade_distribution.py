#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAW trade-potential report (NO exit) for our strategy's entries.

For each entry (4h signal + atr>=5%+adx>=40 + fear gate), over the next 48h on
1m data, we measure the RAW excursion regardless of any exit:
  MFE = max(high)/entry - 1   (how high it goes = upside potential)
  MAE = min(low)/entry  - 1   (how low it goes  = downside risk)
Then: how many trades' PEAK reaches >= +3/5/7/9/10/12/15/20/25/30%, and how many
trades' TROUGH reaches <= -3/.../-30%. 6 months. This shows the ceiling the exit
is trying to capture. Run:  python experiments/trade_distribution.py
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
                hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
                mfe = (hi.max() / price - 1) * 100
                mae = (lo.min() / price - 1) * 100
                permonth[mname].append((mfe, mae))
        print(f"  {mname}: {len(permonth[mname])} trades", flush=True)
        del ps, raw; gc.collect()

    rows = [x for m in MONTHS for x in permonth[m]]
    mfe = np.array([r[0] for r in rows]); mae = np.array([r[1] for r in rows])
    n = len(rows); nmonths = len(MONTHS)
    print(f"\n##### RAW TRADE POTENTIAL — NO EXIT ({n} trades, {nmonths} months, 48h window) #####")
    print(f"total trades: {n}  |  per-month avg: {n/nmonths:.0f}")
    print(f"mean PEAK (MFE): {mfe.mean():+.1f}%   median: {np.median(mfe):+.1f}%")
    print(f"mean TROUGH (MAE): {mae.mean():+.1f}%   median: {np.median(mae):+.1f}%")
    print(f"\n{'PEAK reaches >= X%':<20}{'count':>7}{'/month':>8}{'% of trades':>12}")
    print("-" * 48)
    for b in WIN_BK:
        c = int((mfe >= b).sum())
        print(f"{('peak >= +'+str(b)+'%'):<20}{c:>7}{c/nmonths:>8.1f}{c/n*100:>11.0f}%")
    print(f"\n{'TROUGH reaches <= -X%':<20}{'count':>7}{'/month':>8}{'% of trades':>12}")
    print("-" * 48)
    for b in LOSS_BK:
        c = int((mae <= -b).sum())
        print(f"{('trough <= -'+str(b)+'%'):<20}{c:>7}{c/nmonths:>8.1f}{c/n*100:>11.0f}%")
    print(f"\nper-month trade counts: " + " ".join(f"{m[2:]}:{len(permonth[m])}" for m in MONTHS))
    print("\nDONE_DIST.", flush=True)


if __name__ == "__main__":
    main()
