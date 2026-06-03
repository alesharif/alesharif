#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-bracket grid (TP + SL) — order-aware on the real 1m path.

For each entry (4h signal + atr>=5%+adx>=40 + fear gate), place a fixed TP at
+TP% and SL at -SL%, and on the 1m path exit at WHICHEVER IS TOUCHED FIRST
(pessimistic: if both in one minute, SL first). Reports per (TP,SL): WR (TP-first),
reward:risk, expectancy/trade, total return, per-month. Finds if a fixed bracket
can hit WR>=50% with a good ratio. 6 months. Run: python experiments/bracket_grid.py
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
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-01": ("2026-01-01", "2026-02-01"),
          "2026-02": ("2026-02-01", "2026-03-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
TPS = [5, 7, 10, 15, 20]
SLS = [3, 5, 7, 10]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def bracket(hi, lo, cl, ep, tp, sl):
    """Return net% for TP/SL bracket, first-touch (SL wins ties). None if neither -> close."""
    tpx = ep * (1 + tp / 100.0); slx = ep * (1 - sl / 100.0)
    for i in range(len(cl)):
        if lo[i] <= slx:           # SL first (pessimistic)
            return -sl - FEE
        if hi[i] >= tpx:
            return tp - FEE
    return (cl[-1] / ep - 1) * 100 - FEE


def main():
    # collect per-trade 1m paths once (entry list), then evaluate all brackets
    paths = []   # (month, hi, lo, cl, ep)
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
                cands.append((sym, price, float(row["vol_pit"])))
            cands.sort(key=lambda x: x[2], reverse=True)
            for sym, price, _ in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 60:
                    continue
                paths.append((mname, d["high"].to_numpy(float), d["low"].to_numpy(float),
                              d["close"].to_numpy(float), price))
        print(f"  {mname} done ({len(paths)})", flush=True)
        del ps, raw; gc.collect()

    nm = len(MONTHS)
    print(f"\n##### FIXED-BRACKET GRID ({len(paths)} trades, {nm} months, first-touch) #####")
    print(f"{'TP/SL':<9}{'R:R':>6}{'WR':>6}{'expect':>9}{'ret_ALL':>9}{'~/mo':>7}   per-month")
    print("-" * 80)
    best = None
    for tp in TPS:
        for sl in SLS:
            nets = {m: [] for m in MONTHS}
            alln = []
            for mname, hi, lo, cl, ep in paths:
                r = bracket(hi, lo, cl, ep, tp, sl)
                nets[mname].append(r); alln.append(r)
            a = np.array(alln)
            wr = (a > 0).mean() * 100
            exp = a.mean()
            ret = a.sum() * POS_W
            pm = " ".join(f"{m[2:]}:{np.array(nets[m]).sum()*POS_W:+.0f}" for m in MONTHS)
            print(f"{f'{tp}/{sl}':<9}{tp/sl:>6.1f}{wr:>5.0f}%{exp:>8.2f}%{ret:>8.0f}%{ret/nm:>6.1f}%   {pm}", flush=True)
            if best is None or ret > best[0]:
                best = (ret, tp, sl, wr)
    print(f"\nBEST: TP/SL {best[1]}/{best[2]}  ret {best[0]:.0f}% (~{best[0]/nm:.1f}%/mo)  WR {best[3]:.0f}%")
    print("\nDONE_BRACKET.", flush=True)


if __name__ == "__main__":
    main()
