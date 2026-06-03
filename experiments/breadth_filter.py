#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Market-breadth regime filter — trade only when the market is broadly UP.

The losing months (Dec/Jan/Feb) drag the 6-month total to +5%. If a breadth gate
sits out 'cold' markets and keeps the 'hot' winning months (Mar/Apr), the total
should climb. Breadth(t) = fraction of liquid coins with close>EMA50(4h). Gate
entries when breadth < B. Exit = 60m/E20 + 2ATR. Test B thresholds vs no-gate.
Run:  python experiments/breadth_filter.py
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
BREADTHS = [0.0, 0.40, 0.50, 0.60]   # 0 = no gate

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"), "2026-01": ("2026-01-01", "2026-02-01"),
          "2026-02": ("2026-02-01", "2026-03-01"), "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"), "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def exit_60_20(tmin, hi, lo, cl, ep, atr):
    sl = ep - HARD_SL_ATR * atr
    sl_i = next((i for i in range(len(lo)) if lo[i] <= sl), None)
    grp = (tmin - tmin[0]) // (3600 * 1000); C = []; T = []
    for g in range(int(grp[-1]) + 1):
        m = grp == g
        if m.any():
            C.append(cl[m][-1]); T.append(tmin[m][-1])
    C = np.array(C); T = np.array(T); et = epx = None
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
    acc = {b: {m: [] for m in MONTHS} for b in BREADTHS}
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e); ff = start - PB.WARMUP_BARS * FOUR
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {}
        for sym, df in raw.items():
            d = S.attach_entry_signal(S.compute_features(df.copy()))
            d["e50"] = ema(d["close"].to_numpy(float), 50)
            ps[sym] = d.set_index("close_time")
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        # market breadth per timestamp = fraction of coins with close>EMA50
        breadth = {}
        for t in times:
            up = tot = 0
            for df in ps.values():
                if t in df.index:
                    r = df.loc[t]; tot += 1
                    if float(r["close"]) > float(r["e50"]):
                        up += 1
            breadth[t] = up / tot if tot else 0.0
        sr = PB.build_stable_ratio(ff, end)
        sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
        # one entry reconstruction per breadth threshold (gate differs -> diff trades)
        for B in BREADTHS:
            open_until = {}
            for t in times:
                open_until = {sy: u for sy, u in open_until.items() if u > t}
                if sblock.get(t, False) or len(open_until) >= MAX_CONC:
                    continue
                if breadth.get(t, 1.0) < B:        # breadth gate
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
                    acc[B][mname].append(exit_60_20(d["time"].to_numpy(), d["high"].to_numpy(float),
                                         d["low"].to_numpy(float), d["close"].to_numpy(float), price, atr))
        avgb = np.mean([breadth[t] for t in times]) if times else 0
        print(f"  {mname} done (avg breadth {avgb*100:.0f}%)", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### BREADTH GATE on 60m/E20 exit (6 months) #####")
    print(f"{'breadth>=':<10}{'trades':>7}{'WR':>6}{'PF':>7}{'ret_ALL':>9}   per-month")
    print("-" * 80)
    for B in BREADTHS:
        alln = [x for m in MONTHS for x in acc[B][m]]
        if not alln:
            print(f"{B:<10}  no trades"); continue
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        pm = " ".join(f"{m[2:]}:{np.array(acc[B][m]).sum()*POS_W:+.0f}" for m in MONTHS)
        lbl = "none" if B == 0 else f"{B:.0%}"
        print(f"{lbl:<10}{len(a):>7}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{a.sum()*POS_W:>8.0f}%   {pm}", flush=True)
    print("\nDONE_BREADTH.", flush=True)


if __name__ == "__main__":
    main()
