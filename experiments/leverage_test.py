#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leverage simulation (curiosity) on our realistic edge (60m/E20, 6 months).

Compounds per-trade returns under leverage L. Each position uses 12.5% of equity
as margin controlling 12.5%*L notional. A position is liquidated if price moves
~1/L against it (per-trade loss floored at -100% of its margin). Reports final
equity multiple, ~monthly %, max drawdown, and ruin — to show whether leverage
turns the thin edge into the goal, or into destruction (volatility drag).
Run:  python experiments/leverage_test.py
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
LEVS = [1, 2, 3, 4, 5, 7, 10]

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
    trades = []   # (exit_time_proxy, net)
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
                net = exit_60_20(d["time"].to_numpy(), d["high"].to_numpy(float),
                                 d["low"].to_numpy(float), d["close"].to_numpy(float), price, atr)
                trades.append((t, net))
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    trades.sort()
    nets = np.array([x[1] for x in trades])
    print(f"\n##### LEVERAGE SIM ({len(nets)} trades, 6 months, compounded) #####")
    print(f"{'lev':>4}{'final x':>10}{'~/mo':>9}{'maxDD':>9}{'worst trade eq':>16}{'ruin?':>8}")
    print("-" * 58)
    for L in LEVS:
        eq = 1.0; peak = 1.0; mdd = 0.0; ruin = False; worst = 0.0
        for _, net in trades:
            ln = L * net / 100.0
            if ln < -1.0:
                ln = -1.0                     # liquidation: lose the position's margin
            delta = POS_W * ln
            worst = min(worst, delta * 100)
            eq *= (1 + delta)
            if eq <= 0:
                ruin = True; eq = 0.0; break
            peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
        permo = (eq ** (1 / 6) - 1) * 100 if eq > 0 else -100
        print(f"{L:>4}{eq:>9.2f}x{permo:>8.1f}%{mdd*100:>8.0f}%{worst:>15.1f}%{('YES' if ruin else 'no'):>8}")
    print("\n(note: sequential model; with 8 concurrent leveraged positions a")
    print(" correlated down-cluster makes real drawdown/ruin WORSE than shown)")
    print("\nDONE_LEVERAGE.", flush=True)


if __name__ == "__main__":
    main()
