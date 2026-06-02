#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A vs B: resting stop (=backtest) vs check-price-at-4h-close (user's literal idea).

The user wants the LIVE bot to act only every 4h. Two ways to implement that,
which we test head-to-head on the SAME entries / months / filters:

  A_RESTING   (= what the 53/0 backtest actually does):
      the stop is a RESTING order in the market. It fills the instant price
      touches it INTRABAR -> exit at the STOP LEVEL (controlled loss). The
      trailing stop is ratcheted every 4h using the candle HIGH.
      per 4h candle (pessimistic order): if low<=sl -> exit at sl; else raise
      trail from the high.

  B_CHECKCLOSE (= the user's literal "check the current price every 4h"):
      the bot looks ONLY at the 4h CLOSE price. If close<=sl -> market-sell at
      the CLOSE price (which, after a crash candle, is far below the stop).
      Trail ratcheted from the close.

Same geometry both sides: SL 1.5*ATR, TRAIL 0.2*ATR (activate +0.2*ATR), no TP
(the backtest's "profit limit" is the dynamic trailing lock). Full entry_signal
+ global stable-ratio<=1.3 gate. 4h frames only -> fast, fully causal.

Run:  python experiments/resting_vs_checkclose.py
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
SL_ATR = 1.5
TRAIL = 0.2
ACTIVATE = 0.2
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
OUT = "results_resting_vs_close"

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
    "2026-04": ("2026-04-01", "2026-05-01"),
    "2026-05": ("2026-05-01", "2026-06-01"),
}
MODES = ["A_RESTING", "B_CHECKCLOSE"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def exit_resting(symbol, et, ep, atr, end):
    """Model A: resting stop fills intrabar at the stop LEVEL; trail from high."""
    df = HR.load_range(symbol, "4h", et + 1, end)
    if df is None or not len(df):
        return None, None, "OPEN"
    peak = ep
    sl = ep - SL_ATR * atr
    trailing = False
    for _, c in df.iterrows():
        high = float(c["high"]); low = float(c["low"])
        ct = int(c["close_time"]) if "close_time" in c else int(c["time"])
        # pessimistic: the dip tests the pre-bar stop first (worst case)
        if low <= sl:
            return ct, sl, "X"                 # filled at the STOP LEVEL
        if high > peak:
            peak = high
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def exit_checkclose(symbol, et, ep, atr, end):
    """Model B: look only at the 4h CLOSE; market-sell at the close price."""
    df = HR.load_range(symbol, "4h", et + 1, end)
    if df is None or not len(df):
        return None, None, "OPEN"
    peak = ep
    sl = ep - SL_ATR * atr
    trailing = False
    for _, c in df.iterrows():
        close = float(c["close"])
        ct = int(c["close_time"]) if "close_time" in c else int(c["time"])
        if close <= sl:
            return ct, close, "X"              # market-sell at the CLOSE price
        if close > peak:
            peak = close
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, mode, stable_block):
    open_until = {}
    nets = []
    fn = exit_resting if mode == "A_RESTING" else exit_checkclose
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
        if stable_block.get(t, False):
            continue
        if len(open_until) >= MAX_CONC:
            continue
        cands = []
        for sym, df in ps.items():
            if sym in open_until or t not in df.index:
                continue
            row = df.loc[t]
            if not bool(row["entry_signal"]):
                continue
            cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            xt, xp, oc = fn(sym, t, price, atr, end)
            if oc == "OPEN":
                open_until[sym] = None
            else:
                nets.append((xp / price - 1) * 100 - FEE)
                open_until[sym] = xt
    return nets


def stats(nets):
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]
    gw = sum(wins)
    gl = -sum(losses)
    pf = gw / gl if gl else float("inf")
    wr = len(wins) / len(nets) * 100 if nets else 0
    ret = sum(POS_USD * n / 100 for n in nets) / 2000 * 100
    avg_l = (sum(losses) / len(losses)) if losses else 0.0
    return dict(n=len(nets), wr=round(wr, 0), pf=round(pf, 2),
                worst=round(min(nets, default=0), 1), avg_loss=round(avg_l, 2),
                ret=round(ret, 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("A_RESTING (=backtest) vs B_CHECKCLOSE (=check price at 4h close)\n")
    print(f"{'mode':<14}{'month':<10}{'trades':>7}{'WR':>6}{'PF':>7}{'avgLoss':>9}{'worst':>9}{'return':>9}")
    print("-" * 71)
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        sr = PB.build_stable_ratio(ff, end)
        stable_block = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX) for t, v in sr.items()}
        for mode in MODES:
            cf = f"{OUT}/{mname}__{mode}.json"
            if os.path.exists(cf):
                d = json.load(open(cf))
                st = {k: d[k] for k in ("n", "wr", "pf", "worst", "avg_loss", "ret")}
            else:
                nets = simulate(ps, times, end, mode, stable_block)
                st = stats(nets)
                json.dump({"month": mname, "mode": mode, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
            print(f"{mode:<14}{mname:<10}{st['n']:>7}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['avg_loss']:>8.1f}%{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        print("-" * 71, flush=True)
        del ps, raw
        gc.collect()

    print("\n##### AGGREGATE (4 months) #####")
    print(f"{'mode':<14}{'trades':>7}{'WR':>6}{'PF':>7}{'worst':>9}{'ret_ALL':>9}   per-month")
    print("-" * 80)
    for mode in MODES:
        an = []
        pm = {}
        for m in MONTHS:
            cf = f"{OUT}/{m}__{mode}.json"
            if os.path.exists(cf):
                d = json.load(open(cf))
                an += d["nets"]
                pm[m] = d["ret"]
        if an:
            st = stats(an)
            pms = " ".join(f"{x[2:]}:{pm.get(x, 0):+.0f}" for x in MONTHS)
            print(f"{mode:<14}{st['n']:>7}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.1f}%   {pms}", flush=True)
    print("\nDONE_RESTING_VS_CLOSE.", flush=True)


if __name__ == "__main__":
    main()
