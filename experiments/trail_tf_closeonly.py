#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model A — "look ONLY at the candle close and decide" — across 15m/1h/2h/4h.

The bot does NOT watch between candle closes. At each X-candle close it looks at
the CLOSE price and decides:
  * close <= stop      -> exit AT the close price (what it actually sees)
  * close > peak        -> raise the trailing stop from the close
  * otherwise           -> hold to the next X close
This is inherently realistic (it exits at the close price it observed — no
fictitious fill, no 1s needed). Entry every 4h + full entry_signal + global
stable-ratio<=1.3 gate, rank vol. SL 1.5*ATR, TRAIL 0.2*ATR (arm +0.2*ATR).

Pairs with trail_tf_sweep.py (model B = stop monitored every 30s). 4 months.
Run:  python experiments/trail_tf_closeonly.py
"""

from __future__ import annotations

import gc, json, os, sys
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
OUT = "results_trail_tf_closeonly"

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
TFS = ["15m", "1h", "2h", "4h"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def exit_closeonly(symbol, et, ep, atr, end, tf):
    """Act only at each tf close, on the close price. Exit AT the close."""
    df = HR.load_range(symbol, tf, et + 1, end)
    if df is None or not len(df):
        return None, None, "OPEN"
    sl = ep - SL_ATR * atr; peak = ep; trailing = False
    for _, c in df.iterrows():
        close = float(c["close"])
        ct = int(c["close_time"]) if "close_time" in c else int(c["time"])
        if close <= sl:
            return ct, close, "X"
        if close > peak:
            peak = close
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
                sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, tf, stable_block):
    open_until = {}; nets = []
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
        if stable_block.get(t, False) or len(open_until) >= MAX_CONC:
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
            xt, xp, oc = exit_closeonly(sym, t, price, atr, end, tf)
            if oc == "OPEN":
                open_until[sym] = None
            else:
                nets.append((xp / price - 1) * 100 - FEE)
                open_until[sym] = xt
    return nets


def stats(nets):
    wins = [n for n in nets if n > 0]; losses = [n for n in nets if n <= 0]
    gw = sum(wins); gl = -sum(losses)
    pf = gw / gl if gl else float("inf")
    wr = len(wins) / len(nets) * 100 if nets else 0
    ret = sum(POS_USD * n / 100 for n in nets) / 2000 * 100
    return dict(n=len(nets), wr=round(wr, 0), pf=round(pf, 2),
                worst=round(min(nets, default=0), 1), ret=round(ret, 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("MODEL A — look ONLY at candle close (close-based), 15m/1h/2h/4h\n")
    print(f"{'tf':<5}{'month':<10}{'trades':>7}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 53)
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        sr = PB.build_stable_ratio(ff, end)
        stable_block = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX) for t, v in sr.items()}
        for tf in TFS:
            cf = f"{OUT}/{mname}__{tf}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); st = {k: d[k] for k in ("n","wr","pf","worst","ret")}
            else:
                nets = simulate(ps, times, end, tf, stable_block)
                st = stats(nets)
                json.dump({"month": mname, "tf": tf, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
            print(f"{tf:<5}{mname:<10}{st['n']:>7}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        print("-" * 53, flush=True)
        del ps, raw; gc.collect()

    print("\n##### AGGREGATE (4 months) #####")
    print(f"{'tf':<5}{'trades':>7}{'WR':>6}{'PF':>7}{'ret_ALL':>9}   per-month")
    print("-" * 70)
    for tf in TFS:
        an = []; pm = {}
        for m in MONTHS:
            cf = f"{OUT}/{m}__{tf}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); an += d["nets"]; pm[m] = d["ret"]
        if an:
            st = stats(an)
            pms = " ".join(f"{x[2:]}:{pm.get(x,0):+.0f}" for x in MONTHS)
            print(f"{tf:<5}{st['n']:>7}{st['wr']:>5.0f}%{st['pf']:>7.2f}{st['ret']:>8.1f}%   {pms}", flush=True)
    print("\nDONE_CLOSEONLY.", flush=True)


if __name__ == "__main__":
    main()
