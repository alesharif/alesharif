#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trail-timeframe sweep (ERROR-FREE): close-based trailing stop raised at each
X-candle close, for X in {15m, 1h, 2h, 4h}. Find the best trail update interval.

Identical, faithful mechanics for every timeframe:
  * entry every 4h (full entry_signal + global stable-ratio<=1.3 gate, rank vol)
  * trailing stop raised ONLY at each X-candle close, FROM THE CLOSE price
    (never the wick high -> never set above market -> no fictitious fill)
  * stop monitored every 30s on 1s data; REALISTIC fill at the market price
    (never above the stop level)
  * SL 1.5*ATR, TRAIL 0.2*ATR (armed after +0.2*ATR)

Shorter X = tighter/faster trail (closer to continuous); longer X = looser
(more breathing room). 4 months incl. the Dec-2025 stress month. Resumable.
Run repeatedly until DONE:  python experiments/trail_tf_sweep.py
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
OUT = "results_trail_tf_sweep"
_SEC = {}

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
TFS = {"15m": 15 * 60 * 1000, "1h": 3600 * 1000,
       "2h": 2 * 3600 * 1000, "4h": 4 * 3600 * 1000}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def exit_close_tf(symbol, et, ep, atr, end, tf_ms):
    """Close-based trail at tf_ms boundaries + 30s realistic stop monitoring."""
    sl = ep - SL_ATR * atr; peak = ep; trailing = False
    five = HR.load_range(symbol, "5m", et + 1, end)
    if five is None or not len(five):
        return None, None, "OPEN"
    for _, c in five.iterrows():
        lo = float(c["low"]); cl = float(c["close"]); ct = int(c["close_time"])
        # ---- 30s stop monitoring (realistic market fill, never above sl) ----
        if lo <= sl:
            day = pd.Timestamp(int(c["time"]), unit="ms").strftime("%Y-%m-%d")
            s = sec_day(symbol, day)
            if s is not None and len(s):
                seg = s[(s["time"] >= c["time"]) & (s["time"] <= c["close_time"])]
                if len(seg):
                    for _, s1 in seg.iloc[::30].iterrows():
                        p = float(s1["close"])
                        if p <= sl:
                            return int(s1["time"]), p, "X"
                else:
                    return int(c["close_time"]), min(sl, cl), "X"
            else:
                return int(c["close_time"]), min(sl, cl), "X"
        # ---- raise trail at the tf close, FROM THE CLOSE price ----
        if (ct + 1) % tf_ms == 0:
            if cl > peak:
                peak = cl
                if (peak - ep) >= ACTIVATE * atr:
                    trailing = True
                    sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, tf_ms, stable_block):
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
            xt, xp, oc = exit_close_tf(sym, t, price, atr, end, tf_ms)
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
    print("TRAIL-TIMEFRAME SWEEP (close-based, realistic) — 15m/1h/2h/4h\n")
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
        for tf, tf_ms in TFS.items():
            cf = f"{OUT}/{mname}__{tf}.json"
            if os.path.exists(cf):
                d = json.load(open(cf)); st = {k: d[k] for k in ("n","wr","pf","worst","ret")}
            else:
                nets = simulate(ps, times, end, tf_ms, stable_block)
                st = stats(nets)
                json.dump({"month": mname, "tf": tf, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
                _SEC.clear(); gc.collect()
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
    print("\nDONE_TRAIL_TF_SWEEP.", flush=True)


if __name__ == "__main__":
    main()
