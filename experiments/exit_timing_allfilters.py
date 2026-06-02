#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exit timing — ALL the ORIGINAL code's conditions & filters, FRAMES ONLY.

Follow-up to exit_timing_orig.py. The user asked: rerun the wait-for-close
exit-timing test but with EVERY condition/filter the real bot uses, on the
four candle-close timeframes only (15m / 1h / 2h / 4h) — NO 1s / no INSTANT
mode (so it is fast and 100% causal).

What "ALL the code's filters" means, taken straight from the real bot:
  * the full entry_signal  (ema_order>=0.75, rsi<75, calm_long>80th-pctl,
    price>EMA200, MACD>0, ADX>=25, 30d quote-vol in [2M,200M], ATR>0)
      -> already inside S.attach_entry_signal
  * the GLOBAL stable-ratio gate: when the four stablecoin pairs
    (USDC/FDUSD/TUSD/DAI vs USDT) see their 4h quote-volume spike above
    1.3x their trailing-30-bar average -> fear / instability -> block ALL
    new entries on that bar.  (binance_sim/portfolio_backtest.build_stable_ratio,
    fetched from data-api.binance.vision since api.binance.com is geo-blocked.)

Everything that was an *added* experiment (ATR cap, htf trend, near-high,
btc-regime, vol-sizing ...) is intentionally OFF — this is the original bot,
nothing more.

ORIGINAL exit geometry: SL 1.5*ATR, TRAIL 0.2*ATR (activate +0.2*ATR), no cap.
Exit rule (user's idea, no look-ahead): at each tf candle CLOSE, look at the
close price now -> if close<=trailing stop exit at that close; else raise the
trailing stop. Decision uses only the close, never the intrabar high/low.

Run:  python experiments/exit_timing_allfilters.py
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
OUT = "results_exit_timing_allfilters"

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
    "2026-04": ("2026-04-01", "2026-05-01"),
    "2026-05": ("2026-05-01", "2026-06-01"),
}
MODES = ["CLOSE_15m", "CLOSE_1h", "CLOSE_2h", "CLOSE_4h"]
CLOSE_TF = {"CLOSE_15m": "15m", "CLOSE_1h": "1h", "CLOSE_2h": "2h", "CLOSE_4h": "4h"}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def exit_on_close(symbol, et, ep, atr, end, tf):
    """Wait-for-close: act only on each tf candle CLOSE (price now).

    Original trailing geometry: the trail only arms once price has advanced
    at least ACTIVATE*ATR above entry; the stop then rides TRAIL*ATR under the
    running peak (never below the initial hard stop).
    """
    df = HR.load_range(symbol, tf, et + 1, end)
    if df is None or not len(df):
        return None, None, "OPEN"
    peak = ep
    sl = ep - SL_ATR * atr
    trailing = False
    for _, c in df.iterrows():
        close = float(c["close"])
        ct = int(c["close_time"]) if "close_time" in c else int(c["time"])
        if close <= sl:
            return ct, close, "X"          # exit at the close price
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
    tf = CLOSE_TF[mode]
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
        if stable_block.get(t, False):       # global fear gate: no new entries
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
            xt, xp, oc = exit_on_close(sym, t, price, atr, end, tf)
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
    return dict(n=len(nets), wr=round(wr, 0), pf=round(pf, 2),
                worst=round(min(nets, default=0), 1), ret=round(ret, 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("EXIT TIMING — ALL original filters (incl. stable-ratio), FRAMES ONLY\n")
    print(f"{'mode':<11}{'month':<10}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 60)
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        # GLOBAL stable-ratio gate, keyed by 4h close_time (aligns with entries)
        sr = PB.build_stable_ratio(ff, end)
        stable_block = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX)
                        for t, v in sr.items()}
        n_block = sum(1 for t in times if stable_block.get(t, False))
        print(f"  ({mname}: {len(ps)} symbols, {len(times)} 4h bars, "
              f"{n_block} blocked by stable-ratio)", flush=True)
        for mode in MODES:
            cf = f"{OUT}/{mname}__{mode}.json"
            if os.path.exists(cf):
                d = json.load(open(cf))
                st = {k: d[k] for k in ("n", "wr", "pf", "worst", "ret")}
            else:
                nets = simulate(ps, times, end, mode, stable_block)
                st = stats(nets)
                json.dump({"month": mname, "mode": mode, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
            print(f"{mode:<11}{mname:<10}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        print("-" * 60, flush=True)
        del ps, raw
        gc.collect()

    # aggregate ranking
    print("\n##### AGGREGATE (4 months) #####")
    print(f"{'mode':<11}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'ret_ALL':>9}   per-month")
    print("-" * 78)
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
            print(f"{mode:<11}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.1f}%   {pms}", flush=True)
    print("\nDONE_EXIT_TIMING_ALLFILTERS.", flush=True)


if __name__ == "__main__":
    main()
