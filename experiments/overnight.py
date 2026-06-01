#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OVERNIGHT autonomous run: tune entry + exit + filters to cure all 3 ills.

Runs unattended to completion. For each candidate strategy config it does a
hybrid (30s) backtest over the four dev months, saves results to disk
(results_overnight/), and prints a ranked summary at the end.

The three ills being attacked:
  1) late entry / missed movers -> BREAKOUT entry (close>HHn + vol surge)
  2) early exit  -> wider trailing stops (let winners run)
  3) toxic coins -> ATR/price cap

Grid (kept modest so it finishes overnight):
  entry   : ORIG, BREAKOUT(HH10), BREAKOUT(HH20)
  trail   : 1.5, 2.5, 4.0   (ATR multiples; wider = let it run)
  atr_cap : 0.12
Each config x 4 months, hybrid 30s exits. Resumable: skips configs already saved.

Run:  python experiments/overnight.py
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
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
ATR_CAP = 0.12
OUT = "results_overnight"
_SEC = {}

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
    "2026-04": ("2026-04-01", "2026-05-01"),
    "2026-05": ("2026-05-01", "2026-06-01"),
}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def add_signals(df):
    """Attach ORIG entry signal + breakout signals (HH10, HH20)."""
    df = S.attach_entry_signal(S.compute_features(df))
    h = df["high"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["quote_av"].to_numpy(float); atr = df["atr"].to_numpy(float)
    ar = atr / np.where(c > 0, c, np.nan)
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    okvol = v > 1.5 * vavg
    okatr = np.isfinite(ar) & (ar <= ATR_CAP) & (atr > 0)
    for N in (10, 20):
        hh = pd.Series(h).rolling(N).max().shift(1).to_numpy()
        sig = np.nan_to_num((c > hh) & okvol & okatr).astype(bool)
        sig[:30] = False
        df[f"bo{N}"] = sig
    # ORIG with the same ATR cap applied (fair comparison)
    o = df["entry_signal"].to_numpy().astype(bool) & okatr
    df["orig_cap"] = o
    return df


def hybrid_exit(symbol, et, ep, atr, end, trail):
    peak = ep; sl = ep - SL_ATR * atr; trailing = False
    five = HR.load_range(symbol, "5m", et + 1, end)
    if five is None or not len(five):
        return None, None, "OPEN"
    margin = 1.0 * atr
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if not ((lo <= sl + margin) or (hi >= peak)):
            if hi > peak:
                peak = hi
                if peak - ep >= trail * atr:
                    trailing = True; sl = max(sl, peak - trail * atr)
            continue
        day = pd.Timestamp(int(c["time"]), unit="ms").strftime("%Y-%m-%d")
        s = sec_day(symbol, day)
        if s is not None and len(s):
            seg = s[(s["time"] >= c["time"]) & (s["time"] <= c["close_time"])]
            if len(seg):
                for _, s1 in seg.iloc[::30].iterrows():
                    ph, pl = float(s1["high"]), float(s1["low"])
                    if ph > peak:
                        peak = ph
                        if peak - ep >= trail * atr:
                            trailing = True; sl = max(sl, peak - trail * atr)
                    if pl <= sl:
                        return int(s1["time"]), sl, ("TRAIL" if trailing else "SL")
                continue
        if lo <= sl:
            return int(c["close_time"]), sl, ("TRAIL" if trailing else "SL")
        if hi > peak:
            peak = hi
            if peak - ep >= trail * atr:
                trailing = True; sl = max(sl, peak - trail * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, sig_col, trail):
    open_until = {}; nets = []
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
        if len(open_until) >= MAX_CONC:
            continue
        cands = []
        for sym, df in ps.items():
            if sym in open_until or t not in df.index:
                continue
            row = df.loc[t]
            if not bool(row[sig_col]):
                continue
            cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            xt, xp, oc = hybrid_exit(sym, t, price, atr, end, trail)
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


def load(s_str, e_str):
    start, end = parse(s_str), parse(e_str)
    ff = start - PB.WARMUP_BARS * FOUR_H
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: add_signals(df.copy()).set_index("close_time") for s, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    return ps, times, end


ENTRIES = [("ORIG", "orig_cap"), ("BO10", "bo10"), ("BO20", "bo20")]
TRAILS = [1.5, 2.5, 4.0]


def main():
    os.makedirs(OUT, exist_ok=True)
    # process MONTH BY MONTH (load once per month, run all configs), save per cell
    for mname, (s, e) in MONTHS.items():
        print(f"\n##### MONTH {mname} #####", flush=True)
        # skip if all cells for this month already saved
        cells = [f"{OUT}/{mname}__{en}__{tr}.json" for en, _ in ENTRIES for tr in TRAILS]
        if all(os.path.exists(c) for c in cells):
            print(f"  all configs cached for {mname}, skipping", flush=True)
            continue
        ps, times, end = load(s, e)
        print(f"  loaded {len(ps)} symbols", flush=True)
        for en, col in ENTRIES:
            for tr in TRAILS:
                cf = f"{OUT}/{mname}__{en}__{tr}.json"
                if os.path.exists(cf):
                    continue
                nets = simulate(ps, times, end, col, tr)
                st = stats(nets)
                json.dump({"month": mname, "entry": en, "trail": tr, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
                print(f"  {mname} {en:<5} trail{tr:<4} -> "
                      f"n{st['n']} WR{st['wr']:.0f}% PF{st['pf']} "
                      f"worst{st['worst']}% ret{st['ret']}%", flush=True)
                _SEC.clear(); gc.collect()
        del ps; gc.collect()

    # ---- aggregate + rank ----
    print("\n\n##### FINAL RANKING (aggregate across 4 months) #####", flush=True)
    agg = {}
    for en, _ in ENTRIES:
        for tr in TRAILS:
            allnets = []
            permonth = {}
            for mname in MONTHS:
                cf = f"{OUT}/{mname}__{en}__{tr}.json"
                if os.path.exists(cf):
                    d = json.load(open(cf))
                    allnets += d["nets"]
                    permonth[mname] = d["ret"]
            if allnets:
                agg[(en, tr)] = (stats(allnets), permonth)
    print(f"{'entry':<7}{'trail':<7}{'trades':>8}{'WR':>6}{'PF':>7}{'worst':>9}{'ret_ALL':>9}"
          f"   per-month ret%")
    print("-" * 90)
    ranked = sorted(agg.items(), key=lambda kv: kv[1][0]["ret"], reverse=True)
    for (en, tr), (st, pm) in ranked:
        pms = " ".join(f"{m}:{pm.get(m,0):+.0f}" for m in MONTHS)
        print(f"{en:<7}{tr:<7}{st['n']:>8}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
              f"{st['worst']:>8.1f}%{st['ret']:>8.1f}%   {pms}", flush=True)
    json.dump({f"{en}_{tr}": v[0] for (en, tr), v in agg.items()},
              open(f"{OUT}/SUMMARY.json", "w"), indent=2)
    print("\nDONE. Full results in results_overnight/.", flush=True)


if __name__ == "__main__":
    main()
