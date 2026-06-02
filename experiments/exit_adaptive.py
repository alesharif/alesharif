#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADAPTIVE exit — switch engine by market regime, decided at ENTRY (causal).

Finding so far (all original filters + stable-ratio gate):
  * BULL  month (e.g. 2025-04): CLOSE_4h is best  (+36%) — let the pump breathe.
  * HARSH month (e.g. 2025-12): CLOSE_4h is worst  (-34%); acting WITHIN the
    candle (INSTANT_30s) is far safer (Dec -0.7%).
So the exit engine should depend on the regime.

REGIME SWITCH (zero look-ahead): at the moment a trade is opened, look at BTC's
higher-timeframe trend at that same 4h bar (BTCUSDT close > EMA300 on 4h, the
bot's own htf_uptrend). Then:
  * BTC up   -> exit on 4h CLOSE   (give the explosion room to run)
  * BTC down -> exit INSTANT 30s   (hybrid 5m+1s, cut the moment price hits stop)
The choice is fixed for the life of that trade (decided only from data <= entry).

Filters: the full entry_signal + the global stable-ratio<=1.3 fear gate (the
protective config that killed the -78% toxic-coin blow-ups). Geometry: original
SL 1.5*ATR, TRAIL 0.2*ATR (activate +0.2*ATR).

We compute three columns for comparison (all with the SAME filters):
  ADAPTIVE     : the regime switch above
  INSTANT_filt : always INSTANT_30s
  CLOSE4h_filt : always CLOSE_4h   (read from results_exit_timing_allfilters)

Resumable: each (month, mode) cached to JSON. Run:  python experiments/exit_adaptive.py
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
OUT = "results_exit_adaptive"
ALLFILT = "results_exit_timing_allfilters"   # reuse CLOSE_4h cells from here
_SEC = {}

MONTHS = {
    "2025-04": ("2025-04-01", "2025-05-01"),
    "2025-12": ("2025-12-01", "2026-01-01"),
    "2026-04": ("2026-04-01", "2026-05-01"),
    "2026-05": ("2026-05-01", "2026-06-01"),
}
MODES = ["ADAPTIVE", "INSTANT_filt"]   # CLOSE4h_filt pulled from ALLFILT


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def exit_instant(symbol, et, ep, atr, end):
    """Hybrid 30s: 5m scan, 1s zoom near the stop. Acts the moment price hits SL."""
    peak = ep
    sl = ep - SL_ATR * atr
    trailing = False
    five = HR.load_range(symbol, "5m", et + 1, end)
    if five is None or not len(five):
        return None, None, "OPEN"
    margin = 1.0 * atr
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        if not ((lo <= sl + margin) or (hi >= peak)):
            if hi > peak:
                peak = hi
                if (peak - ep) >= ACTIVATE * atr:
                    trailing = True
                if trailing:
                    sl = max(sl, peak - TRAIL * atr)
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
                        if (peak - ep) >= ACTIVATE * atr:
                            trailing = True
                        if trailing:
                            sl = max(sl, peak - TRAIL * atr)
                    if pl <= sl:
                        return int(s1["time"]), sl, "X"
                continue
        if lo <= sl:
            return int(c["close_time"]), sl, "X"
        if hi > peak:
            peak = hi
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def exit_on_close(symbol, et, ep, atr, end, tf="4h"):
    """Wait-for-close: act only on each tf candle CLOSE (price now)."""
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
            return ct, close, "X"
        if close > peak:
            peak = close
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, mode, stable_block, btc_up):
    open_until = {}
    nets = []
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
        bull = btc_up.get(t, 1.0) >= 0.5     # regime at entry (causal)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            if mode == "INSTANT_filt":
                xt, xp, oc = exit_instant(sym, t, price, atr, end)
            elif mode == "ADAPTIVE":
                if bull:
                    xt, xp, oc = exit_on_close(sym, t, price, atr, end, "4h")
                else:
                    xt, xp, oc = exit_instant(sym, t, price, atr, end)
            else:
                xt, xp, oc = exit_on_close(sym, t, price, atr, end, "4h")
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
    print("ADAPTIVE EXIT (BTC-regime switch) vs pure engines — all filters + stable-ratio\n")
    print(f"{'mode':<14}{'month':<10}{'trades':>7}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 62)
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        sr = PB.build_stable_ratio(ff, end)
        stable_block = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX) for t, v in sr.items()}
        # BTC regime map (close_time -> htf_uptrend) from the prefetched universe
        btc = ps.get("BTCUSDT")
        if btc is None:
            btc = ps.get("BTC-USDT")
        btc_up = {int(t): float(v) for t, v in btc["htf_uptrend"].items()} if btc is not None else {}
        n_bull = sum(1 for t in times if btc_up.get(t, 1.0) >= 0.5)
        print(f"  ({mname}: {len(ps)} symbols, {len(times)} 4h bars; "
              f"BTC bull {n_bull}/{len(times)}, stable-blocked "
              f"{sum(1 for t in times if stable_block.get(t, False))})", flush=True)
        for mode in MODES:
            cf = f"{OUT}/{mname}__{mode}.json"
            if os.path.exists(cf):
                d = json.load(open(cf))
                st = {k: d[k] for k in ("n", "wr", "pf", "worst", "ret")}
            else:
                nets = simulate(ps, times, end, mode, stable_block, btc_up)
                st = stats(nets)
                json.dump({"month": mname, "mode": mode, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
                _SEC.clear()
                gc.collect()
            print(f"{mode:<14}{mname:<10}{st['n']:>7}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        # pull pure CLOSE_4h (all-filters) for the same month if available
        c4 = f"{ALLFILT}/{mname}__CLOSE_4h.json"
        if os.path.exists(c4):
            d = json.load(open(c4))
            print(f"{'CLOSE4h_filt':<14}{mname:<10}{d['n']:>7}{d['wr']:>5.0f}%{d['pf']:>7.2f}"
                  f"{d['worst']:>8.1f}%{d['ret']:>8.2f}%", flush=True)
        print("-" * 62, flush=True)
        del ps, raw
        gc.collect()

    print("\n##### AGGREGATE (4 months) #####")
    print(f"{'mode':<14}{'trades':>7}{'WR':>6}{'PF':>7}{'worst':>9}{'ret_ALL':>9}   per-month")
    print("-" * 80)
    pull = {"CLOSE4h_filt": ALLFILT}
    for mode in MODES + ["CLOSE4h_filt"]:
        an = []
        pm = {}
        src = pull.get(mode, OUT)
        fname = "CLOSE_4h" if mode == "CLOSE4h_filt" else mode
        for m in MONTHS:
            cf = f"{src}/{m}__{fname}.json"
            if os.path.exists(cf):
                d = json.load(open(cf))
                an += d["nets"]
                pm[m] = d["ret"]
        if an:
            st = stats(an)
            pms = " ".join(f"{x[2:]}:{pm.get(x, 0):+.0f}" for x in MONTHS)
            print(f"{mode:<14}{st['n']:>7}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.1f}%   {pms}", flush=True)
    print("\nDONE_ADAPTIVE.", flush=True)


if __name__ == "__main__":
    main()
