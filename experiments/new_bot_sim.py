#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful simulation of the FIXED live bot (paper_bot_v2.0_4h-trail) over a
full month, vs the OLD bot, on identical entries.

NEW bot (paper_bot_v2_fixed.py) mechanics, reproduced exactly:
  * entry every 4h (full entry_signal + global stable-ratio<=1.3 gate, rank vol)
  * STOP monitored every 30s on the live price (1s data stepped at 30s); on a
    touch it books the exit at the STOP LEVEL (sl_price), as the code does.
  * TRAILING stop raised ONLY at each 4h candle close, from that candle's HIGH
    (peak - 0.2*ATR, armed after +0.2*ATR). No intra-candle ratcheting.

OLD bot (the bug): same, but the trail is ratcheted every 30s from the live
price (continuous) — reproduced with the old hybrid engine.

Month: 2026-05. Output: trades / WR / PF / worst / return for NEW vs OLD, so we
see the lift from moving the trail to 4h. Run: python experiments/new_bot_sim.py
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
OUT = "results_new_bot_sim"
_SEC = {}

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}
MODES = ["NEW_4h_trail", "OLD_30s_trail"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def exit_new(symbol, et, ep, atr, end):
    """NEW bot: 30s stop monitor (book at level) + trail raised only at 4h high."""
    sl = ep - SL_ATR * atr; peak = ep; trailing = False
    five = HR.load_range(symbol, "5m", et + 1, end)
    if five is None or not len(five):
        return None, None, "OPEN"
    cur4h_high = ep
    for _, c in five.iterrows():
        hi, lo = float(c["high"]), float(c["low"])
        ct = int(c["close_time"])
        if hi > cur4h_high:
            cur4h_high = hi
        # ---- 30s stop monitoring (REALISTIC fill: at market, never above sl) ----
        if lo <= sl:
            day = pd.Timestamp(int(c["time"]), unit="ms").strftime("%Y-%m-%d")
            s = sec_day(symbol, day)
            if s is not None and len(s):
                seg = s[(s["time"] >= c["time"]) & (s["time"] <= c["close_time"])]
                if len(seg):
                    for _, s1 in seg.iloc[::30].iterrows():   # sample every 30s
                        p = float(s1["close"])                 # 30s-polled last price
                        if p <= sl:
                            # fill at the actual market price (= p, which is <= sl),
                            # NEVER above market — removes the 4h-trail fictitious fill
                            return int(s1["time"]), p, "X"
                    # dipped between 30s samples but no sample caught it -> bot misses it
                else:
                    return int(c["close_time"]), min(sl, float(c["close"])), "X"
            else:
                return int(c["close_time"]), min(sl, float(c["close"])), "X"
        # ---- raise trailing ONLY at a 4h candle close (from the 4h high) ----
        if (ct + 1) % FOUR_H == 0:
            if cur4h_high > peak:
                peak = cur4h_high
                if (peak - ep) >= ACTIVATE * atr:
                    trailing = True
                    sl = max(sl, peak - TRAIL * atr)
            cur4h_high = float(c["close"])
    return None, None, "OPEN"


def exit_old(symbol, et, ep, atr, end):
    """OLD bot: 30s stop monitor + trail ratcheted continuously (every 30s/5m)."""
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
                if (peak - ep) >= ACTIVATE * atr:
                    trailing = True; sl = max(sl, peak - TRAIL * atr)
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
                            trailing = True; sl = max(sl, peak - TRAIL * atr)
                    if pl <= sl:
                        return int(s1["time"]), min(sl, float(s1["close"])), "X"
                continue
        if lo <= sl:
            return int(c["close_time"]), min(sl, float(c["close"])), "X"
        if hi > peak:
            peak = hi
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True; sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def simulate(ps, times, end, mode, stable_block):
    fn = exit_new if mode == "NEW_4h_trail" else exit_old
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
            xt, xp, oc = fn(sym, t, price, atr, end)
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
    print("NEW BOT (4h-trail) vs OLD BOT (30s-trail) — faithful 30s-stop sim, 2026-05\n")
    print(f"{'mode':<15}{'month':<10}{'trades':>7}{'WR':>6}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 63)
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
                d = json.load(open(cf)); st = {k: d[k] for k in ("n","wr","pf","worst","ret")}
            else:
                nets = simulate(ps, times, end, mode, stable_block)
                st = stats(nets)
                json.dump({"month": mname, "mode": mode, **st,
                           "nets": [round(x, 3) for x in nets]}, open(cf, "w"))
                _SEC.clear(); gc.collect()
            print(f"{mode:<15}{mname:<10}{st['n']:>7}{st['wr']:>5.0f}%{st['pf']:>7.2f}"
                  f"{st['worst']:>8.1f}%{st['ret']:>8.2f}%", flush=True)
        print("-" * 63, flush=True)
        del ps, raw; gc.collect()
    print("\nDONE_NEW_BOT_SIM.", flush=True)


if __name__ == "__main__":
    main()
