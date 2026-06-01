#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the user's idea: exit ONLY at 4h candle close (no intrabar 30s stop).

Compares three exit models on April 2025 (bull) + December 2025 (hard):
  A) hybrid 30s stop (reality)         : SL/trail checked every 30s (1s data)
  B) 4h-close-only (user's idea)       : ignore price intrabar; at each 4h
     close apply the SAME stop logic ONCE on that bar's low/high, plus an
     optional wide catastrophic stop.
  C) 4h-close + wide catastrophic 25%  : like B but a hard -25% guard intrabar.

Shows trades, WR, PF, worst, return per month, so we see if ignoring intrabar
wiggle (trusting the 4h frame) actually earns more — and how risky it is.

Run:  python experiments/exit_4h_vs_30s.py
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
SL_ATR = 1.5
TRAIL = 1.5
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
MAX_ATR = 0.12          # keep the ATR<=12% cap (best from prior sweep)
_SEC = {}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def exit_30s(symbol, et, ep, atr, end):
    """Model A: realistic hybrid — trail checked at 30s via 1s data."""
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
                if peak - ep >= TRAIL * atr:
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
                        if peak - ep >= TRAIL * atr:
                            trailing = True; sl = max(sl, peak - TRAIL * atr)
                    if pl <= sl:
                        return int(s1["time"]), sl, ("TRAIL" if trailing else "SL")
                continue
        if lo <= sl:
            return int(c["close_time"]), sl, ("TRAIL" if trailing else "SL")
        if hi > peak:
            peak = hi
            if peak - ep >= TRAIL * atr:
                trailing = True; sl = max(sl, peak - TRAIL * atr)
    return None, None, "OPEN"


def exit_4h_close(symbol, et, ep, atr, end, df4, cata=0.0):
    """Model B/C: decide only at each 4h close (no intrabar), like the backtest.

    At each 4h close after entry: update trailing from this bar's CLOSE, and
    exit if the CLOSE is at/below the stop. Optional catastrophic intrabar stop
    'cata' (fraction, e.g. 0.25) still triggers on the bar low for safety.
    """
    peak = ep; sl = ep - SL_ATR * atr; trailing = False
    bars = df4[(df4.index > et) & (df4.index <= end)]
    for ct, row in bars.iterrows():
        close = float(row["close"]); low = float(row["low"])
        # catastrophic intrabar guard (optional)
        if cata > 0 and low <= ep * (1 - cata):
            return int(ct), ep * (1 - cata), "CATA"
        # update trailing from the CLOSE (4h decision point)
        if close > peak:
            peak = close
            if peak - ep >= TRAIL * atr:
                trailing = True; sl = max(sl, peak - TRAIL * atr)
        # exit decision on the close
        if close <= sl:
            return int(ct), close, ("TRAIL" if trailing else "SL")
    return None, None, "OPEN"


def simulate(ps, times, end, mode):
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
            if not bool(row["entry_signal"]):
                continue
            ar = float(row["atr"]) / float(row["close"])
            if ar > MAX_ATR:
                continue
            cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
            if mode == "30s":
                xt, xp, oc = exit_30s(sym, t, price, atr, end)
            elif mode == "4h":
                xt, xp, oc = exit_4h_close(sym, t, price, atr, end, ps[sym], cata=0.0)
            else:  # 4h + catastrophic 25%
                xt, xp, oc = exit_4h_close(sym, t, price, atr, end, ps[sym], cata=0.25)
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
    return len(nets), wr, pf, min(nets, default=0), ret


def load(s_str, e_str):
    start, end = parse(s_str), parse(e_str)
    ff = start - PB.WARMUP_BARS * FOUR_H
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for s, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    return ps, times, end


def main():
    months = {"2025-04 (bull)": ("2025-04-01", "2025-05-01"),
              "2025-12 (hard)": ("2025-12-01", "2026-01-01")}
    modes = [("30s stop (reality)", "30s"),
             ("4h-close only (idea)", "4h"),
             ("4h + cata -25%", "4hc")]
    print("Exit model test: 30s stop vs 4h-close-only (SL1.5/TRAIL1.5, ATR<=12%)")
    print(f"{'exit model':<22}{'month':<16}{'trades':>8}{'WR':>7}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 78)
    for name, (s, e) in months.items():
        print(f"  loading {name} ...", flush=True)
        ps, times, end = load(s, e)
        for label, mode in modes:
            nets = simulate(ps, times, end, mode)
            n, wr, pf, worst, ret = stats(nets)
            print(f"{label:<22}{name:<16}{n:>8}{wr:>6.0f}%{pf:>7.2f}{worst:>8.1f}%{ret:>8.2f}%", flush=True)
            _SEC.clear()
        print("-" * 78)
        import gc; del ps; gc.collect()


if __name__ == "__main__":
    main()
