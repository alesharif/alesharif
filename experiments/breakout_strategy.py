#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BREAKOUT strategy — full hybrid backtest on the four months.

ENTRY (4h close): close > highest-high of prior N bars AND volume > Vx * its
  20-bar average AND ATR/price <= MAX_ATR (skip collapsing coins).
  -> catches the START of a breakout (diagnosis: 94% coverage, enters +7% in).
EXIT (hybrid 5m + 30s zoom): ATR trailing stop; we sweep the trail width to
  let winners run. Hard stop = SL_ATR * ATR.

Compares trail widths {1.0, 1.5, 2.5} on each month, vs the ORIGINAL strategy
baseline already known. Per-month: trades, WR, PF, worst, return.

Run:  python experiments/breakout_strategy.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
SL_ATR = 1.5
MAX_ATR = 0.12
BREAK_N = 10            # breakout lookback
VOL_X = 1.5            # volume surge multiple
_SEC = {}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def sec_day(symbol, day):
    k = (symbol, day)
    if k not in _SEC:
        _SEC[k] = HR.load_day(symbol, "1s", day)
    return _SEC[k]


def add_breakout_signal(df):
    """Attach a 'bo_signal' column: breakout close + volume surge."""
    h = df["high"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["quote_av"].to_numpy(float); atr = df["atr"].to_numpy(float)
    hh = pd.Series(h).rolling(BREAK_N).max().shift(1).to_numpy()
    vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    ar = atr / np.where(c > 0, c, np.nan)
    sig = (c > hh) & (v > VOL_X * vavg) & np.isfinite(ar) & (ar <= MAX_ATR) & (atr > 0)
    sig = np.nan_to_num(sig).astype(bool)
    sig[:30] = False
    df["bo_signal"] = sig
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


def simulate(ps, times, end, trail):
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
            if not bool(row["bo_signal"]):
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
    return len(nets), wr, pf, min(nets, default=0), ret


def load(s_str, e_str):
    start, end = parse(s_str), parse(e_str)
    ff = start - PB.WARMUP_BARS * FOUR_H
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {}
    for s, df in raw.items():
        d = add_breakout_signal(S.compute_features(df.copy()))
        ps[s] = d.set_index("close_time")
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    return ps, times, end


def main():
    months = {
        "2025-04 bull": ("2025-04-01", "2025-05-01"),
        "2025-12 hard": ("2025-12-01", "2026-01-01"),
        "2026-04": ("2026-04-01", "2026-05-01"),
        "2026-05": ("2026-05-01", "2026-06-01"),
    }
    trails = [1.0, 1.5, 2.5]
    print("BREAKOUT strategy (hybrid 30s exit) — breakout entry + trailing stop")
    print(f"entry: close>HH{BREAK_N} & vol>{VOL_X}x & ATR<= {MAX_ATR*100:.0f}%  | SL {SL_ATR}xATR\n")
    print(f"{'trail':<7}{'month':<15}{'trades':>8}{'WR':>7}{'PF':>7}{'worst':>9}{'return':>9}")
    print("-" * 62)
    for name, (s, e) in months.items():
        print(f"  loading {name} ...", flush=True)
        ps, times, end = load(s, e)
        for tr in trails:
            nets = simulate(ps, times, end, tr)
            n, wr, pf, worst, ret = stats(nets)
            print(f"{tr:<7}{name:<15}{n:>8}{wr:>6.0f}%{pf:>7.2f}{worst:>8.1f}%{ret:>8.2f}%", flush=True)
            _SEC.clear()
        print("-" * 62)
        import gc; del ps; gc.collect()


if __name__ == "__main__":
    main()
