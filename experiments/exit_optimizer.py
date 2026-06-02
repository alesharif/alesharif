#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exit optimizer — replay each trade's REAL 1m path and test scaled-TP exits.

Goal (user's target): capture as much of the pump peak (MFE) as possible, with a
low failure rate, consistently across ALL months — realistically (TP limit
orders fill on the way up; stop/trail fill at their level on the way down).

For every taken trade we replay the cached 1m path (48h) and apply several
candidate exits, measuring realized net %, win rate, per-month return, and the
CAPTURE RATIO = realized% / MFE% (how much of the real peak we banked).

Candidate exits (ATR-based; portions sum to 1):
  CUR     : current bot — SL 1.5ATR, trail 0.2ATR (100%)            [baseline]
  TRAIL_W : SL 2ATR, trail 1ATR from peak (100%), arm after +0.5ATR  [wide trail]
  SCALE_A : SL 2ATR; +1ATR->40%, +2.5ATR->30%, runner 30% trail 1ATR; BE after TP1
  SCALE_B : SL 2ATR; +1.5ATR->50%, runner 50% trail 1ATR;            BE after TP1
  TP_ONLY : SL 2ATR; +1.5ATR->100% (single take-profit, no runner)

Realistic fills on the 1m path:
  * TP: filled when a 1m HIGH >= level  (limit sell on the way up)
  * stop/trail: filled when a 1m LOW <= level (fill at the level)
  * leftover at window end: marked out at the last close
Run repeatedly until DONE:  python experiments/exit_optimizer.py
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
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
HOLD_H = 48
OUT = "results_exit_optimizer"
PATHS = f"{OUT}/paths.json"

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


# ---- exit configs: (sl_atr, [(tp_atr, frac), ...], runner_frac, trail_atr, activate_atr) ----
EXIT = dict(sl=99.0, tps=[], runner=1.0, trail=2.0, act=1.0)   # NOSTOP_T (wide trail)
REGIME_GATE = False                                            # BTC gate OFF (too blunt)
# per-coin quality filters to test: (name, atr_ratio_min, adx_min)
FILTERS = [
    ("none",            0.00,  0),
    ("atr>=3%",         0.03,  0),
    ("atr>=4%",         0.04,  0),
    ("atr>=5%",         0.05,  0),
    ("atr>=4%+adx40",   0.04, 40),
    ("atr>=5%+adx40",   0.05, 40),
]


def simulate_exit(highs, lows, closes, ep, atr, cfg):
    """Replay 1m path, return realized net fraction (before fee) for this config."""
    sl = ep - cfg["sl"] * atr
    realized = 0.0
    remaining = 1.0
    tps = [[ep + lv * atr, fr, False] for lv, fr in cfg["tps"]]
    runner = cfg["runner"]
    peak = ep
    trailing = False
    be_done = False
    n = len(closes)
    for i in range(n):
        hi = highs[i]; lo = lows[i]
        # 1) hard/trail stop on whatever remains (fill at the level)
        if lo <= sl and remaining > 0:
            realized += remaining * (sl / ep - 1.0)
            remaining = 0.0
            return realized
        # 2) take-profit limit fills (on the way up)
        for tp in tps:
            if not tp[2] and hi >= tp[0]:
                realized += tp[1] * (tp[0] / ep - 1.0)
                remaining -= tp[1]
                tp[2] = True
                if not be_done:      # move stop to breakeven after first TP
                    sl = max(sl, ep)
                    be_done = True
        # 3) runner trailing (only the runner portion rides; arm + ratchet)
        if hi > peak:
            peak = hi
            if (peak - ep) >= cfg["act"] * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - cfg["trail"] * atr)
        if remaining <= 1e-9:
            return realized
    # window end: mark remaining at last close
    realized += remaining * (closes[-1] / ep - 1.0)
    return realized


def main():
    os.makedirs(OUT, exist_ok=True)
    # record every taken trade once: (net, atr_ratio, adx, month)
    recs = []
    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e)
        ff = start - PB.WARMUP_BARS * FOUR_H
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        sr = PB.build_stable_ratio(ff, end)
        sblock = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX) for t, v in sr.items()}
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
                if bool(row["entry_signal"]):
                    cands.append((sym, float(row["close"]), float(row["atr"]),
                                  float(row["vol_pit"]), float(row["adx"])))
            cands.sort(key=lambda x: x[3], reverse=True)
            for sym, price, atr, _, adx in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 5:
                    continue
                hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
                cl = d["close"].to_numpy(float)
                r = simulate_exit(hi, lo, cl, price, atr, EXIT)
                recs.append((r * 100 - FEE, atr / price, adx, mname))
        print(f"  {mname}: cumulative {len(recs)} trades", flush=True)
        del ps, raw; gc.collect()

    nets = np.array([x[0] for x in recs]); arx = np.array([x[1] for x in recs])
    adxx = np.array([x[2] for x in recs]); mon = [x[3] for x in recs]
    print(f"\n##### PER-COIN QUALITY FILTERS on NOSTOP_T ({len(recs)} trades), no BTC gate #####")
    print(f"{'filter':<16}{'trades':>7}{'avg%':>8}{'WR':>6}{'PF':>7}   per-month return ($2000,12.5%)")
    print("-" * 80)
    for name, amin, dmin in FILTERS:
        m = (arx >= amin) & (adxx >= dmin)
        n = nets[m]
        if len(n) == 0:
            continue
        wins = n[n > 0]; losses = n[n <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        wr = (n > 0).mean() * 100
        pm = {mm: 0.0 for mm in MONTHS}
        for net_i, mm in zip(nets[m], [mon[i] for i in range(len(mon)) if m[i]]):
            pm[mm] += net_i * 0.125
        pms = " ".join(f"{mm[2:]}:{pm[mm]:+.0f}" for mm in MONTHS)
        print(f"{name:<16}{len(n):>7}{n.mean():>7.2f}%{wr:>5.0f}%{pf:>7.2f}   {pms}", flush=True)
    print("\nDONE_EXIT_OPTIMIZER.", flush=True)


if __name__ == "__main__":
    main()
