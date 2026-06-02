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
CONFIGS = {
    "CUR":     dict(sl=1.5, tps=[], runner=1.0, trail=0.2, act=0.2),
    "WIDE3.5": dict(sl=3.5, tps=[(1.0, 0.4), (3.0, 0.3)], runner=0.3, trail=1.5, act=1.0),
    "WIDE5":   dict(sl=5.0, tps=[(1.0, 0.4), (3.0, 0.3)], runner=0.3, trail=1.5, act=1.0),
    "NOSTOP":  dict(sl=99.0, tps=[(1.0, 0.4), (3.0, 0.3)], runner=0.3, trail=1.5, act=1.0),
    "NOSTOP_T": dict(sl=99.0, tps=[], runner=1.0, trail=2.0, act=1.0),
}


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
    # accumulators per config
    acc = {name: dict(nets=[], caps=[], pm={m: [] for m in MONTHS}) for name in CONFIGS}
    ntr = 0
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
                    cands.append((sym, float(row["close"]), float(row["atr"]), float(row["vol_pit"])))
            cands.sort(key=lambda x: x[3], reverse=True)
            for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                if d is None or len(d) < 5:
                    continue
                hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
                cl = d["close"].to_numpy(float)
                mfe = (hi.max() / price - 1) * 100
                ntr += 1
                for name, cfg in CONFIGS.items():
                    r = simulate_exit(hi, lo, cl, price, atr, cfg)
                    net = r * 100 - FEE
                    acc[name]["nets"].append(net)
                    acc[name]["pm"][mname].append(net)
                    if mfe > 0.5:
                        acc[name]["caps"].append(max(net, -50) / mfe)
        print(f"  {mname}: cumulative {ntr} trades", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### EXIT OPTIMIZER ({ntr} trades) #####")
    print(f"{'config':<9}{'avg net%':>10}{'WR':>6}{'PF':>7}{'capture%':>10}   per-month return ($2000,12.5%)")
    print("-" * 82)
    for name in CONFIGS:
        nets = np.array(acc[name]["nets"])
        wins = nets[nets > 0]; losses = nets[nets <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        wr = (nets > 0).mean() * 100 if len(nets) else 0
        pmret = {m: (np.array(v).sum() * 0.125 if v else 0.0) for m, v in acc[name]["pm"].items()}
        pms = " ".join(f"{m[2:]}:{pmret[m]:+.0f}" for m in MONTHS)
        cap = np.median(acc[name]["caps"]) * 100 if acc[name]["caps"] else 0
        print(f"{name:<9}{nets.mean():>9.2f}%{wr:>5.0f}%{pf:>7.2f}{cap:>9.0f}%   {pms}", flush=True)
    print("\nDONE_EXIT_OPTIMIZER.", flush=True)


if __name__ == "__main__":
    main()
