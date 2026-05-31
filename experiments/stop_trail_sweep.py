#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimise the hard-stop x trailing-stop geometry (Binance, 2022-2026).

The losing trades are big (full 1.5*ATR) while winners are tiny (0.2*ATR
trail). This sweeps SL_ATR x TRAIL_ATR to find a less asymmetric reward/risk.
Entry signals don't depend on the stops, so we featurise once and only re-run
the exit walk per combination.

Columns include avg win / avg loss so the reward:risk change is visible.

Run:  python experiments/stop_trail_sweep.py 2022-01-01 2026-01-01
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def _metrics(rep):
    c = rep.trades
    wins = [t.net_pct for t in c if t.net_pct > 0]
    losses = [t.net_pct for t in c if t.net_pct <= 0]
    gw, gl = sum(wins), -sum(losses)
    pf = (gw / gl) if gl > 0 else float("inf")
    wr = (len(wins) / len(c) * 100) if c else 0.0
    aw = (sum(wins) / len(wins)) if wins else 0.0
    al = (sum(losses) / len(losses)) if losses else 0.0
    return rep.total_return * 100, len(c), wr, aw, al, pf, rep.max_drawdown * 100


def main():
    start = parse(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
    end = parse(sys.argv[2] if len(sys.argv) > 2 else "2026-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS

    symbols = PIT.list_all_usdt_symbols()
    print(f"Universe: {len(symbols)} symbols; loading archive (cached) ...")
    raw = PIT.prefetch_universe(symbols, fetch_from, end, log=lambda *a: None)
    per_symbol = {s: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for s, df in raw.items()}
    print(f"Featurised {len(per_symbol)} symbols.\n")

    base = dict(rank="vol", exit_model="pessimistic", source="archive",
                vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05)

    sl_grid = [0.8, 1.0, 1.5, 2.0]
    trail_grid = [0.2, 0.5, 1.0]

    print("Stop x Trail sweep (mild costs, correct sizing). BASELINE = SL 1.5 / TRAIL 0.2")
    print(f"{'SL':>4}{'TRAIL':>7}{'return%':>14}{'trades':>8}{'win%':>7}"
          f"{'avgWin':>8}{'avgLoss':>9}{'PF':>6}{'maxDD%':>8}")
    print("-" * 71)
    for sl in sl_grid:
        for tr in trail_grid:
            saved = (S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR)
            S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = sl, tr, tr  # activate once up by trail
            try:
                rep = _run(per_symbol, start, end, base)
            finally:
                S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = saved
            ret, n, wr, aw, al, pf, dd = _metrics(rep)
            tag = "  <- baseline" if (sl == 1.5 and tr == 0.2) else ""
            print(f"{sl:>4}{tr:>7}{ret:>13,.0f}%{n:>8}{wr:>6.1f}%"
                  f"{aw:>7.2f}%{al:>8.2f}%{pf:>6.2f}{dd:>7.1f}%{tag}")


def _run(per_symbol, start, end, cfg):
    import binance_sim.pit_universe as P
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, P.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    P.prefetch_universe = lambda syms, a, b, **k: {
        s: d.reset_index() for s, d in per_symbol.items()}
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, P.prefetch_universe) = of, oc, oa, op


if __name__ == "__main__":
    main()
