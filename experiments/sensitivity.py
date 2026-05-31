#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parameter-sensitivity / overfitting probe.

The strategy's entry thresholds (ema_order>=0.75, rsi<75, adx>=25, ...) were
hand-picked by studying historical charts. If the strong result sits on a
*fragile peak* in parameter space, nudging those thresholds slightly will make
performance swing wildly -> a hallmark of in-sample overfitting. If it is
*robust*, results stay in a tight band.

We featurise every symbol once, then only re-apply the (cheap) entry-signal
thresholds per variant and re-run the portfolio walk. Uses the cached archive
data, so no downloads.

Run:  python experiments/sensitivity.py 2024-01-01 2025-01-01
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


def main():
    start = parse(sys.argv[1] if len(sys.argv) > 1 else "2024-01-01")
    end = parse(sys.argv[2] if len(sys.argv) > 2 else "2025-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS

    symbols = PIT.list_all_usdt_symbols()
    print(f"Universe: {len(symbols)} symbols; loading cached archive data...")
    raw = PIT.prefetch_universe(symbols, fetch_from, end, log=lambda *a: None)
    # featurise ONCE (raw indicators don't depend on the thresholds)
    feats = {s: S.compute_features(df.copy()) for s, df in raw.items()}
    print(f"Featurised {len(feats)} symbols.\n")

    # config matching the user's latest spec
    cfg = dict(rank="vol", slippage=0.10, exit_model="pessimistic",
               capital_cap=1_000_000.0, source="archive")

    variants = [
        ("BASELINE (as hand-tuned)", {}),
        ("ema_order >= 0.70", {"EMA_ORDER_MIN": 0.70}),
        ("ema_order >= 0.80", {"EMA_ORDER_MIN": 0.80}),
        ("ema_order >= 1.00", {"EMA_ORDER_MIN": 1.00}),
        ("rsi < 70", {"RSI_MAX": 70.0}),
        ("rsi < 80", {"RSI_MAX": 80.0}),
        ("adx >= 20", {"ADX_MIN": 20.0}),
        ("adx >= 30", {"ADX_MIN": 30.0}),
        ("calm pctl 70", {"CALM_PCTL": 70.0}),
        ("calm pctl 90", {"CALM_PCTL": 90.0}),
    ]

    print(f"{'variant':<28}{'return%':>14}{'trades':>9}{'win%':>8}{'PF':>7}{'maxDD%':>9}")
    print("-" * 75)
    base_ret = None
    for label, overrides in variants:
        saved = {k: getattr(S, k) for k in overrides}
        for k, v in overrides.items():
            setattr(S, k, v)
        try:
            # rebuild only the entry-signal column under the new thresholds
            per_symbol = {}
            for sym, df in feats.items():
                d = S.attach_entry_signal(df.copy())
                per_symbol[sym] = d.set_index("close_time")
            rep = _run_prefeaturised(per_symbol, start, end, cfg)
        finally:
            for k, v in saved.items():
                setattr(S, k, v)

        closed = rep.trades
        wins = sum(1 for t in closed if t.net_pct > 0)
        gw = sum(t.net_pct for t in closed if t.net_pct > 0)
        gl = -sum(t.net_pct for t in closed if t.net_pct <= 0)
        pf = (gw / gl) if gl > 0 else float("inf")
        wr = (wins / len(closed) * 100) if closed else 0.0
        ret = rep.total_return * 100
        if base_ret is None:
            base_ret = ret
        flag = ""
        if base_ret:
            ratio = ret / base_ret if base_ret else 0
            if ratio < 0.5 or ratio > 2:
                flag = "  <-- swings >2x"
        print(f"{label:<28}{ret:>13,.0f}%{len(closed):>9}{wr:>7.1f}%{pf:>7.2f}"
              f"{rep.max_drawdown*100:>8.1f}%{flag}")


def _run_prefeaturised(per_symbol, start, end, cfg):
    """Call PB.run but skip its loader by injecting pre-featurised frames."""
    orig_fetch = PB.fetch_klines_range
    orig_compute = PB.S.compute_features
    orig_attach = PB.S.attach_entry_signal
    orig_prefetch = None
    import binance_sim.pit_universe as P
    orig_prefetch = P.prefetch_universe

    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    # archive branch calls PIT.prefetch_universe then compute/attach (identity);
    # feed it frames that already carry the entry_signal, reset for set_index.
    P.prefetch_universe = lambda syms, a, b, **k: {
        s: d.reset_index() for s, d in per_symbol.items()}
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        PB.fetch_klines_range = orig_fetch
        PB.S.compute_features = orig_compute
        PB.S.attach_entry_signal = orig_attach
        P.prefetch_universe = orig_prefetch


if __name__ == "__main__":
    main()
