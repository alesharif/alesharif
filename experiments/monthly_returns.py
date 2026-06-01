#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Month-by-month returns for the two finalist stacks (Binance 2022-2026).

  S2 = freshness(age<=2) + golden EMA50>200            (return-optimal)
  S3 = S2 + near-high 0.95                             (safety-optimal)

Monthly return = % change of the mark-to-market equity curve per calendar month.

Run:  python experiments/monthly_returns.py
"""

from __future__ import annotations

import sys
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import daily_filters as DF               # noqa: E402


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def monthly(rep):
    s = pd.Series(rep.equity_curve,
                  index=pd.to_datetime(rep.equity_times, unit="ms"))
    m = s.resample("ME").last()
    ret = m.pct_change() * 100
    ret.iloc[0] = (m.iloc[0] / rep.initial_capital - 1) * 100
    return ret, m


def main():
    start, end = parse("2022-01-01"), parse("2026-01-01")
    fetch_from = start - PB.WARMUP_BARS * PB.FOUR_H_MS
    print("Loading + featurising Binance (cached) ...")
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), fetch_from, end, log=lambda *a: None)
    per_symbol = {s: DF.add_daily_filters(S.attach_entry_signal(S.compute_features(df.copy()))).set_index("close_time")
                  for s, df in raw.items()}
    print(f"  {len(per_symbol)} symbols\n")

    S.SL_ATR, S.TRAIL_ATR, S.ACTIVATE_ATR = 2.0, 0.2, 0.2
    common = dict(rank="vol", exit_model="pessimistic", source="archive",
                  vol_sizing=True, vol_threshold=30_000.0, vol_size_pct=0.001,
                  capital_cap=1_000_000.0, slippage=0.0, slip_atr=0.05, slip_impact=0.05,
                  max_signal_age=2, daily_filter_col="d_ema50_200")

    rep_s2 = _run(per_symbol, start, end, dict(common))
    rep_s3 = _run(per_symbol, start, end, dict(common, near_high_min=0.95))
    r2, _ = monthly(rep_s2)
    r3, _ = monthly(rep_s3)

    print(f"{'month':<10}{'S2 ret%':>10}{'S3 ret%':>10}")
    print("-" * 30)
    for ts in r2.index:
        m = ts.strftime("%Y-%m")
        print(f"{m:<10}{r2.loc[ts]:>9.1f}%{r3.loc[ts]:>9.1f}%")

    def stats(r, label):
        neg = (r < 0).sum()
        print(f"\n{label}: months={len(r)} | positive={len(r)-neg} | negative={neg} "
              f"| best={r.max():.1f}% | worst={r.min():.1f}% | median={r.median():.1f}%")
    stats(r2, "S2 (return-optimal)")
    stats(r3, "S3 (safety-optimal)")


def _run(per_symbol, start, end, cfg):
    of, oc, oa, op = (PB.fetch_klines_range, PB.S.compute_features,
                      PB.S.attach_entry_signal, PIT.prefetch_universe)
    PB.fetch_klines_range = lambda *a, **k: None
    PB.S.compute_features = lambda df: df
    PB.S.attach_entry_signal = lambda df: df
    PIT.prefetch_universe = lambda s, a, b, **k: {x: d.reset_index() for x, d in per_symbol.items()}
    try:
        return PB.run(list(per_symbol.keys()), start, end, log=lambda *a: None, **cfg)
    finally:
        (PB.fetch_klines_range, PB.S.compute_features,
         PB.S.attach_entry_signal, PIT.prefetch_universe) = of, oc, oa, op


if __name__ == "__main__":
    main()
