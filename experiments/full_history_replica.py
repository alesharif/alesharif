#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replicate the user's ORIGINAL backtest across ALL months (2021-10 -> 2026-02).

Goal: reproduce monthly_live.csv exactly — same strategy, all details, on the
full survivorship-free universe, with the user's described 4h exit logic:
  on each 4h candle, if price rose to lift the trailing target -> ride it; if it
  fell to the stop -> close on the drop; if between -> hold to next candle.
This is portfolio_backtest.run() with:
  * source     = archive   (point-in-time universe incl. delisted coins)
  * exit_model = pessimistic   (the user's "assume the worst": test the dip
                 against the stop before crediting the rise — no look-ahead)
  * default filters: full entry_signal + the global stable-ratio<=1.3 gate
  * compounding ON: each position = 12.5% of CURRENT capital (as in the file)

Output: a month-by-month table (trades, WR%, PF, profit$, capital$) printed
beside the user's own numbers, so we can see how faithfully the causal port
reproduces the original — and exactly where (and how much) it diverges.

Trades are cached to JSON so the monthly aggregation can be re-run instantly.
Run:  python experiments/full_history_replica.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

START = "2021-10-01"
END = "2026-03-01"            # cover through 2026-02 (last row of the file)
OUT = "results_full_replica"
TRADES_CACHE = f"{OUT}/trades.json"
HIS_CSV = "/root/.claude/uploads/572251c1-c91f-4915-aa3d-13346fc169e2/7647fce2-monthly_live.csv"


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def run_backtest():
    syms = PIT.list_all_usdt_symbols()
    print(f"Universe: {len(syms)} USDT symbols. Running full-history replica "
          f"({START} -> {END}, archive, pessimistic, compounding)...", flush=True)
    rep = PB.run(syms, parse(START), parse(END), source="archive",
                 exit_model="pessimistic", rank="vol", log=print)
    trades = [{"entry_time": t.entry_time, "exit_time": t.exit_time,
               "net_pct": round(t.net_pct, 4), "pnl": t.pnl, "outcome": t.outcome}
              for t in rep.trades]
    json.dump({"final_capital": rep.final_capital,
               "btc_buy_hold": rep.btc_buy_hold,
               "n_symbols": rep.n_symbols, "trades": trades},
              open(TRADES_CACHE, "w"))
    print(f"\nSaved {len(trades)} trades. Final capital: ${rep.final_capital:,.2f}", flush=True)
    return trades


def load_his():
    his = {}
    if not os.path.exists(HIS_CSV):
        return his
    for r in list(csv.reader(open(HIS_CSV)))[1:]:
        if len(r) >= 6:
            his[r[0]] = dict(n=int(r[1]), wr=float(r[2]), pf=float(r[3]),
                             profit=float(r[4]), cap=float(r[5]))
    return his


def month_of(ms):
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%Y-%m")


def aggregate(trades):
    by_month = defaultdict(list)
    for t in trades:
        by_month[month_of(t["exit_time"])].append(t)
    his = load_his()
    cap = S.INITIAL_CAPITAL
    rows = []
    for m in sorted(by_month):
        ts = by_month[m]
        nets = [t["net_pct"] for t in ts]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]
        gw = sum(wins); gl = -sum(losses)
        pf = gw / gl if gl > 0 else float("inf")
        wr = len(wins) / len(nets) * 100 if nets else 0
        profit = sum(t["pnl"] for t in ts)
        cap += profit
        rows.append(dict(month=m, n=len(ts), wr=wr, pf=pf, profit=profit, cap=cap))

    print("\n" + "=" * 96)
    print("FULL-HISTORY REPLICA (causal port)  vs  YOUR ORIGINAL FILE")
    print("=" * 96)
    print(f"{'month':<9}| {'OURS: trades':>12}{'WR':>5}{'PF':>7}{'capital$':>16} "
          f"|| {'YOURS: WR':>9}{'PF':>6}{'capital$':>16}")
    print("-" * 96)
    pos_months = 0
    for r in rows:
        h = his.get(r["month"])
        cap_s = f"{r['cap']:.3e}" if abs(r["cap"]) >= 1e7 else f"{r['cap']:,.0f}"
        if h:
            hcap_s = f"{h['cap']:.3e}" if h["cap"] >= 1e7 else f"{h['cap']:,.0f}"
            htxt = f"{h['wr']:>8.0f}%{h['pf']:>6.2f}{hcap_s:>16}"
        else:
            htxt = f"{'—':>9}{'—':>6}{'—':>16}"
        pf_s = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "inf"
        print(f"{r['month']:<9}| {r['n']:>12}{r['wr']:>4.0f}%{pf_s:>7}{cap_s:>16} "
              f"|| {htxt}")
        if r["profit"] > 0:
            pos_months += 1
    print("-" * 96)
    print(f"OURS: {len(rows)} months, {pos_months} winning / {len(rows)-pos_months} losing. "
          f"Final capital ${rows[-1]['cap']:.3e}" if rows else "no trades")
    if his:
        hwin = sum(1 for v in his.values() if v["profit"] > 0)
        print(f"YOURS: {len(his)} months, {hwin} winning / {len(his)-hwin} losing.")


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(TRADES_CACHE):
        print(f"Loading cached trades from {TRADES_CACHE} ...", flush=True)
        trades = json.load(open(TRADES_CACHE))["trades"]
    else:
        trades = run_backtest()
    aggregate(trades)
    print("\nDONE_FULL_REPLICA.", flush=True)


if __name__ == "__main__":
    main()
