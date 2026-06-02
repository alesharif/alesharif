#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the TRAIL-update frequency explain the live-vs-backtest gap?

Same reconstructed live-bot trades (31 May -> 2 Jun, attack_v4.py strategy,
NO monitoring/age filters — it doesn't have them). Two exit models, both with
the stop monitored continuously (fills intrabar at the level):

  A_4h : trailing stop RAISED only at each 4h candle close (from the 4h high)
         = the backtest = the user's "stop 30s / exit 4h" idea.
  C_5m : trailing stop RAISED every 5m candle (from the 5m high)
         = a proxy for the live bot reading every 30s and ratcheting the trail
           continuously (5m is looser than 30s, so this UNDERSTATES the churn).

If A_4h >> C_5m, the trail-update frequency is the leak, and moving the trail to
4h-only (Model A) is the fix that recovers the backtest edge.

4h via data-api (has today); 5m via data-api /klines. Run: python experiments/trail_freq.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

SL_ATR = 1.5
TRAIL = 0.2
ACTIVATE = 0.2
FEE = 0.2
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
ENTRY_START = "2026-05-31 00:00"
END = "2026-06-03 00:00"
ACTUAL = {"BNBUSDT": -3.95, "HBARUSDT": -4.95, "XLMUSDT": -11.29}
_FIVE = {}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def fetch_5m(symbol, start_ms, end_ms):
    if symbol in _FIVE:
        return _FIVE[symbol]
    rows = []
    cur = start_ms
    while cur < end_ms:
        data = PB._http_get("/klines", {"symbol": symbol, "interval": "5m",
                                        "startTime": cur, "endTime": end_ms, "limit": 1000})
        if not data:
            break
        rows.extend(data)
        if len(data) < 1000:
            break
        cur = int(data[-1][6]) + 1
        time.sleep(0.03)
    if not rows:
        _FIVE[symbol] = None
        return None
    arr = [(float(r[2]), float(r[3]), float(r[4]), int(r[6])) for r in rows]  # high,low,close,ct
    _FIVE[symbol] = arr
    return arr


def exit_trail(rows, ep, atr):
    """rows = list of (high, low, close, close_time). Stop fills at level; trail from high."""
    peak = ep; sl = ep - SL_ATR * atr; trailing = False
    last_close = ep; last_ct = None
    for high, low, close, ct in rows:
        last_close, last_ct = close, ct
        if low <= sl:
            return ct, sl, "X"
        if high > peak:
            peak = high
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - TRAIL * atr)
    return last_ct, last_close, "OPEN"


def main():
    syms = PIT.list_all_usdt_symbols()
    estart, end = parse(ENTRY_START), parse(END)
    ff = estart - (S.MIN_HISTORY + 5) * FOUR_H
    print(f"Loading 4h for {len(syms)} symbols (entry reconstruction) ...", flush=True)
    ps = {}
    for i, sym in enumerate(syms, 1):
        df = PB.fetch_klines_range(sym, ff, end)
        if df is None or len(df) < S.MIN_HISTORY:
            continue
        ps[sym] = S.attach_entry_signal(S.compute_features(df)).set_index("close_time")
        if i % 120 == 0:
            print(f"  {i}/{len(syms)} (usable {len(ps)})", flush=True)
    print(f"Usable: {len(ps)}", flush=True)

    sr = PB.build_stable_ratio(ff, end)
    stable_block = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX) for t, v in sr.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if estart <= t <= end})

    open_until = {}
    trades = []
    for t in times:
        open_until = {s: u for s, u in open_until.items() if u is None or u > t}
        if stable_block.get(t, False) or len(open_until) >= MAX_CONC:
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
            df = ps[sym]
            after = df[df.index > t]
            rows4 = list(zip(after["high"].astype(float), after["low"].astype(float),
                             after["close"].astype(float), after.index.astype("int64")))
            if not rows4:
                open_until[sym] = None
                continue
            xtA, xpA, ocA = exit_trail(rows4, price, atr)             # 4h trail
            five = fetch_5m(sym, t + 1, end)                          # 5m trail
            if five:
                xtC, xpC, ocC = exit_trail(five, price, atr)
            else:
                xtC, xpC, ocC = xtA, xpA, ocA
            netA = (xpA / price - 1) * 100 - (FEE if ocA != "OPEN" else 0)
            netC = (xpC / price - 1) * 100 - (FEE if ocC != "OPEN" else 0)
            trades.append(dict(sym=sym, et=t, ep=price, netA=netA, ocA=ocA,
                               netC=netC, ocC=ocC))
            open_until[sym] = xtA

    def ts(ms):
        return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%m-%d %H:%M")
    print("\n" + "=" * 88)
    print("TRAIL FREQUENCY: A_4h (=backtest/your idea) vs C_5m (~live continuous trail)")
    print("=" * 88)
    print(f"{'symbol':<11}{'entry(UTC)':<13}{'A_4h net%':>11}{'A out':>9}"
          f"{'C_5m net%':>11}{'C out':>9}{'actual%':>9}")
    print("-" * 88)
    sA = sC = 0.0
    for tr in sorted(trades, key=lambda x: x["et"]):
        a = ACTUAL.get(tr["sym"])
        astr = f"{a:+.2f}" if a is not None else "—"
        print(f"{tr['sym']:<11}{ts(tr['et']):<13}{tr['netA']:>10.2f}%{tr['ocA']:>9}"
              f"{tr['netC']:>10.2f}%{tr['ocC']:>9}{astr:>9}")
        sA += tr["netA"]; sC += tr["netC"]
    print("-" * 88)
    nA = [t["netA"] for t in trades]; nC = [t["netC"] for t in trades]
    wr = lambda n: (sum(1 for x in n if x > 0) / len(n) * 100) if n else 0
    print(f"trades={len(trades)}   A_4h: sum {sA:+.2f}%  WR {wr(nA):.0f}%   "
          f"C_5m: sum {sC:+.2f}%  WR {wr(nC):.0f}%")
    print(f"capital impact (x0.125): A_4h {sA*0.125:+.2f}%   C_5m {sC*0.125:+.2f}%   "
          f"(bot actual ~ -3.2%)")
    print("\nDONE_TRAIL_FREQ.", flush=True)


if __name__ == "__main__":
    main()
