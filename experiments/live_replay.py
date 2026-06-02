#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay the LIVE bot's actual trades (attack_v4.py, started 31 May 2026) and
test what MODEL A (resting stop + 4h trail-from-high) vs MODEL B (check price at
4h close) would have done — vs what the live 30s bot actually did.

The live bot runs OUR strategy (4h entry, full filters + stable-ratio gate,
rank by volume, 8 concurrent, 12.5% each). It exits every 30s. We reconstruct
its entry set from the strategy itself (the entries are deterministic) over
31 May -> now, using data-api klines (which carry today's 4h candles, unlike
the 1-day-lagged archive), then simulate two 4h-only exit mechanics:

  A_RESTING   : stop is a resting order, fills intrabar at the STOP LEVEL;
                trailing ratcheted from each 4h candle HIGH  (= the backtest)
  B_CHECKCLOSE: look only at the 4h CLOSE; sell at the close price

Validation anchors from the user's screenshots (31 May 04:00 UTC entries):
  XLMUSDT  entry 0.2412  SL 0.21443
  IOTAUSDT entry 0.0632  SL 0.0596168
Actual 30s-bot exits (2 Jun): BNBUSDT -3.95%, HBARUSDT -4.95%, XLMUSDT -11.29% (all SL)

Run:  python experiments/live_replay.py
"""

from __future__ import annotations

import sys

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
POS_USD = 250.0
MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000

ENTRY_START = "2026-05-31 00:00"
END = "2026-06-03 00:00"
ACTUAL = {"BNBUSDT": -3.95, "HBARUSDT": -4.95, "XLMUSDT": -11.29}  # 30s-bot, all SL


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def exit_resting(rows, ep, atr):
    peak = ep; sl = ep - SL_ATR * atr; trailing = False
    for high, low, close, ct in rows:
        if low <= sl:
            return ct, sl, "SL/TRAIL"
        if high > peak:
            peak = high
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - TRAIL * atr)
    return ct, close, "OPEN"


def exit_checkclose(rows, ep, atr):
    peak = ep; sl = ep - SL_ATR * atr; trailing = False
    for high, low, close, ct in rows:
        if close <= sl:
            return ct, close, "SL/TRAIL"
        if close > peak:
            peak = close
            if (peak - ep) >= ACTIVATE * atr:
                trailing = True
            if trailing:
                sl = max(sl, peak - TRAIL * atr)
    return ct, close, "OPEN"


def main():
    syms = PIT.list_all_usdt_symbols()
    estart, end = parse(ENTRY_START), parse(END)
    ff = estart - (S.MIN_HISTORY + 5) * FOUR_H
    print(f"Loading 4h (data-api, has today) for {len(syms)} symbols ...", flush=True)
    ps = {}
    for i, sym in enumerate(syms, 1):
        df = PB.fetch_klines_range(sym, ff, end)
        if df is None or len(df) < S.MIN_HISTORY:
            continue
        df = S.attach_entry_signal(S.compute_features(df))
        ps[sym] = df.set_index("close_time")
        if i % 80 == 0:
            print(f"  {i}/{len(syms)} (usable {len(ps)})", flush=True)
    print(f"Usable: {len(ps)} symbols", flush=True)

    sr = PB.build_stable_ratio(ff, end)
    stable_block = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX) for t, v in sr.items()}

    # entry candle closes from 31 May onward
    times = sorted({int(t) for df in ps.values() for t in df.index if estart <= t <= end})

    # reconstruct portfolio entries (rank by vol, 8 concurrent, stable gate)
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
            rows = list(zip(after["high"].astype(float), after["low"].astype(float),
                            after["close"].astype(float), after.index.astype("int64")))
            if not rows:
                open_until[sym] = None
                continue
            xtA, xpA, ocA = exit_resting(rows, price, atr)
            xtB, xpB, ocB = exit_checkclose(rows, price, atr)
            netA = (xpA / price - 1) * 100 - (FEE if ocA != "OPEN" else 0)
            netB = (xpB / price - 1) * 100 - (FEE if ocB != "OPEN" else 0)
            trades.append(dict(sym=sym, et=t, ep=price, atr=atr, sl=price - SL_ATR * atr,
                               netA=netA, ocA=ocA, netB=netB, ocB=ocB))
            open_until[sym] = xtA   # use A's exit to free the slot (resting=backtest)

    # ---- report ----
    def ts(ms):
        return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%m-%d %H:%M")
    print("\n" + "=" * 92)
    print("LIVE BOT TRADE REPLAY — A_RESTING (=backtest) vs B_CHECKCLOSE")
    print("=" * 92)
    print(f"{'symbol':<11}{'entry(UTC)':<13}{'entryPx':>11}{'SL':>11}"
          f"{'A net%':>9}{'A out':>10}{'B net%':>9}{'B out':>10}{'actual%':>9}")
    print("-" * 92)
    sumA = sumB = 0.0
    for tr in sorted(trades, key=lambda x: x["et"]):
        a = ACTUAL.get(tr["sym"])
        astr = f"{a:+.2f}" if a is not None else "—"
        print(f"{tr['sym']:<11}{ts(tr['et']):<13}{tr['ep']:>11.5g}{tr['sl']:>11.5g}"
              f"{tr['netA']:>8.2f}%{tr['ocA']:>10}{tr['netB']:>8.2f}%{tr['ocB']:>10}{astr:>9}")
        sumA += tr["netA"]; sumB += tr["netB"]
    print("-" * 92)
    nA = [t["netA"] for t in trades]; nB = [t["netB"] for t in trades]
    def wr(n): return sum(1 for x in n if x > 0) / len(n) * 100 if n else 0
    print(f"trades={len(trades)}   A: sum {sumA:+.2f}%  WR {wr(nA):.0f}%   "
          f"B: sum {sumB:+.2f}%  WR {wr(nB):.0f}%")
    # anchor validation
    print("\nAnchor check (should match screenshots):")
    for tr in trades:
        if tr["sym"] in ("XLMUSDT", "IOTAUSDT"):
            print(f"  {tr['sym']}: entry {tr['ep']:.5g} (scr "
                  f"{'0.2412' if tr['sym']=='XLMUSDT' else '0.0632'}), "
                  f"SL {tr['sl']:.5g} (scr {'0.21443' if tr['sym']=='XLMUSDT' else '0.0596168'})")
    print("\nDONE_LIVE_REPLAY.", flush=True)


if __name__ == "__main__":
    main()
