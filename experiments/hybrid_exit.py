#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HYBRID exit: does snapping the blow-off top early beat the plain trail?

The reverse-mining (top_signature.py) found a robust, 6-month-validated mark of
a REAL top: a volume climax (>3x avg) — and, combined with a rejection wick or
upper-Bollinger break, precision rises to ~43-46% (lift 13-15x). But recall is
low (~20%): it only catches the BLOW-OFF tops, not slow rollovers.

So we don't replace the trail — we ADD to it. Baseline = our robust exit
(1h close < EMA20, + 2*ATR hard stop). Hybrid = same, but ALSO exit at the
close of any 1h candle where the climax combo fires (snap the top early).
Whichever triggers first wins. The ONLY honest test is P&L: does the early snap
net more across 6 months, or does the false-alarm half (we exit then it runs
on) eat the gain?  Stats != profit.

Variants of the climax combo:
  V1  uwick>0.5 & vol>3x                 (recall 19% / prec 43%)
  V2  bb_%b>1  & vol>3x                  (recall 27% / prec 25%, more snaps)
  V3  uwick>0.5 & vol>3x & bb_%b>1       (recall 13% / prec 46%, purest)

Run:  python experiments/hybrid_exit.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FEE = 0.2; MAX_CONC = 8; POS_W = 0.125; FOUR = 4 * 3600 * 1000
HOLD_H = 48; HARD_SL_ATR = 2.0; FEAR = 1.15; ATR_MIN = 0.05; ADX_MIN = 40.0
EMA_SPAN = 20; WARM_H = 60                # 1h warmup to seed ema20/volma20/bb

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"), "2026-01": ("2026-01-01", "2026-02-01"),
          "2026-02": ("2026-02-01", "2026-03-01"), "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"), "2026-05": ("2026-05-01", "2026-06-01")}

VARIANTS = ["baseline", "V1_uwick", "V2_bb", "V3_pure"]


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def climax_flags(o, h, l, c, v):
    """Return dict of boolean arrays for each climax-combo variant on 1h candles."""
    rng = np.where((h - l) == 0, 1e-9, h - l)
    uwick = (h - np.maximum(o, c)) / rng > 0.5
    volma = pd.Series(v).rolling(20).mean().to_numpy()
    vclim = v > 3 * volma
    sma20 = pd.Series(c).rolling(20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20).std().to_numpy()
    bb = c > (sma20 + 2 * std20)
    return {"V1_uwick": uwick & vclim,
            "V2_bb": bb & vclim,
            "V3_pure": uwick & vclim & bb}


def simulate(df, t0, ep, atr):
    """Return dict variant->return% for one trade, using 1h candles in df
    (df spans entry-WARM .. entry+HOLD so indicators are seeded)."""
    tm = df["time"].to_numpy()
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    e = ema(c, EMA_SPAN)
    cx = climax_flags(o, h, l, c, v)
    sl = ep - HARD_SL_ATR * atr
    idx = np.where(tm >= t0)[0]                       # holding-window candles
    if len(idx) == 0:
        return {k: 0.0 for k in VARIANTS}

    def run(use_climax):
        for i in idx:
            if i < EMA_SPAN + 1:
                continue
            if l[i] <= sl:                            # hard stop (intrabar)
                return (min(sl, c[i]) / ep - 1) * 100 - FEE
            if use_climax is not None and bool(cx[use_climax][i]) and np.isfinite(cx[use_climax][i]):
                return (c[i] / ep - 1) * 100 - FEE    # snap the blow-off top
            if c[i] < e[i]:                           # trend break (baseline exit)
                return (c[i] / ep - 1) * 100 - FEE
        return (c[idx[-1]] / ep - 1) * 100 - FEE      # timeout

    return {"baseline": run(None), "V1_uwick": run("V1_uwick"),
            "V2_bb": run("V2_bb"), "V3_pure": run("V3_pure")}


def main():
    print("HYBRID exit — snap the blow-off top (climax combo) on top of the trail\n", flush=True)
    acc = {var: {m: [] for m in MONTHS} for var in VARIANTS}

    for mname, (s, e) in MONTHS.items():
        start, end = parse(s), parse(e); ff = start - PB.WARMUP_BARS * FOUR
        raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
        ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
              for sym, df in raw.items()}
        times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
        sr = PB.build_stable_ratio(ff, end)
        sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
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
                if not bool(row["entry_signal"]):
                    continue
                price = float(row["close"]); atr = float(row["atr"]); adx = float(row["adx"])
                if atr / price < ATR_MIN or adx < ADX_MIN:
                    continue
                cands.append((sym, price, atr, float(row["vol_pit"])))
            cands.sort(key=lambda x: x[3], reverse=True)
            for sym, price, atr, _ in cands[:MAX_CONC - len(open_until)]:
                open_until[sym] = t + HOLD_H * 3600 * 1000
                df = HR.load_range(sym, "1h", t - WARM_H * 3600 * 1000,
                                   t + HOLD_H * 3600 * 1000)
                if df is None or len(df) < WARM_H:
                    continue
                r = simulate(df, t, price, atr)
                for var in VARIANTS:
                    acc[var][mname].append(r[var])
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    print(f"\n##### HYBRID EXIT vs BASELINE (6 months) #####")
    print(f"{'variant':<12}{'WR':>6}{'PF':>7}{'ret_ALL':>9}{'~/mo':>7}   per-month")
    print("-" * 86)
    for var in VARIANTS:
        alln = [x for m in MONTHS for x in acc[var][m]]
        a = np.array(alln); wins = a[a > 0]; losses = a[a <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
        ret = a.sum() * POS_W
        pm = " ".join(f"{m[2:]}:{np.array(acc[var][m]).sum()*POS_W:+.0f}" for m in MONTHS)
        print(f"{var:<12}{(a>0).mean()*100:>5.0f}%{pf:>7.2f}{ret:>8.0f}%{ret/6:>6.1f}%   {pm}", flush=True)
    print("\nالأساس = خروج EMA20 فقط. الصيغ = + قطف الذروة مبكّراً.")
    print("إن زاد ret/PF عن الأساس => قطف الذروة يربح فعلاً (ليس إحصاءً فقط).")
    print("إن نقص => الإنذارات الكاذبة (خروج مبكّر ثم يكمل الصعود) تأكل الحافة.")
    print("\nDONE_HYBRID.", flush=True)


if __name__ == "__main__":
    main()
