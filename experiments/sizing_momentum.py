#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MOMENTUM sizing: tilt capital toward the strongest-momentum entries.

entry_filter.py showed the edge lives in a handful of already-explosive entries
(run24h>30% avg +11.6%/trade vs +0.21% baseline; removing them => -37%). So the
lever is the OPPOSITE of filtering: size UP on strong-momentum entries.

Honest design: weights are NORMALIZED so mean weight == baseline POS_W (0.125),
i.e. SAME total capital deployed. Any gain therefore comes from ALLOCATION
(tilt), not from extra leverage. We also report RISK: worst month + how much of
the gross profit comes from just the top-3 trades (tilting by the same variable
that marks the fat tail can be a bet on one outlier — per-month + concentration
expose that).

Run:  python experiments/sizing_momentum.py
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
EMA_SPAN = 20; WARM_H = 60

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"), "2026-01": ("2026-01-01", "2026-02-01"),
          "2026-02": ("2026-02-01", "2026-03-01"), "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"), "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def analyze(df, t0, ep, atr):
    """Return (run24h, ext, baseline_ret%) or None."""
    tm = df["time"].to_numpy()
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    e = ema(c, EMA_SPAN)
    idx = np.where(tm >= t0)[0]
    if len(idx) == 0 or idx[0] < EMA_SPAN + 1:
        return None
    k = idx[0]
    ext = (c[k] - e[k]) / e[k] if e[k] > 0 else 0.0
    run24 = (c[k] / c[k-24] - 1) if k >= 24 else 0.0
    sl = ep - HARD_SL_ATR * atr
    ret = (c[idx[-1]] / ep - 1) * 100 - FEE
    for i in idx:
        if i < EMA_SPAN + 1:
            continue
        if l[i] <= sl:
            ret = (min(sl, c[i]) / ep - 1) * 100 - FEE; break
        if c[i] < e[i]:
            ret = (c[i] / ep - 1) * 100 - FEE; break
    return run24, ext, ret


# weighting schemes: name -> function(run24 array, ext array) -> raw weights
SCHEMES = {
    "equal":        lambda r, x: np.ones_like(r),
    "run_mod(3x)":  lambda r, x: 1 + 3.0 * np.clip(r, 0, None),
    "run_agg(8x)":  lambda r, x: 1 + 8.0 * np.clip(r, 0, None),
    "ext_mod(3x)":  lambda r, x: 1 + 3.0 * np.clip(x, 0, None),
    "ext_agg(8x)":  lambda r, x: 1 + 8.0 * np.clip(x, 0, None),
    "tier_run":     lambda r, x: np.where(r > 0.30, 3.0, np.where(r > 0.10, 2.0, 1.0)),
}


def main():
    print("MOMENTUM sizing — tilt capital to strongest entries (mean weight fixed)\n", flush=True)
    R = []; X = []; RET = []; MON = []
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
                df = HR.load_range(sym, "1h", t - WARM_H * 3600 * 1000, t + HOLD_H * 3600 * 1000)
                if df is None or len(df) < WARM_H:
                    continue
                res = analyze(df, t, price, atr)
                if res is None:
                    continue
                r, x, ret = res
                R.append(r); X.append(x); RET.append(ret); MON.append(mname)
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    R = np.array(R); X = np.array(X); RET = np.array(RET); MON = np.array(MON)
    n = len(RET)
    print(f"\n##### MOMENTUM SIZING — {n} trades, mean weight fixed = {POS_W} #####")
    print(f"{'scheme':<14}{'PF':>7}{'total':>8}{'~/mo':>7}{'worst_mo':>9}{'top3%':>7}   per-month")
    print("-" * 92)
    for name, fn in SCHEMES.items():
        w_raw = fn(R, X).astype(float)
        w = w_raw / w_raw.mean() * POS_W            # normalize: mean weight == POS_W
        contrib = RET * w                            # P&L contribution per trade (% of total acct)
        pos = (RET > 0)
        pf = (RET[pos] * w[pos]).sum() / -(RET[~pos] * w[~pos]).sum() if (~pos).any() else float("inf")
        total = contrib.sum()
        permo = {m: contrib[MON == m].sum() for m in MONTHS}
        worst = min(permo.values())
        # concentration: share of gross positive P&L from the top-3 contributing trades
        gp = contrib[contrib > 0]
        top3 = np.sort(gp)[-3:].sum() / gp.sum() * 100 if gp.sum() > 0 else 0
        pm = " ".join(f"{m[2:]}:{permo[m]:+.0f}" for m in MONTHS)
        print(f"{name:<14}{pf:>7.2f}{total:>+7.0f}%{total/6:>+6.1f}%{worst:>+8.0f}%"
              f"{top3:>6.0f}%   {pm}", flush=True)
    print("\nالأفضل = total/PF أعلى مع worst_mo غير أسوأ و top3% غير مرتفع جداً.")
    print("top3% عالٍ => العائد كلّه من 2-3 صفقات (هشّ، مراهنة على outlier).")
    print("\nDONE_SIZING.", flush=True)


if __name__ == "__main__":
    main()
