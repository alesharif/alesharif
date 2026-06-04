#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ENTRY filter: are 'already-exhausted' entries (late, near a top) net losers?

Different use of the validated top-marks: not to EXIT (that clipped the fat tail
— see hybrid_exit.py), but to AVOID ENTERING a coin that is ALREADY stretched /
already had a volume climax at entry time. Our entry signal needs ADX>40, so we
buy coins already moving — the question is whether the ones that are ALREADY
overextended/climaxed at entry are late (exhausted) and worse than fresh ones.

Method (clean, no refill complexity): take the baseline trades, compute each
trade's STATE AT ENTRY (from 1h candles up to entry) + its baseline-exit return
(1h close<EMA20 + 2*ATR stop). For each filter, split trades into REMOVED (state
true) vs KEPT, and report the mean/PF of each. A good filter => REMOVED trades
have clearly negative expectancy and KEPT trades have higher PF/return.

Run:  python experiments/entry_filter.py
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


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def analyze(df, t0, ep, atr):
    """Return (entry_state dict, baseline_return%) or None."""
    tm = df["time"].to_numpy()
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    e = ema(c, EMA_SPAN); rs = rsi(c)
    volma = pd.Series(v).rolling(20).mean().to_numpy()
    sma20 = pd.Series(c).rolling(20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20).std().to_numpy()
    idx = np.where(tm >= t0)[0]
    if len(idx) == 0 or idx[0] < EMA_SPAN + 1:
        return None
    k = idx[0]                                   # entry candle index
    # ---- state AT ENTRY (using info up to & including entry candle) ----
    ext = (c[k] - e[k]) / e[k] if e[k] > 0 else 0.0
    r = rs[k]
    clim6 = bool((v[max(0, k-5):k+1] > 3 * volma[max(0, k-5):k+1]).any())
    run24 = (c[k] / c[k-24] - 1) if k >= 24 else (c[k] / c[idx[0]-1] - 1 if idx[0] >= 1 else 0.0)
    bb_above = bool(c[k] > (sma20[k] + 2 * std20[k])) if np.isfinite(std20[k]) else False
    state = {"ext>15%": ext > 0.15, "ext>25%": ext > 0.25, "rsi>70": r > 70,
             "rsi>75": r > 75, "climax<=6h": clim6, "run24h>30%": run24 > 0.30,
             "bb_above": bb_above}
    # ---- baseline exit return (EMA20 close-break + 2*ATR stop) ----
    sl = ep - HARD_SL_ATR * atr
    ret = (c[idx[-1]] / ep - 1) * 100 - FEE
    for i in idx:
        if i < EMA_SPAN + 1:
            continue
        if l[i] <= sl:
            ret = (min(sl, c[i]) / ep - 1) * 100 - FEE; break
        if c[i] < e[i]:
            ret = (c[i] / ep - 1) * 100 - FEE; break
    return state, ret


def main():
    print("ENTRY filter — are already-exhausted entries net losers?\n", flush=True)
    states = []; rets = []; mons = []
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
                st, ret = res
                states.append(st); rets.append(ret); mons.append(mname)
        print(f"  {mname} done", flush=True)
        del ps, raw; gc.collect()

    rets = np.array(rets); mons = np.array(mons)
    n = len(rets)
    base_pf = rets[rets > 0].sum() / -rets[rets <= 0].sum() if (rets <= 0).any() else float("inf")
    print(f"\n##### ENTRY-FILTER analysis — {n} trades #####")
    print(f"baseline: mean {rets.mean():+.2f}%/trade  PF {base_pf:.2f}  "
          f"total {rets.sum()*POS_W:+.0f}%  WR {(rets>0).mean()*100:.0f}%\n")
    keys = list(states[0].keys())
    print(f"{'filter (skip if true)':<22}{'n_rm':>6}{'rm_mean':>9}{'rm_PF':>7} | "
          f"{'n_keep':>7}{'kp_mean':>9}{'kp_PF':>7}{'kp_total':>9}")
    print("-" * 88)
    for key in keys:
        mask = np.array([s[key] for s in states], dtype=bool)
        rm = rets[mask]; kp = rets[~mask]
        if len(rm) == 0 or len(kp) == 0:
            print(f"{key:<22}{len(rm):>6}   (degenerate split)"); continue
        rm_pf = rm[rm > 0].sum() / -rm[rm <= 0].sum() if (rm <= 0).any() else float("inf")
        kp_pf = kp[kp > 0].sum() / -kp[kp <= 0].sum() if (kp <= 0).any() else float("inf")
        print(f"{key:<22}{len(rm):>6}{rm.mean():>+8.2f}%{rm_pf:>7.2f} | "
              f"{len(kp):>7}{kp.mean():>+8.2f}%{kp_pf:>7.2f}{kp.sum()*POS_W:>+8.0f}%")
    print("\nفلتر جيّد = rm_mean سالب بوضوح (المُستبعَدة خاسرة) و kp_PF أعلى من الأساس.")
    print("احذر: لو kp_total أقل بكثير => الفلتر يقصّ صفقات رابحة أيضاً (ذيل سمين).")
    print("\nDONE_ENTRYFILTER.", flush=True)


if __name__ == "__main__":
    main()
