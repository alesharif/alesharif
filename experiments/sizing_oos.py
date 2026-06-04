#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OUT-OF-SAMPLE / robustness for momentum sizing.

momentum sizing (sizing_momentum.py) lifted total +11% -> +70-79%, but top-3
trades carried ~60% of the profit. With a MONOTONIC lever (more tilt => more
in-sample return) the honest question isn't parameter choice — it's whether the
benefit is BROAD or driven by a few trades/months. Three diagnostics:

  1) per-month: does tilt beat equal in MOST months, or just April/Jan?
  2) drop-outliers: remove the top 1/3/5 contributing trades — does the edge die?
  3) two interleaved folds (Dec,Feb,Apr) vs (Jan,Mar,May): is the edge in BOTH?

Trades are cached to results_topsig/trades_6mo.npz on first run (heavy backtest),
so re-runs are instant. Run:  python experiments/sizing_oos.py
"""

from __future__ import annotations

import gc, os, sys
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
CACHE = "results_topsig/trades_6mo.npz"

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"), "2026-01": ("2026-01-01", "2026-02-01"),
          "2026-02": ("2026-02-01", "2026-03-01"), "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"), "2026-05": ("2026-05-01", "2026-06-01")}

# True out-of-sample regime: spread across 2024 (unseen). Use:  ... sizing_oos.py 2024
if "2024" in sys.argv:
    CACHE = "results_topsig/trades_2024.npz"
    MONTHS = {"2024-02": ("2024-02-01", "2024-03-01"), "2024-04": ("2024-04-01", "2024-05-01"),
              "2024-06": ("2024-06-01", "2024-07-01"), "2024-08": ("2024-08-01", "2024-09-01"),
              "2024-10": ("2024-10-01", "2024-11-01"), "2024-12": ("2024-12-01", "2025-01-01")}
    FOLDS_2024 = True
else:
    FOLDS_2024 = False


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def analyze(df, t0, ep, atr):
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


def build_trades():
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
    os.makedirs("results_topsig", exist_ok=True)
    np.savez(CACHE, R=np.array(R), X=np.array(X), RET=np.array(RET), MON=np.array(MON))
    return np.array(R), np.array(X), np.array(RET), np.array(MON)


def weights(metric, k, mask=None):
    """w_raw=1+k*clip(metric,0), normalized so mean weight over `mask` == POS_W."""
    w = 1 + k * np.clip(metric, 0, None)
    sel = np.ones(len(w), bool) if mask is None else mask
    w = w / w[sel].mean() * POS_W
    return w


def main():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        R, X, RET, MON = z["R"], z["X"], z["RET"], z["MON"]
        print(f"loaded cached trades ({len(RET)})\n", flush=True)
    else:
        print("building trades (heavy, one-time)...\n", flush=True)
        R, X, RET, MON = build_trades()

    mlist = list(MONTHS.keys())
    LEVERS = {"equal": (None, 0), "run_k8": (R, 8.0), "ext_k8": (X, 8.0)}

    def total(metric, k, idx):
        w = weights(metric if metric is not None else R, k)   # global mean-norm
        c = RET * w
        return c[idx].sum()

    # ---- 1) per-month: tilt vs equal ----
    print("##### 1) PER-MONTH return (global mean-weight = POS_W) #####")
    print(f"{'month':<9}{'equal':>8}{'run_k8':>9}{'ext_k8':>9}   (Δ vs equal)")
    wins = {"run_k8": 0, "ext_k8": 0}
    for m in mlist:
        idx = (MON == m)
        eq = total(None, 0, idx); rk = total(R, 8, idx); xk = total(X, 8, idx)
        if rk > eq: wins["run_k8"] += 1
        if xk > eq: wins["ext_k8"] += 1
        print(f"{m[2:]:<9}{eq:>+7.0f}%{rk:>+8.0f}%{xk:>+8.0f}%   "
              f"run {rk-eq:+.0f}, ext {xk-eq:+.0f}")
    print(f"{'TOTAL':<9}{total(None,0,slice(None)):>+7.0f}%"
          f"{total(R,8,slice(None)):>+8.0f}%{total(X,8,slice(None)):>+8.0f}%")
    print(f"months tilt beats equal (of 6):  run_k8={wins['run_k8']}  ext_k8={wins['ext_k8']}")

    # ---- 2) drop top-N contributing trades ----
    print("\n##### 2) DROP top-N contributing trades — does the edge survive? #####")
    print(f"{'lever':<9}{'drop0':>8}{'drop1':>8}{'drop3':>8}{'drop5':>8}")
    for name, (metric, k) in LEVERS.items():
        w = weights(metric if metric is not None else R, k)
        contrib = RET * w
        order = np.argsort(-contrib)               # biggest winners first
        row = []
        for d in (0, 1, 3, 5):
            keep = np.ones(len(contrib), bool); keep[order[:d]] = False
            row.append(contrib[keep].sum())
        print(f"{name:<9}" + "".join(f"{v:>+7.0f}%" for v in row))
    print("إن انهار run/ext إلى ~equal بعد حذف 3-5 صفقات => العائد من outliers (هشّ).")

    # ---- 3) two interleaved folds (each locally mean-normalized) ----
    print("\n##### 3) TWO INTERLEAVED FOLDS — is the edge in BOTH halves? #####")
    folds = {f"A({','.join(m[5:] for m in mlist[0::2])})": mlist[0::2],
             f"B({','.join(m[5:] for m in mlist[1::2])})": mlist[1::2]}
    print(f"{'fold':<14}{'equal':>8}{'run_k8':>9}{'ext_k8':>9}")
    for fname, fms in folds.items():
        sel = np.isin(MON, fms)
        def tot_local(metric, k):
            w = weights(metric if metric is not None else R, k, mask=sel)
            return (RET[sel] * w[sel]).sum()
        print(f"{fname:<14}{tot_local(None,0):>+7.0f}%{tot_local(R,8):>+8.0f}%{tot_local(X,8):>+8.0f}%")
    print("تفوّق في النصفين => متانة؛ تفوّق في نصف أبريل فقط => حظّ ماضٍ.")
    print("\nDONE_SIZING_OOS.", flush=True)


if __name__ == "__main__":
    main()
