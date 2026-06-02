#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anatomy of pumps: what do explosions share, and how do they differ from
non-pumps? Reviews the 1..7 candles (4h) BEFORE each move.

For April 2025 (rich bull month) we label each (symbol, bar) as:
  PUMP    : forward max high over next 12 bars >= +30%
  NORMAL  : forward max < +10% (clearly didn't explode)
We then profile the 7 bars leading INTO the trigger bar and compare the two
groups' average fingerprints across many features, so we can see:
  * what pumps share (consistent pre-pump signature)
  * what separates pumps from look-alikes (discriminative features)

Also clusters pumps by their pre-pump shape to reveal pump TYPES.
Fast (4h only, no 1s). Run:  python experiments/anatomy_of_pumps.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FOUR_H = 4 * 3600 * 1000
LOOK = 7         # candles before the trigger to profile
FWD = 12         # forward horizon (~2 days)
PUMP = 30.0      # pump threshold %
NORMAL = 10.0    # non-pump ceiling %


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    start, end = parse("2025-04-01"), parse("2025-05-01")
    ff = start - PB.WARMUP_BARS * FOUR_H
    print("Loading + featurising April 2025 ...", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {s: S.compute_features(df.copy()).set_index("close_time") for s, df in raw.items()}
    print(f"  {len(ps)} symbols\n", flush=True)

    # collect samples: at each in-month bar with enough history+future, label
    pumps = []      # list of feature dicts
    normals = []
    pump_shapes = []  # 7-bar return shape for clustering
    for sym, df in ps.items():
        idx = list(df.index)
        c = df["close"].to_numpy(float); h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float); v = df["quote_av"].to_numpy(float)
        atr = df["atr"].to_numpy(float); rsi = df["rsi"].to_numpy(float)
        adx = df["adx"].to_numpy(float)
        e20 = ema(c, 20); e50 = ema(c, 50)
        vavg = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        n = len(c)
        for i in range(LOOK, n - FWD):
            t = idx[i]
            if not (start <= t <= end):
                continue
            fwd = (h[i + 1:i + 1 + FWD].max() / c[i] - 1) * 100
            label = None
            if fwd >= PUMP:
                label = "pump"
            elif fwd < NORMAL:
                label = "normal"
            else:
                continue
            # fingerprint of the trigger bar i (using info up to i, no look-ahead)
            rng = h[i] - l[i]
            feat = dict(
                ret_1=(c[i] / c[i - 1] - 1) * 100,                       # last bar return
                ret_3=(c[i] / c[i - 3] - 1) * 100,                       # 3-bar return
                ret_7=(c[i] / c[i - 7] - 1) * 100,                       # 7-bar run-up
                vol_ratio=(v[i] / vavg[i]) if vavg[i] > 0 else 0,        # volume surge
                atr_pct=atr[i] / c[i] * 100 if c[i] else 0,             # volatility
                rsi=rsi[i],
                adx=adx[i],
                above_e20=1.0 if c[i] > e20[i] else 0.0,
                above_e50=1.0 if c[i] > e50[i] else 0.0,
                close_pos=((c[i] - l[i]) / rng) if rng > 0 else 0.5,    # close in bar
                green_7=int(sum(c[i - k] > c[i - k - 1] for k in range(7))),  # green bars of last 7
            )
            if label == "pump":
                pumps.append(feat)
                # 7-bar normalized return shape (for clustering)
                shape = [(c[i - 6 + k] / c[i - 7] - 1) * 100 for k in range(7)]
                pump_shapes.append(shape)
            else:
                normals.append(feat)

    pdf = pd.DataFrame(pumps); ndf = pd.DataFrame(normals)
    print(f"Samples: pumps={len(pdf)}  normals={len(ndf)}\n", flush=True)

    # --- A) average fingerprint: pumps vs normals ---
    print("=== A) PRE-PUMP FINGERPRINT (mean of trigger bar) ===")
    print(f"{'feature':<12}{'PUMP':>10}{'NORMAL':>10}{'separation':>12}")
    print("-" * 44)
    for col in pdf.columns:
        pm = pdf[col].mean(); nm = ndf[col].mean()
        sd = (pdf[col].std() + ndf[col].std()) / 2 or 1
        sep = (pm - nm) / sd     # standardized difference (discriminative power)
        flag = "  <== strong" if abs(sep) > 0.5 else ""
        print(f"{col:<12}{pm:>10.2f}{nm:>10.2f}{sep:>11.2f}{flag}")

    # --- B) cluster pump shapes into types ---
    print("\n=== B) PUMP TYPES (k-means on 7-bar pre-pump shape, k=3) ===")
    X = np.array(pump_shapes)
    if len(X) >= 3:
        # tiny k-means (no sklearn dependency)
        rng = np.random.default_rng(0)
        cent = X[rng.choice(len(X), 3, replace=False)]
        for _ in range(25):
            d = np.linalg.norm(X[:, None, :] - cent[None, :, :], axis=2)
            lab = d.argmin(1)
            for k in range(3):
                if (lab == k).any():
                    cent[k] = X[lab == k].mean(0)
        for k in range(3):
            grp = X[lab == k]
            if len(grp) == 0:
                continue
            shape = grp.mean(0)
            print(f"  type {k+1} ({len(grp)} pumps): 7-bar run-up shape "
                  f"[{' '.join(f'{x:+.0f}' for x in shape)}]%")
        print("  (each number = cumulative % move at that bar before the trigger)")


if __name__ == "__main__":
    main()
