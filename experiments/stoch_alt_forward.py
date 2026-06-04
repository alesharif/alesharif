#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the user's stochastic MTF signal predict 'explosion' on ALT coins?

Fixes two fair critiques: (1) test on the small/mid ALT universe (the real
target), not heavy BTC; (2) measure EXIT-AGNOSTIC forward returns so a too-tight
exit doesn't bias the result. We ask the core question directly:

  When (monthly stoch %K>%D) AND (weekly %K crosses above %D),
  is the forward return over the next 4 / 12 weeks BIGGER than the coin's
  baseline forward return at random weeks?

If signal >> baseline -> the setup truly raises explosion odds (real edge).
If signal ~ baseline -> the 'ready to explode' feeling is illusion.
Uses our cached 4h universe, resampled to weekly/monthly. Run:
  python experiments/stoch_alt_forward.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402

FOUR = 4 * 3600 * 1000
START = "2023-01-01"; END = "2026-06-01"


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def resample(df, rule):
    g = df.set_index("dt")
    return pd.DataFrame({"h": g["high"].resample(rule).max(),
                         "l": g["low"].resample(rule).min(),
                         "c": g["close"].resample(rule).last()}).dropna()


def stoch(h, l, c, n=14, d=3):
    llv = pd.Series(l).rolling(n).min(); hhv = pd.Series(h).rolling(n).max()
    k = (100 * (c - llv) / (hhv - llv).replace(0, np.nan)).fillna(50.0)
    return k.to_numpy(), k.rolling(d).mean().fillna(50.0).to_numpy()


def main():
    print("STOCH-MTF forward-return test on ALT universe (exit-agnostic)\n", flush=True)
    t0, t1 = parse(START), parse(END)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), t0, t1, log=lambda *a: None)
    print(f"  universe: {len(raw)} symbols; resampling weekly/monthly...", flush=True)

    sig4, sig12, base4, base12 = [], [], [], []
    n_coins = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        wk = resample(df, "W"); mo = resample(df, "ME")
        if len(wk) < 30 or len(mo) < 16:
            continue
        n_coins += 1
        wk_k, wk_d = stoch(wk["h"].to_numpy(), wk["l"].to_numpy(), wk["c"].to_numpy())
        mo_k, mo_d = stoch(mo["h"].to_numpy(), mo["l"].to_numpy(), mo["c"].to_numpy())
        mo_bull = pd.Series(mo_k > mo_d, index=mo.index).shift(1)
        mbw = mo_bull.reindex(wk.index, method="ffill").fillna(False).to_numpy()
        c = wk["c"].to_numpy(); n = len(c)
        cu = np.zeros(n, bool)
        cu[1:] = (wk_k[1:] > wk_d[1:]) & (wk_k[:-1] <= wk_d[:-1])
        for i in range(20, n - 12):
            f4 = c[i + 4] / c[i] - 1
            f12 = c[i + 12] / c[i] - 1
            base4.append(f4); base12.append(f12)             # baseline: every week
            if cu[i] and mbw[i]:                              # the user's signal
                sig4.append(f4); sig12.append(f12)
        del wk, mo
    del raw; gc.collect()

    def rep(name, a):
        a = np.array(a) * 100
        print(f"{name:<22}{len(a):>8}{a.mean():>+9.1f}%{np.median(a):>+9.1f}%"
              f"{(a > 0).mean()*100:>7.0f}%{(a > 20).mean()*100:>8.0f}%")

    print(f"\n##### {n_coins} alt coins #####")
    print(f"{'set':<22}{'n':>8}{'mean':>10}{'median':>9}{'win%':>7}{'>+20%':>8}")
    print("-" * 65)
    print("--- forward 4 weeks ---")
    rep("SIGNAL (4w)", sig4); rep("baseline (4w)", base4)
    print("--- forward 12 weeks ---")
    rep("SIGNAL (12w)", sig12); rep("baseline (12w)", base12)
    print("\nإذا SIGNAL ≈ baseline => الإشارة لا ترفع احتمال الانفجار فوق الصدفة.")
    print("'>+20%' = نسبة الحالات التي حقّقت انفجاراً (+20%) خلال المدة.")
    print("\nDONE_ALTFWD.", flush=True)


if __name__ == "__main__":
    main()
