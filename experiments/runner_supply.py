#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WHAT CHANGED IN 2025? Reverse diagnostic: measure the market's RUNNER SUPPLY
over time — the % of liquid coins that achieved a +100% (and +50%) move within a
forward 90-day window, per month. The strategy feeds on runners; if the supply
collapsed in 2025, that explains the ~17-month losing streak (a market regime
change: alt-runner era -> BTC-dominance drought). Run: python experiments/runner_supply.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

S, E = "2022-06-01", "2026-06-01"
LIQ_MIN = 300_000; FWD = 90        # forward window (days) = strategy hold


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)


def main():
    print("RUNNER SUPPLY over time: % of liquid coins that run +50% / +100% in 90d\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    mbs = pd.date_range(pd.Timestamp("2022-09-01", tz="UTC"), pd.Timestamp(E, tz="UTC"), freq="MS")
    mk = [x.strftime("%Y-%m") for x in mbs]; mt = [int(x.timestamp()*1000) for x in mbs]
    r50 = {k: 0 for k in mk}; r100 = {k: 0 for k in mk}; tot = {k: 0 for k in mk}

    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        d = pd.DataFrame({"h": g["high"].resample("D").max(), "c": g["close"].resample("D").last(),
                          "v": g["volume"].resample("D").sum(), "t": g["time"].resample("D").last()}).dropna()
        c = d["c"].to_numpy(); h = d["h"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
        n = len(c); dv = c*v
        if n < 60:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        for k, mbt in zip(mk, mt):
            i = np.searchsorted(t, mbt, side="left")
            if i < 30 or i >= n - 5:
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            tot[k] += 1
            fwd_max = h[i:min(i+FWD, n)].max() / c[i] - 1
            if fwd_max >= 0.50: r50[k] += 1
            if fwd_max >= 1.00: r100[k] += 1
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    print(f"{'month':<9}{'liquid':>7}{'run+50%':>9}{'run+100%':>10}")
    yr = {}
    for k in mk:
        if tot[k] == 0:
            continue
        p50 = r50[k]/tot[k]*100; p100 = r100[k]/tot[k]*100
        y = k[:4]; yr.setdefault(y, []).append((p50, p100))
        bar = "#" * int(p100/2)
        print(f"{k:<9}{tot[k]:>7}{p50:>7.0f}%{p100:>8.0f}%  {bar}")
    print(f"\n{'YEAR':<9}{'avg +50%':>10}{'avg +100%':>11}")
    for y in sorted(yr):
        a = np.array(yr[y])
        print(f"{y:<9}{a[:,0].mean():>9.0f}%{a[:,1].mean():>10.0f}%")
    print("\nإن انهار 'run+100%' في 2025 => السوق توقّف عن إنتاج الرابضين = سبب الخسارة المستمرّة.")
    print("\nDONE_SUPPLY.", flush=True)


if __name__ == "__main__":
    main()
