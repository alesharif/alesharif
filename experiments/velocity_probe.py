#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do pumps start with a visible VELOCITY burst on 15m candles?

Tests the user's idea: a fast/sudden pump is invisible on a 4h candle until
it's over, but on 15m we can measure short-term price velocity and catch the
launch early.

For April 2025: for every coin, find its big pumps (15m forward max over next
~16 bars = 4h >= +20%). Then look at the 15m bars AT/just-before the launch and
measure velocity = % price change over the last K 15m bars (K=1,2,4 = 15/30/60
min) and the volume burst. Compare pump-launch bars vs random non-pump bars to
see if velocity is a strong, early discriminator.

Pure 15m analysis (fast). Run:  python experiments/velocity_probe.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

FIFTEEN = 15 * 60 * 1000
FWD = 16          # 16 x 15m = 4 hours forward
PUMP = 20.0       # a pump = +20% within next 4h (measured on 15m)


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def main():
    start, end = parse("2025-04-01"), parse("2025-05-01")
    # warm-up: a few days of 15m before for rolling stats
    warm = start - 5 * 24 * 3600 * 1000
    print("Listing symbols ...", flush=True)
    syms = PIT.list_all_usdt_symbols()
    print(f"  {len(syms)} symbols; loading 15m (this is lighter than 1s) ...", flush=True)

    pump_v = {1: [], 2: [], 4: []}     # velocity at launch for each lookback K
    pump_vol = []
    norm_v = {1: [], 2: [], 4: []}
    norm_vol = []
    done = 0
    for sym in syms:
        df = HR.load_range(sym, "15m", warm, end)
        done += 1
        if done % 80 == 0:
            print(f"  scanned {done}/{len(syms)}", flush=True)
        if df is None or len(df) < 60:
            continue
        c = df["close"].to_numpy(float); h = df["high"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        t = df["time"].to_numpy()
        vavg = pd.Series(v).rolling(40).mean().shift(1).to_numpy()  # ~10h avg
        n = len(c)
        for i in range(8, n - FWD):
            if not (start <= t[i] <= end):
                continue
            fwd = (h[i + 1:i + 1 + FWD].max() / c[i] - 1) * 100
            is_pump = fwd >= PUMP
            is_norm = fwd < 5.0
            if not (is_pump or is_norm):
                continue
            vel = {K: (c[i] / c[i - K] - 1) * 100 for K in (1, 2, 4)}
            volr = v[i] / vavg[i] if vavg[i] > 0 else 0
            if is_pump:
                for K in (1, 2, 4):
                    pump_v[K].append(vel[K])
                pump_vol.append(volr)
            else:
                for K in (1, 2, 4):
                    norm_v[K].append(vel[K])
                norm_vol.append(volr)

    print(f"\nSamples: pump-launch bars={len(pump_vol)}  normal bars={len(norm_vol)}\n")
    print("=== VELOCITY at launch (15m): pump vs normal ===")
    print(f"{'metric':<22}{'PUMP(med)':>11}{'NORMAL(med)':>13}{'separation':>12}")
    print("-" * 58)

    def report(name, pa, na):
        pa = np.array(pa); na = np.array(na)
        pm = np.median(pa); nm = np.median(na)
        sd = (pa.std() + na.std()) / 2 or 1
        sep = (np.mean(pa) - np.mean(na)) / sd
        flag = "  <== strong" if abs(sep) > 0.4 else ""
        print(f"{name:<22}{pm:>10.2f}%{nm:>12.2f}%{sep:>11.2f}{flag}")

    report("velocity 15min (K=1)", pump_v[1], norm_v[1])
    report("velocity 30min (K=2)", pump_v[2], norm_v[2])
    report("velocity 60min (K=4)", pump_v[4], norm_v[4])
    report("volume burst (x avg)", pump_vol, norm_vol)

    # how many pump launches had a clear early velocity > threshold?
    print("\n=== EARLY-CATCH potential: pump launches with velocity above X ===")
    for K in (2, 4):
        pa = np.array(pump_v[K]); na = np.array(norm_v[K])
        for thr in (3, 5, 8):
            p_hit = (pa >= thr).mean() * 100
            n_hit = (na >= thr).mean() * 100
            print(f"  K={K*15}min vel>={thr}% : catches {p_hit:.0f}% of pumps, "
                  f"false on {n_hit:.0f}% of normals")


if __name__ == "__main__":
    main()
