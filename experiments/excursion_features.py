#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does ADX (or other entry features) predict the pump size (MFE) / oscillation?

Same 1m excursion measurement as excursion_profile.py, but also records the
ENTRY features (adx, atr_ratio, ema_order, rsi, calm_long, signal_age), then
buckets the trades by each feature and reports the mean MFE / MAE / oscillation
per bucket. If high-ADX trades show systematically higher MFE, we condition the
exit (TP level / runner width) on ADX.

Reuses the 1m cache from excursion_profile.py, so it is fast on a warm cache.
Run:  python experiments/excursion_features.py
"""

from __future__ import annotations

import gc, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

MAX_CONC = 8
FOUR_H = 4 * 3600 * 1000
HOLD_H = 48
OUT = "results_excursion"
CACHE = f"{OUT}/excursions_feat.json"

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def excursion(symbol, et, ep):
    end = et + HOLD_H * 3600 * 1000
    df = HR.load_range(symbol, "1m", et + 1, end)
    if df is None or len(df) < 5:
        return None
    hi = df["high"].to_numpy(float); lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    mfe = (hi.max() / ep - 1) * 100
    mae = (lo.min() / ep - 1) * 100
    rets = np.abs(np.diff(cl) / cl[:-1]) * 100
    osc = float(np.nanmean(rets)) if len(rets) else 0.0
    return mfe, mae, osc


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(CACHE):
        rows = json.load(open(CACHE))
        print(f"Loaded {len(rows)} cached.", flush=True)
    else:
        rows = []
        for mname, (s, e) in MONTHS.items():
            start, end = parse(s), parse(e)
            ff = start - PB.WARMUP_BARS * FOUR_H
            raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
            ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
                  for sym, df in raw.items()}
            times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
            sr = PB.build_stable_ratio(ff, end)
            sblock = {t: (np.isfinite(v) and v > S.STABLE_RATIO_MAX) for t, v in sr.items()}
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
                    if bool(row["entry_signal"]):
                        cands.append((sym, float(row["close"]), float(row["vol_pit"]), row))
                cands.sort(key=lambda x: x[2], reverse=True)
                for sym, price, _, row in cands[:MAX_CONC - len(open_until)]:
                    ex = excursion(sym, t, price)
                    open_until[sym] = t + HOLD_H * 3600 * 1000
                    if ex:
                        rows.append(dict(
                            mfe=round(ex[0], 2), mae=round(ex[1], 2), osc=round(ex[2], 3),
                            adx=round(float(row["adx"]), 1),
                            atr_ratio=round(float(row["atr_ratio"]), 4),
                            ema_order=round(float(row["ema_order"]), 2),
                            rsi=round(float(row["rsi"]), 1),
                            calm=round(float(row["calm_long"]), 3),
                            age=int(row["signal_age"])))
            print(f"  {mname}: total {len(rows)}", flush=True)
            del ps, raw; gc.collect()
        json.dump(rows, open(CACHE, "w"))
        print(f"Saved {len(rows)}.", flush=True)

    if not rows:
        print("No data."); return
    arr = {k: np.array([r[k] for r in rows], float) for k in
           ("mfe", "mae", "osc", "adx", "atr_ratio", "ema_order", "rsi", "calm", "age")}

    def buckets(feat, edges):
        print(f"\n=== MFE / MAE / osc by {feat} ===")
        print(f"{'bucket':<14}{'n':>6}{'mean MFE':>11}{'mean MAE':>11}{'mean osc':>11}{'%peak>=5':>10}")
        print("-" * 63)
        v = arr[feat]
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = (v >= lo) & (v < hi)
            if m.sum() == 0:
                continue
            lbl = f"[{lo:g},{hi:g})"
            print(f"{lbl:<14}{int(m.sum()):>6}{arr['mfe'][m].mean():>10.1f}%"
                  f"{arr['mae'][m].mean():>10.1f}%{arr['osc'][m].mean():>10.3f}"
                  f"{(arr['mfe'][m]>=5).mean()*100:>9.0f}%")

    print(f"\n##### FEATURE -> EXCURSION ({len(rows)} trades) #####")
    buckets("adx", [25, 30, 35, 40, 50, 200])
    buckets("atr_ratio", [0, 0.02, 0.04, 0.06, 0.10, 1])
    buckets("ema_order", [0.74, 0.99, 1.01])
    buckets("rsi", [0, 45, 55, 65, 75])
    buckets("age", [1, 2, 3, 5, 1000])
    # simple correlations with MFE
    print("\n=== correlation of each feature with MFE ===")
    for k in ("adx", "atr_ratio", "ema_order", "rsi", "calm", "age"):
        c = np.corrcoef(arr[k], arr["mfe"])[0, 1]
        print(f"  corr({k:<10}, MFE) = {c:+.3f}")
    print("\nDONE_EXCURSION_FEAT.", flush=True)


if __name__ == "__main__":
    main()
