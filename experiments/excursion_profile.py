#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Excursion profile (MFE / MAE) of the strategy's trades at 1-MINUTE resolution.

For every trade the bot actually takes (4h entry_signal + stable-ratio gate,
rank vol, 8 concurrent), we walk forward on 1m candles for a fixed window and
measure, relative to the entry price:
  * MFE  = maximum favorable excursion  = max(high)/entry - 1   (the real peak)
  * MAE  = maximum adverse excursion     = min(low)/entry  - 1   (the worst dip)
  * t_MFE = hours from entry to the peak
  * osc  = mean 1m absolute return (%)   = the noise level

Then we print the DISTRIBUTIONS (percentiles). These tell us:
  - how high pumps actually reach (MFE distribution -> where to place take-profit
    LIMIT orders, which fill legitimately on the way up)
  - how much heat trades take before running (MAE -> stop placement)
  - the 1m noise (why a tight trailing stop gets whipsawed)

Window: HOLD_H hours forward per trade. 4 months. 1m data (lighter than 1s),
cached; disk_guard recommended. Run:  python experiments/excursion_profile.py
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
HOLD_H = 48                       # forward window per trade (hours)
OUT = "results_excursion"

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def excursion(symbol, et, ep):
    """Return (mfe%, mae%, t_mfe_h, mean_abs_1m_ret%) over HOLD_H hours, or None."""
    end = et + HOLD_H * 3600 * 1000
    df = HR.load_range(symbol, "1m", et + 1, end)
    if df is None or len(df) < 5:
        return None
    hi = df["high"].to_numpy(float); lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float); tm = df["time"].to_numpy()
    mfe = (hi.max() / ep - 1) * 100
    mae = (lo.min() / ep - 1) * 100
    i_peak = int(np.argmax(hi))
    t_mfe = (int(tm[i_peak]) - et) / 3600000.0
    rets = np.abs(np.diff(cl) / cl[:-1]) * 100
    osc = float(np.nanmean(rets)) if len(rets) else 0.0
    return mfe, mae, t_mfe, osc


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = f"{OUT}/excursions.json"
    if os.path.exists(cache):
        rows = json.load(open(cache))
        print(f"Loaded {len(rows)} cached excursions.", flush=True)
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
            # reconstruct the taken trades (8 concurrent, rank vol, stable gate)
            open_until = {}; taken = 0
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
                        cands.append((sym, float(row["close"]), float(row["vol_pit"])))
                cands.sort(key=lambda x: x[2], reverse=True)
                for sym, price, _ in cands[:MAX_CONC - len(open_until)]:
                    ex = excursion(sym, t, price)
                    open_until[sym] = t + HOLD_H * 3600 * 1000   # block ~ window
                    if ex:
                        rows.append(dict(month=mname, sym=sym, mfe=round(ex[0], 2),
                                         mae=round(ex[1], 2), t_mfe=round(ex[2], 1),
                                         osc=round(ex[3], 3)))
                        taken += 1
            print(f"  {mname}: {taken} trades profiled", flush=True)
            del ps, raw; gc.collect()
        json.dump(rows, open(cache, "w"))
        print(f"Saved {len(rows)} excursions.", flush=True)

    if not rows:
        print("No data."); return
    mfe = np.array([r["mfe"] for r in rows]); mae = np.array([r["mae"] for r in rows])
    tmfe = np.array([r["t_mfe"] for r in rows]); osc = np.array([r["osc"] for r in rows])
    pct = [10, 25, 50, 75, 90, 95]
    print(f"\n===== EXCURSION PROFILE ({len(rows)} trades, {HOLD_H}h window, 1m data) =====\n")
    print(f"{'percentile':<12}{'MFE%(peak)':>12}{'MAE%(dip)':>12}{'t_MFE(h)':>11}")
    print("-" * 47)
    for p in pct:
        print(f"{('p'+str(p)):<12}{np.percentile(mfe,p):>11.1f}%{np.percentile(mae,p):>11.1f}%"
              f"{np.percentile(tmfe,p):>10.1f}")
    print("-" * 47)
    print(f"{'mean':<12}{mfe.mean():>11.1f}%{mae.mean():>11.1f}%{tmfe.mean():>10.1f}")
    print(f"\nmean 1m oscillation: {osc.mean():.3f}% per minute")
    # how many trades reach common TP levels
    print("\n=== share of trades whose PEAK reaches at least X% (TP feasibility) ===")
    for lvl in (1, 2, 3, 5, 8, 12, 20):
        print(f"  peak >= +{lvl:>2}% : {(mfe >= lvl).mean()*100:5.1f}% of trades")
    print("\n=== how deep trades dip first (MAE) ===")
    for lvl in (-1, -2, -3, -5, -8):
        print(f"  dipped <= {lvl:>3}% : {(mae <= lvl).mean()*100:5.1f}% of trades")
    print("\nDONE_EXCURSION.", flush=True)


if __name__ == "__main__":
    main()
