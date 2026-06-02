#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISCOVERY: what 15m signals at entry separate the BIG pumps from the duds?

For every taken trade (4h signal + fear 1.15 + atr>=5%+adx>=40), we measure
several 15m features AS OF the entry bar (causal, bars up to entry only):
  vel_45m  : price % change over last 3 x 15m bars
  vel_2h   : price % change over last 8 x 15m bars
  volsurge : last 15m volume / mean(prev 20)
  bbw_pct  : Bollinger bandwidth percentile vs last 50 (low = coiled / squeeze)
  rsi15    : 15m RSI(14)
Then we read the trade's eventual peak (MFE) from the 1m path (48h) and report,
per feature bucket, the mean MFE and the share that become BIG pumps (MFE>=20%).
This tells us which launch signal to build the catcher on.

Run:  python experiments/launch_discovery.py
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
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0
OUT = "results_launch_discovery"
CACHE = f"{OUT}/feats.json"

MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
          "2026-03": ("2026-03-01", "2026-04-01"),
          "2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def feats15(sym, t):
    """15m features as of entry bar t (causal)."""
    df = HR.load_range(sym, "15m", t - 30 * 3600 * 1000, t)
    if df is None or len(df) < 55:
        return None
    df = df[df["close_time"] <= t]
    if len(df) < 55:
        return None
    c = df["close"].to_numpy(float); v = df["volume"].to_numpy(float)
    vel45 = (c[-1] / c[-4] - 1) * 100
    vel2h = (c[-1] / c[-9] - 1) * 100
    volsurge = v[-1] / (v[-21:-1].mean() + 1e-12)
    ma = pd.Series(c).rolling(20).mean(); sd = pd.Series(c).rolling(20).std()
    bbw = ((4 * sd) / ma * 100).to_numpy()
    bbw_now = bbw[-1]
    bbw_pct = (np.nanmean(bbw[-50:] <= bbw_now)) * 100   # percentile rank of current width
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    rsi15 = 100 - 100 / (1 + ru[-1] / (rd[-1] + 1e-12))
    return dict(vel45=vel45, vel2h=vel2h, volsurge=volsurge, bbw_pct=bbw_pct, rsi15=rsi15)


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(CACHE):
        recs = json.load(open(CACHE))
        print(f"Loaded {len(recs)} cached.", flush=True)
    else:
        recs = []
        for mname, (s, e) in MONTHS.items():
            start, end = parse(s), parse(e)
            ff = start - PB.WARMUP_BARS * FOUR_H
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
                    cands.append((sym, price, float(row["vol_pit"])))
                cands.sort(key=lambda x: x[2], reverse=True)
                for sym, price, _ in cands[:MAX_CONC - len(open_until)]:
                    open_until[sym] = t + HOLD_H * 3600 * 1000
                    f = feats15(sym, t)
                    if f is None:
                        continue
                    d = HR.load_range(sym, "1m", t + 1, t + HOLD_H * 3600 * 1000)
                    if d is None or len(d) < 60:
                        continue
                    mfe = (d["high"].to_numpy(float).max() / price - 1) * 100
                    f["mfe"] = round(mfe, 2)
                    recs.append(f)
            print(f"  {mname}: {len(recs)} trades", flush=True)
            del ps, raw; gc.collect()
        json.dump(recs, open(CACHE, "w"))
        print(f"Saved {len(recs)}.", flush=True)

    if not recs:
        print("no data"); return
    A = {k: np.array([r[k] for r in recs], float) for k in
         ("vel45", "vel2h", "volsurge", "bbw_pct", "rsi15", "mfe")}
    mfe = A["mfe"]
    print(f"\n##### LAUNCH DISCOVERY ({len(recs)} trades) — what precedes big pumps? #####")
    print(f"baseline: mean MFE {mfe.mean():.1f}% | big-pump(>=20%) share {(mfe>=20).mean()*100:.0f}% "
          f"| dud(<3%) share {(mfe<3).mean()*100:.0f}%")

    def buckets(feat, edges, fmt="g"):
        v = A[feat]
        print(f"\n=== {feat} ===")
        print(f"{'bucket':<16}{'n':>6}{'mean MFE':>11}{'%big(>=20)':>12}{'%dud(<3)':>11}")
        for i in range(len(edges) - 1):
            m = (v >= edges[i]) & (v < edges[i + 1])
            if m.sum() < 5:
                continue
            lbl = f"[{edges[i]:{fmt}},{edges[i+1]:{fmt}})"
            print(f"{lbl:<16}{int(m.sum()):>6}{mfe[m].mean():>10.1f}%"
                  f"{(mfe[m]>=20).mean()*100:>11.0f}%{(mfe[m]<3).mean()*100:>10.0f}%")

    buckets("vel45", [-50, 0, 3, 6, 12, 200])
    buckets("vel2h", [-50, 0, 5, 10, 20, 200])
    buckets("volsurge", [0, 1, 2, 4, 8, 1e9])
    buckets("bbw_pct", [0, 20, 40, 60, 80, 101])
    buckets("rsi15", [0, 45, 55, 65, 75, 101])
    print("\n=== correlation with MFE ===")
    for k in ("vel45", "vel2h", "volsurge", "bbw_pct", "rsi15"):
        print(f"  corr({k:<9}, MFE) = {np.corrcoef(A[k], mfe)[0,1]:+.3f}")
    print("\nDONE_LAUNCH_DISCOVERY.", flush=True)


if __name__ == "__main__":
    main()
