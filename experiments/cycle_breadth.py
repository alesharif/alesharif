#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User's market-cycle idea via EXPLOSIVE breadth = the runner-production rate.

Daily across all liquid coins: % whose trailing-30-day return >= +50% (i.e. the
market is currently PRODUCING runners). This is the cycle gauge. We test:
  (1) does it separate the developed system's WINNING vs LOSING months?
  (2) is it PERSISTENT (autocorrelated) -> a cycle you can catch, not random?
  (3) does gating trades on it (breadth high) improve expectancy AND consistency?
In-sample. Lagged 1 day at entry (no lookahead). Run:  python experiments/cycle_breadth.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

SL = 0.10; TP = 1.0; WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
DAY = 86400000


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Cycle gauge = explosive breadth (% coins +50% in 30d). Predict losing months?\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)

    run_cnt = {}; tot_cnt = {}          # day_ms -> counts
    trades = []                          # (entry_t, net_ret, month)
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        d = pd.DataFrame({"o": g["open"].resample("D").first(), "h": g["high"].resample("D").max(),
                          "l": g["low"].resample("D").min(), "c": g["close"].resample("D").last(),
                          "v": g["volume"].resample("D").sum(), "t": g["time"].resample("D").last()}).dropna()
        c = d["c"].to_numpy(); n = len(c)
        if n < 220:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        # explosive-breadth contribution: per day, is this coin a current runner?
        for i in range(30, n):
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            day = int(t[i]) // DAY
            tot_cnt[day] = tot_cnt.get(day, 0) + 1
            if c[i] / c[i-30] - 1 >= 0.50:
                run_cnt[day] = run_cnt.get(day, 0) + 1
        # developed-system trades
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float)
        L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n):
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and int(t[i]) >= ms(SF) and c[i] > e200[i]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            ent_t, ent_px = int(t[i]), float(c[i]); j0 = np.searchsorted(T, ent_t)
            if j0 >= len(T):
                continue
            sl = ent_px*(1-SL); tp = ent_px*(1+TP); end_t = ent_t+WINDOW_MS
            r = None; j = j0
            while j < len(T) and T[j] <= end_t:
                if L[j] <= sl: r = (-SL*100, True); break
                if H[j] >= tp: r = (TP*100, False); break
                j += 1
            if r is None:
                r = ((C[min(j, len(T)-1)]/ent_px-1)*100, False)
            nr = r[0] - 1.0 - (1.0 if r[1] else 0.0)
            trades.append((ent_t, nr, pd.Timestamp(ent_t, unit="ms").strftime("%Y-%m")))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    days = sorted(tot_cnt)
    bday = np.array([run_cnt.get(dd, 0)/tot_cnt[dd]*100 for dd in days])
    dms = np.array([dd*DAY for dd in days])
    bser = pd.Series(bday, index=pd.to_datetime(dms, unit="ms"))

    def breadth_at(t):                  # lagged: yesterday's value
        idx = np.searchsorted(dms, t, side="left") - 1
        return bday[idx] if idx >= 0 else np.nan

    # (1) win vs lose months
    mo = {}
    for et, nr, mk in trades:
        mo.setdefault(mk, []).append((nr, breadth_at(et)))
    win_b, los_b = [], []
    print(f"{'month':<9}{'win?':>6}{'mean%':>8}{'breadth':>9}")
    for mk in sorted(mo):
        nets = [x[0] for x in mo[mk]]; brs = [x[1] for x in mo[mk] if np.isfinite(x[1])]
        mean = np.mean(nets); br = np.mean(brs) if brs else float("nan")
        (win_b if mean > 0 else los_b).append(br)
        print(f"{mk:<9}{('WIN' if mean>0 else 'lose'):>6}{mean:>+7.1f}%{br:>8.0f}%")
    print(f"\nexplosive-breadth  WIN months: {np.nanmean(win_b):.0f}%   "
          f"LOSE months: {np.nanmean(los_b):.0f}%   gap: {np.nanmean(win_b)-np.nanmean(los_b):+.0f}%")

    # (2) persistence: monthly breadth autocorrelation (lag 1)
    mser = bser.resample("ME").mean().dropna()
    ac = mser.autocorr(lag=1)
    print(f"monthly breadth autocorrelation (lag1): {ac:+.2f}  "
          f"(>0.3 => persistent/cyclical => catchable)")

    # (3) gate trades on breadth threshold
    allb = np.array([breadth_at(et) for et, _, _ in trades])
    print(f"\n{'gate':<22}{'n':>7}{'exp':>9}{'months+':>10}{'cons%':>7}")
    for thr in [0, np.nanmedian(allb), np.nanpercentile(allb, 70)]:
        sel = [(nr, mk) for (et, nr, mk), b in zip(trades, allb) if np.isfinite(b) and b >= thr]
        if not sel:
            continue
        rets = np.array([x[0] for x in sel]); m2 = {}
        for nr, mk in sel:
            m2.setdefault(mk, []).append(nr)
        posm = sum(1 for k in m2 if np.mean(m2[k]) > 0)
        name = "no gate" if thr == 0 else f"breadth>={thr:.0f}%"
        print(f"{name:<22}{len(rets):>7}{rets.mean():>+8.1f}%{posm:>6}/{len(m2):<3}{posm/len(m2)*100:>6.0f}%")
    print("\nإن كان gap كبيراً + autocorr>0.3 + البوّابة ترفع cons% => فكرتك صحيحة والدورة قابلة للّحاق.")
    print("\nDONE_CYCLE.", flush=True)


if __name__ == "__main__":
    main()
