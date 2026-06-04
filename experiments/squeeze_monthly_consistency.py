#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Option #1: monthly CONSISTENCY of the developed squeeze system + a MONTHLY
trend filter, tested with and without the 99-day filter.

Developed system: squeeze breakout + daily above-EMA200 + SL10 + TP+100% + liquid.
We answer:
  (A) is the edge spread across months, or driven by a few? (per-month table)
  (B) does adding a MONTHLY trend filter (monthly close > 6-month SMA) help,
      tested once WITH the 99-day-downtrend filter and once WITHOUT it?
In-sample 2022-2026 (longest window for monthly stats). Net @1% RT + 1% stop slip.
Run:  python experiments/squeeze_monthly_consistency.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

SL = 0.10; TP = 1.0; WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def resample(df, rule):
    g = df.set_index("dt")
    return pd.DataFrame({"o": g["open"].resample(rule).first(), "h": g["high"].resample(rule).max(),
                         "l": g["low"].resample(rule).min(), "c": g["close"].resample(rule).last(),
                         "v": g["volume"].resample(rule).sum(), "t": g["time"].resample(rule).last()}).dropna()


def main():
    print("Monthly consistency + monthly trend filter (with/without 99d)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    # each trade: (month_key, net_ret, pass99, pass_monthly)
    trades = []
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        d = resample(df, "D"); mo = resample(df, "ME")
        c = d["c"].to_numpy(); n = len(c)
        if n < 220 or len(mo) < 8:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        dv = c*v
        # monthly trend gate: monthly close > SMA(monthly close,6), lagged
        mc = mo["c"].to_numpy(); mt = mo["t"].to_numpy()
        msma = pd.Series(mc).rolling(6).mean().to_numpy()
        mbull = np.array([(mc[i] > msma[i]) if np.isfinite(msma[i]) else False for i in range(len(mc))])
        mbull_times = mt  # month-close times aligned with mbull (use last completed month)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float)
        L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n):
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and int(t[i]) >= ms(SF) and c[i] > e200[i]):       # developed: above daily EMA200
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:               # liquid only
                continue
            ent_t, ent_px = int(t[i]), float(c[i])
            j0 = np.searchsorted(T, ent_t)
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
            pass99 = c[i] >= c[i-99]
            p = np.searchsorted(mbull_times, ent_t, side="right") - 1   # last completed month
            pass_m = bool(mbull[p]) if p >= 0 else False
            mk = pd.Timestamp(ent_t, unit="ms").strftime("%Y-%m")
            trades.append((mk, nr, pass99, pass_m))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    arr = trades
    print(f"developed system: {len(arr)} trades\n", flush=True)

    # (A) per-month consistency (base developed system)
    months = {}
    for mk, nr, _, _ in arr:
        months.setdefault(mk, []).append(nr)
    keys = sorted(months)
    pos = sum(1 for k in keys if np.mean(months[k]) > 0)
    print(f"##### (A) PER-MONTH consistency: {pos}/{len(keys)} months positive #####")
    for k in keys:
        a = np.array(months[k]); bar = "#" * min(40, int(abs(a.mean())))
        print(f"  {k}  n={len(a):>3}  exp={a.mean():>+6.1f}%  {bar}")

    # (B) monthly filter x 99d combos — now WITH per-month consistency
    def expe(p99, pm):
        sel = [(mk, nr) for mk, nr, x99, xm in arr if (xm if pm else True) and (x99 if p99 else True)]
        if not sel:
            return float("nan"), 0, 0, 0
        mo = {}
        for mk, nr in sel:
            mo.setdefault(mk, []).append(nr)
        posm = sum(1 for k in mo if np.mean(mo[k]) > 0)
        return np.mean([r for _, r in sel]), len(sel), posm, len(mo)
    print(f"\n##### (B) MONTHLY filter x 99-day filter (+ consistency) #####")
    print(f"{'config':<26}{'n':>7}{'exp':>9}{'months+':>10}{'cons%':>7}")
    for pm, p99, name in [(False, False, "base (developed)"),
                          (False, True, "+ 99d only"),
                          (True, False, "+ monthly only"),
                          (True, True, "+ monthly + 99d")]:
        m, nn, posm, tot = expe(p99, pm)
        cons = posm/tot*100 if tot else float("nan")
        print(f"{name:<26}{nn:>7}{m:>+8.1f}%{posm:>6}/{tot:<3}{cons:>6.0f}%")
    print("\nمتفرّقة عبر الأشهر => حافة حقيقية. مركّزة في قلّة => هشّة.")
    print("\nDONE_CONS.", flush=True)


if __name__ == "__main__":
    main()
