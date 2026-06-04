#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAITHFUL test of the user's ACTUAL mechanic: signal + TP +50% (first-touch).

The previous tests measured close-to-close returns — NOT how the user trades.
He enters on his monthly stochastic deep-oversold signal and takes profit when
price TOUCHES +50% at any moment (first-touch), claiming ~90% hit rate. That can
be true AND still unprofitable — it depends on what happens to the trades that
DON'T hit +50% (the stop / the bleeders / the still-open underwater ones).

So we simulate it exactly, on intraday (4h) data, first-touch:
  entry  = first 4h close after the signal month-end
  TP     = +50%  (win)
  stop   = none / -50% / -30%   (three scenarios)
  window = up to 9 months; if neither touched -> close at last price (mark-to-mkt)
Report: real TP-hit%, stop%, timeout%, AND full expectancy net of fees.
Run:  python experiments/tp50_test.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

START = "2022-06-01"; END = "2026-06-01"
DS = int(pd.Timestamp(START, tz="UTC").timestamp()*1000)
DE = int(pd.Timestamp(END, tz="UTC").timestamp()*1000)
TP = 0.50
WINDOW_MS = 270 * 24 * 3600 * 1000          # 9 months max hold
FEE = 0.2                                    # round-trip %
STOPS = {"no stop": None, "stop -50%": 0.50, "stop -30%": 0.30}


def resample_m(df):
    g = df.set_index("dt")
    return pd.DataFrame({"h": g["high"].resample("ME").max(),
                         "l": g["low"].resample("ME").min(),
                         "c": g["close"].resample("ME").last(),
                         "t": g["time"].resample("ME").last()}).dropna()


def stoch1433(h, l, c):
    ll = pd.Series(l).rolling(14).min(); hh = pd.Series(h).rolling(14).max()
    rawk = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    k = rawk.rolling(3).mean(); d = k.rolling(3).mean()
    return k.to_numpy(), d.to_numpy()


def simulate(stop):
    raw = SHARED["raw"]
    res = []  # (outcome_str, return_pct, days)
    for sym, df in raw.items():
        t = df["time"].to_numpy(); H = df["high"].to_numpy(float)
        L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for ent_t, ent_px in SHARED["sigs"].get(sym, []):
            j0 = np.searchsorted(t, ent_t)
            if j0 >= len(t):
                continue
            tp_px = ent_px * (1 + TP)
            sl_px = ent_px * (1 - stop) if stop else None
            end_t = ent_t + WINDOW_MS
            out, ret, dt = "timeout", None, None
            j = j0
            while j < len(t) and t[j] <= end_t:
                if H[j] >= tp_px:
                    out, ret, dt = "TP", TP*100 - FEE, (t[j]-ent_t)/86400000; break
                if sl_px is not None and L[j] <= sl_px:
                    out, ret, dt = "stop", -stop*100 - FEE, (t[j]-ent_t)/86400000; break
                j += 1
            if ret is None:
                jl = min(j, len(t)-1)
                ret = (C[jl]/ent_px - 1)*100 - FEE; dt = WINDOW_MS/86400000
            res.append((out, ret, dt))
    return res


SHARED = {}


def main():
    print("FAITHFUL TP+50% first-touch test of the user's signal\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), DS, DE, log=lambda *a: None)
    print(f"  universe: {len(raw)} symbols; finding signals...", flush=True)
    sigs = {}; ncoins = 0
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        mo = resample_m(df)
        if len(mo) < 20:
            continue
        ncoins += 1
        k, d = stoch1433(mo["h"].to_numpy(), mo["l"].to_numpy(), mo["c"].to_numpy())
        mc = mo["c"].to_numpy(); mt = mo["t"].to_numpy()
        lst = []
        for i in range(16, len(mc)):
            if np.isfinite(k[i]) and np.isfinite(d[i]) and (k[i] > d[i]) and (d[i] <= 10):
                lst.append((int(mt[i]), float(mc[i])))     # enter at signal month close
        if lst:
            sigs[sym] = lst
        df.drop(columns=["dt"], inplace=True)
    SHARED["raw"] = raw; SHARED["sigs"] = sigs
    nsig = sum(len(v) for v in sigs.values())
    print(f"  {ncoins} coins, {nsig} signals (monthly %K>%D & %D<=10)\n", flush=True)

    print(f"{'scenario':<12}{'n':>6}{'TP%':>7}{'stop%':>7}{'timeo%':>8}"
          f"{'WIN%':>7}{'mean':>9}{'median':>9}{'TPdays':>8}")
    print("-" * 76)
    for name, stop in STOPS.items():
        r = simulate(stop)
        n = len(r)
        outs = np.array([x[0] for x in r]); rets = np.array([x[1] for x in r])
        tpdays = np.array([x[2] for x in r if x[0] == "TP"])
        print(f"{name:<12}{n:>6}{(outs=='TP').mean()*100:>6.0f}%"
              f"{(outs=='stop').mean()*100:>6.0f}%{(outs=='timeout').mean()*100:>7.0f}%"
              f"{(rets>0).mean()*100:>6.0f}%{rets.mean():>+8.1f}%{np.median(rets):>+8.1f}%"
              f"{(tpdays.mean() if len(tpdays) else 0):>7.0f}")
    print("\nWIN% = نسبة الصفقات الرابحة فعلاً. mean = التوقّع الكامل/صفقة (الحَكَم).")
    print("لاحظ: 'timeout' = لم تصل +50% ولا الوقف -> أُغلقت بسعر السوق (قد تكون تحت الماء).")
    print("\nDONE_TP50.", flush=True)


if __name__ == "__main__":
    main()
