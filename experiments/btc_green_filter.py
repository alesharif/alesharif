#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BTC weekly GREEN-candle regime filter (user's idea): only trade when the last
completed WEEKLY BTC candle closed green (close > open). Test alone and combined
with the 99-day filter (our best), on the developed squeeze system. In-sample.
Run:  python experiments/btc_green_filter.py
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


def main():
    print("BTC weekly GREEN-candle filter (alone & with 99d), developed system\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)

    # BTC weekly green regime (last completed weekly candle close>open), lagged
    btc = raw["BTCUSDT"]
    bts = btc.set_index(pd.to_datetime(btc["time"], unit="ms")).sort_index()
    wk = pd.DataFrame({"o": bts["open"].resample("W").first(),
                       "c": bts["close"].resample("W").last()}).dropna()
    green = (wk["c"] > wk["o"]).astype(float).shift(1)        # prior completed week
    pct_green = (green == 1.0).mean()*100
    print(f"  BTC weekly: {len(wk)} weeks, green {pct_green:.0f}% of weeks", flush=True)

    def btc_green(t):
        v = green.asof(pd.Timestamp(t, unit="ms"))
        return bool(v == 1.0) if pd.notna(v) else False

    trades = []          # (entry_t, net_ret, month_key, pass99, btc_green_ok)
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
        o = d["o"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy()
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        dv = c*v
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
            trades.append((ent_t, nr, pd.Timestamp(ent_t, unit="ms").strftime("%Y-%m"),
                           c[i] >= c[i-99], btc_green(ent_t)))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def report(name, sel):
        if not sel:
            print(f"{name:<26}  (no trades)"); return
        rets = np.array([x[1] for x in sel]); mo = {}
        for x in sel:
            mo.setdefault(x[2], []).append(x[1])
        posm = sum(1 for k in mo if np.mean(mo[k]) > 0)
        print(f"{name:<26}{len(rets):>7}{rets.mean():>+8.1f}%{posm:>5}/{len(mo):<3}{posm/len(mo)*100:>6.0f}%")

    print(f"\n{'config':<26}{'n':>7}{'exp':>9}{'months+':>9}{'cons%':>7}")
    print("-" * 58)
    report("base (developed)", trades)
    report("99d only", [x for x in trades if x[3]])
    report("BTC weekly-green", [x for x in trades if x[4]])
    report("BTC green + 99d", [x for x in trades if x[3] and x[4]])
    print("\nنقارن: هل شرط BTC الأخضر الأسبوعي يرفع التوقّع/الثبات فوق 99 يوم؟")
    print("\nDONE_GREEN.", flush=True)


if __name__ == "__main__":
    main()
