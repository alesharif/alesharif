#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Market-regime filters: BTC-trend vs stablecoin-flow, each on daily/weekly/
monthly — which filter and which timeframe best fixes the CONSISTENCY problem?

Developed system: squeeze breakout + daily above-EMA200 + SL10 + TP+100% + liquid
(the 2394-trade base; baseline was only 15/42 months positive). We only take a
trade when the market regime is 'risk-on':
  BTC filter   : BTC close > EMA20 on the timeframe (BTC uptrend)
  STABLE filter: stablecoin-volume ratio < EMA20 on the timeframe (money NOT
                 fleeing to stables = risk-on)
Reported per config: trades, expectancy, and MONTHS-POSITIVE (the real goal).
In-sample 2022-2026. Run:  python experiments/regime_filters.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402

SL = 0.10; TP = 1.0; WINDOW_MS = 90*24*3600*1000; LIQ_MIN = 300_000
S, E, SF = "2022-06-01", "2026-06-01", "2022-09-01"
TFS = {"daily": "D", "weekly": "W", "monthly": "ME"}


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def regime_series(close_series, rule, bullish_when_above):
    """Return (period_end_times[], risk_on[]) lagged (period must be closed)."""
    r = close_series.resample(rule)
    c = r.last().dropna()
    if len(c) < 25:
        return np.array([]), np.array([])
    e = pd.Series(c.values).ewm(span=20, adjust=False).mean().to_numpy()
    if bullish_when_above:
        risk = c.values > e
    else:
        risk = c.values < e
    # period_end time = last index timestamp of each period (ms)
    pend = np.asarray(c.index.astype("int64")) // 10**6
    return pend, risk.astype(bool)


def query(pend, risk, t):
    if len(pend) == 0:
        return False
    idx = np.searchsorted(pend, t, side="left") - 1     # last period ended strictly before t
    return bool(risk[idx]) if idx >= 0 else False


def main():
    print("Regime filters: BTC vs STABLE, on daily/weekly/monthly (in-sample)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)

    # ---- build regime series ----
    btc = raw.get("BTCUSDT")
    btc_close = pd.Series(btc["close"].to_numpy(float),
                          index=pd.to_datetime(btc["time"], unit="ms")).sort_index()
    sr = PB.build_stable_ratio(ms(S), ms(E))
    sr_items = sorted(sr.items())
    stable_close = pd.Series([v for _, v in sr_items],
                             index=pd.to_datetime([t for t, _ in sr_items], unit="ms")).sort_index()
    REG = {}
    for tfn, rule in TFS.items():
        REG[("BTC", tfn)] = regime_series(btc_close, rule, True)
        REG[("STABLE", tfn)] = regime_series(stable_close, rule, False)

    # ---- developed-system trades ----
    trades = []          # (entry_t, net_ret, month_key)
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
            trades.append((ent_t, nr, pd.Timestamp(ent_t, unit="ms").strftime("%Y-%m")))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def report(name, sel):
        if not sel:
            print(f"{name:<20}  (no trades)"); return
        rets = np.array([x[1] for x in sel])
        mo = {}
        for _, nr, mk in sel:
            mo.setdefault(mk, []).append(nr)
        posm = sum(1 for k in mo if np.mean(mo[k]) > 0)
        print(f"{name:<20}{len(rets):>7}{rets.mean():>+9.1f}%{posm:>6}/{len(mo):<3}"
              f"{posm/len(mo)*100:>6.0f}%")

    print(f"\n{'config':<20}{'n':>7}{'exp':>10}{'months+':>10}{'cons%':>7}")
    print("-" * 56)
    report("BASE (no filter)", trades)
    for filt in ("BTC", "STABLE"):
        for tfn in TFS:
            pend, risk = REG[(filt, tfn)]
            sel = [x for x in trades if query(pend, risk, x[0])]
            report(f"{filt} / {tfn}", sel)
    print("\nالفائز = أعلى cons% (نسبة الأشهر الموجبة) مع حفاظ معقول على التوقّع وعدد الصفقات.")
    print("\nDONE_REGIME.", flush=True)


if __name__ == "__main__":
    main()
