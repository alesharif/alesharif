#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOLD-STANDARD out-of-sample: run BOTH developed strategies with their exact
Binance-tuned parameters on KUCOIN's ~900 coins (an independent dataset). If the
edges hold here, they are real; if they collapse, they were overfit to Binance.

BREAKOUT (trend): daily BB-squeeze->expand + MACD up + EMA20~50 + bullish + close>
  EMA200 + liquid + 99-filter(close>=close[-99]); exit TP+100/SL-10, 90d.
MEAN-REVERSION (drought): weekly-RSI cross>30 + daily close>EMA10 + liquid +
  not-down->30%/99d; exit TP+15/SL-15, 60d.
Net 1% round-trip + 1% stop slip. Per regime FULL/2024/2025. Daily-resolution
first-touch (stop assumed first on ambiguous days). Run: python experiments/kucoin_oos.py
"""

from __future__ import annotations

import json, time, urllib.request
import numpy as np
import pandas as pd

LIQ_MIN = 300_000; COST = 1.0; STOP_SLIP = 1.0
START = int(pd.Timestamp("2022-06-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-06-01", tz="UTC").timestamp())
SF = int(pd.Timestamp("2022-09-01", tz="UTC").timestamp()*1000)
UA = {"User-Agent": "Mozilla/5.0"}


def get(url, tries=3):
    for k in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read())
        except Exception:
            time.sleep(0.5*(k+1))
    return None


def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def first_touch(h, l, c, i, tp, sl, horizon):
    end = min(i+1+horizon, len(c))
    for j in range(i+1, end):
        if l[j] <= c[i]*(1-sl):
            return -sl*100 - COST - STOP_SLIP
        if h[j] >= c[i]*(1+tp):
            return tp*100 - COST
    return (c[end-1]/c[i]-1)*100 - COST


def bucket(store, yr, nr):
    store["full"].append(nr)
    if yr in (2024, 2025):
        store[yr].append(nr)


def main():
    print("KuCoin OUT-OF-SAMPLE: Binance-tuned breakout + MR on 900 KuCoin coins\n", flush=True)
    r = get("https://api.kucoin.com/api/v1/symbols")
    syms = [s["symbol"] for s in r["data"] if s.get("quoteCurrency") == "USDT" and s.get("enableTrading")]
    print(f"  {len(syms)} symbols; fetching daily + backtesting...", flush=True)
    BO = {"full": [], 2024: [], 2025: []}; MR = {"full": [], 2024: [], 2025: []}
    DBG = {"cross": 0, "afteridx": 0, "afterconf": 0}
    done = 0
    for sym in syms:
        j = get(f"https://api.kucoin.com/api/v1/market/candles?type=1day&symbol={sym}&startAt={START}&endAt={END}")
        time.sleep(0.07); done += 1
        if done % 200 == 0:
            print(f"   {done}/{len(syms)}...", flush=True)
        if not j or j.get("code") != "200000" or not j.get("data"):
            continue
        a = np.array([[float(x[0])*1000, float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[6])]
                      for x in j["data"]])           # t_ms, open, close, high, low, turnover
        a = a[a[:, 0].argsort()]
        if len(a) < 230:
            continue
        t = a[:, 0]; o = a[:, 1]; c = a[:, 2]; h = a[:, 3]; l = a[:, 4]; turn = a[:, 5]
        n = len(c)
        liq = pd.Series(turn).rolling(30).mean().to_numpy()
        e20 = ema(c, 20); e50 = ema(c, 50); e200 = ema(c, 200); e10 = ema(c, 10)
        macd = ema(c, 12)-ema(c, 26); sg = ema(macd, 9); hist = macd-sg
        sma = pd.Series(c).rolling(20).mean(); std = pd.Series(c).rolling(20).std()
        bbw = (4*std/sma).to_numpy(); q25 = pd.Series(bbw).rolling(100).quantile(0.25).to_numpy()
        # ---- BREAKOUT ----
        for i in range(200, n-1):
            if t[i] < SF or not np.isfinite(liq[i]) or liq[i] <= LIQ_MIN or i < 99:
                continue
            if (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and c[i] >= c[i-99]):
                nr = first_touch(h, l, c, i, 1.00, 0.10, 90)
                bucket(BO, int(pd.Timestamp(t[i], unit="ms").year), nr)
        # ---- MEAN-REVERSION (weekly RSI) ----
        s = pd.Series(c, index=pd.to_datetime(t, unit="ms"))
        wk = s.resample("W").last().dropna(); wc = wk.to_numpy()
        wt = np.array([ts.value // 10**6 for ts in wk.index])   # ns -> ms, unit-safe
        if len(wc) >= 25:
            wr = rsi(wc, 14)
            for wi in range(15, len(wc)):
                if not (wr[wi-1] <= 30 and wr[wi] > 30 and wt[wi] >= SF):
                    continue
                DBG["cross"] += 1
                i = int(np.searchsorted(t, wt[wi], side="right")-1)
                if i < 99 or i >= n-1 or not np.isfinite(liq[i]) or liq[i] <= LIQ_MIN:
                    continue
                DBG["afteridx"] += 1
                if not (c[i] > e10[i] and (c[i]/c[i-99]-1) >= -0.30):
                    continue
                DBG["afterconf"] += 1
                nr = first_touch(h, l, c, i, 0.15, 0.15, 60)
                bucket(MR, int(pd.Timestamp(t[i], unit="ms").year), nr)

    def rep(name, st):
        def m(k):
            x = np.array(st[k]); return x.mean() if len(x) else float("nan")
        wr2 = (np.array(st["full"]) > 0).mean()*100 if st["full"] else float("nan")
        print(f"{name:<14}{len(st['full']):>7}{m('full'):>+8.1f}%{m(2024):>+9.1f}%{m(2025):>+9.1f}%{wr2:>7.0f}%")

    print(f"\n{'strategy':<14}{'n':>7}{'FULL':>9}{'2024(trend)':>10}{'2025(drought)':>10}{'win%':>8}")
    print("-" * 60)
    rep("BREAKOUT", BO)
    rep("MEAN-REVERT", MR)
    print(f"\nMR diag: weekly-crosses={DBG['cross']}, after idx/liq={DBG['afteridx']}, "
          f"after confirm+99={DBG['afterconf']}")
    print("\nBinance kept: breakout 2024 ~+19%, MR 2025 ~+5%. هل تصمد على KuCoin؟")
    print("\nDONE_KUOOS.", flush=True)


if __name__ == "__main__":
    main()
