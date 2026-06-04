#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runner supply on KUCOIN (900 USDT coins) vs Binance — did the 2025 collapse
happen on KuCoin too, or do its many small coins still produce runners?

Per month: % of liquid KuCoin coins whose forward-90d max gain >= +50%/+100%.
KuCoin daily kline = [time_s, open, close, high, low, volume, turnover(USDT)].
Run:  python experiments/kucoin_supply.py
"""

from __future__ import annotations

import json, time, urllib.request
import numpy as np
import pandas as pd

LIQ_MIN = 300_000; FWD = 90
START = int(pd.Timestamp("2022-06-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-06-01", tz="UTC").timestamp())
UA = {"User-Agent": "Mozilla/5.0"}


def get(url, tries=3):
    for k in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read())
        except Exception:
            time.sleep(0.5 * (k + 1))
    return None


def main():
    print("KuCoin runner supply (% coins +50%/+100% in 90d) per month/year\n", flush=True)
    r = get("https://api.kucoin.com/api/v1/symbols")
    syms = [s["symbol"] for s in r["data"] if s.get("quoteCurrency") == "USDT" and s.get("enableTrading")]
    print(f"  {len(syms)} USDT symbols; fetching daily klines...", flush=True)

    mbs = pd.date_range(pd.Timestamp("2022-09-01", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC"), freq="MS")
    mk = [x.strftime("%Y-%m") for x in mbs]; mt = [int(x.timestamp()) for x in mbs]
    r50 = {k: 0 for k in mk}; r100 = {k: 0 for k in mk}; tot = {k: 0 for k in mk}

    done = 0
    for sym in syms:
        url = f"https://api.kucoin.com/api/v1/market/candles?type=1day&symbol={sym}&startAt={START}&endAt={END}"
        j = get(url)
        time.sleep(0.08)
        done += 1
        if done % 150 == 0:
            print(f"   {done}/{len(syms)}...", flush=True)
        if not j or j.get("code") != "200000" or not j.get("data"):
            continue
        d = j["data"]
        # rows: [time_s, open, close, high, low, volume, turnover]; newest first -> reverse
        arr = np.array([[float(x[0]), float(x[2]), float(x[3]), float(x[6])] for x in d])  # t, close, high, turnover
        arr = arr[arr[:, 0].argsort()]
        ts = arr[:, 0]; c = arr[:, 1]; h = arr[:, 2]; turn = arr[:, 3]
        n = len(c)
        if n < 60:
            continue
        for k, mbt in zip(mk, mt):
            i = np.searchsorted(ts, mbt, side="left")
            if i < 30 or i >= n - 5:
                continue
            if np.nanmean(turn[max(0, i-30):i]) <= LIQ_MIN:
                continue
            tot[k] += 1
            fwd_max = h[i:min(i+FWD, n)].max() / c[i] - 1
            if fwd_max >= 0.50: r50[k] += 1
            if fwd_max >= 1.00: r100[k] += 1

    print(f"\n{'month':<9}{'liquid':>7}{'run+50%':>9}{'run+100%':>10}")
    yr = {}
    for k in mk:
        if tot[k] == 0:
            continue
        p50 = r50[k]/tot[k]*100; p100 = r100[k]/tot[k]*100
        yr.setdefault(k[:4], []).append((p50, p100))
        print(f"{k:<9}{tot[k]:>7}{p50:>7.0f}%{p100:>8.0f}%  {'#'*int(p100/2)}")
    print(f"\n{'YEAR':<9}{'avg+50%':>9}{'avg+100%':>10}    (Binance +100%: 2024=24% 2025=8%)")
    for y in sorted(yr):
        a = np.array(yr[y])
        print(f"{y:<9}{a[:,0].mean():>8.0f}%{a[:,1].mean():>9.0f}%")
    print("\nلو KuCoin 2025 أعلى بكثير من Binance (8%) => عملاته الصغيرة أبقت الرابضين => النظام يعمل عليه.")
    print("\nDONE_KUCOIN.", flush=True)


if __name__ == "__main__":
    main()
