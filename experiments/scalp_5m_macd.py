#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5m MACD zero-cross scalp with multi-timeframe MACD-rising filter.
Entry (5m): MACD line crosses above 0. Filter: MACD line rising (slope up) on the
last CLOSED bar of 15m, 1h, 4h, 1d (all four). Exit: TP +3% / SL -2% first touch.
Cost: 0.25% round trip (fees+slippage). 30 liquid coins, 2024-10 -> 2025-06.
Splits alt-season (2024-10..2025-01) vs drought (2025-01..2025-06).
Run: python experiments/scalp_5m_macd.py
"""

from __future__ import annotations

import sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from binance_sim import hires_data as HR                  # noqa: E402

W0, W1 = "2024-10-01", "2025-06-01"
COST = 0.25; TP = 0.03; SL = 0.02; MAXHOLD = 3*24*12      # 3 days in 5m bars
COINS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT",
         "LINKUSDT","DOTUSDT","LTCUSDT","TRXUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT",
         "INJUSDT","SUIUSDT","SEIUSDT","TIAUSDT","FILUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT",
         "RUNEUSDT","ALGOUSDT","ICPUSDT","PEPEUSDT","WIFUSDT","ORDIUSDT","FETUSDT","GALAUSDT"]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def macd_line(close): return ema(close, 12) - ema(close, 26)


def frame_macd(df5, rule):
    g = df5.set_index(pd.to_datetime(df5["time"], unit="ms"))
    o = pd.DataFrame({"c": g["close"].resample(rule).last()}).dropna()
    c = o["c"].to_numpy()
    if len(c) < 30:
        return None, None
    m = macd_line(c)
    start = np.array([ts.value // 10**6 for ts in o.index])
    return m, start


def rising_at(m, start, ent_t):
    """MACD rising on the last CLOSED bar before ent_t (no lookahead)."""
    if m is None:
        return False
    ci = np.searchsorted(start, ent_t, side="right") - 1   # bar containing ent_t (forming)
    j = ci - 1                                             # last fully closed bar
    if j < 1:
        return False
    return m[j] > m[j-1]


def main():
    print("5m MACD zero-cross scalp + multi-TF MACD-rising filter\n", flush=True)
    s0, s1 = ms(W0), ms(W1)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(W0, W1, freq="D")]
    # parallel prefetch all (coin, day) 5m archives into cache
    print(f"تحميل 5m: {len(COINS)} عملة × {len(days)} يوم ...", flush=True)
    t0 = time.time(); tasks = [(c, d) for c in COINS for d in days]
    done = [0]
    def fetch(cd):
        HR.load_day(cd[0], "5m", cd[1]); done[0] += 1
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(fetch, tasks))
    print(f"  اكتمل التحميل في {(time.time()-t0)/60:.1f} دقيقة\n", flush=True)

    rows = []   # (ent_t, ret_net, coin)
    for c in COINS:
        df5 = HR.load_range(c, "5m", s0, s1)
        if df5 is None or len(df5) < 500:
            print(f"  {c}: لا بيانات", flush=True); continue
        df5 = df5.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        t5 = df5["time"].to_numpy(); c5 = df5["close"].to_numpy(float)
        h5 = df5["high"].to_numpy(float); l5 = df5["low"].to_numpy(float)
        m5 = macd_line(c5)
        m15, s15 = frame_macd(df5, "15min"); m1h, s1h = frame_macd(df5, "1h")
        m4h, s4h = frame_macd(df5, "4h"); m1d, s1d = frame_macd(df5, "1D")
        n = len(c5); nsig = 0
        for i in range(30, n-1):
            if not (m5[i-1] <= 0 and m5[i] > 0):           # 5m MACD zero-cross up
                continue
            ent_t = int(t5[i])
            if not (rising_at(m15, s15, ent_t) and rising_at(m1h, s1h, ent_t)
                    and rising_at(m4h, s4h, ent_t) and rising_at(m1d, s1d, ent_t)):
                continue
            P0 = c5[i]; tp = P0*(1+TP); sl = P0*(1-SL); ret = None
            end = min(i+1+MAXHOLD, n)
            for k in range(i+1, end):
                if l5[k] <= sl: ret = -SL*100 - COST; break
                if h5[k] >= tp: ret = TP*100 - COST; break
            if ret is None:
                ret = (c5[end-1]/P0-1)*100 - COST
            rows.append((ent_t, ret, c)); nsig += 1
        print(f"  {c}: {nsig} إشارة", flush=True)

    if not rows:
        print("لا إشارات."); return
    rows.sort()
    ent = np.array([r[0] for r in rows]); ret = np.array([r[1] for r in rows])
    split = ms("2025-01-01")

    def stat(mask, lab):
        a = ret[mask]
        if not len(a):
            print(f"{lab:<22} لا صفقات"); return
        print(f"{lab:<22}{len(a):>7}{a.mean():>+9.2f}%{(a > 0).mean()*100:>8.0f}%{a.sum():>+10.0f}%")

    print(f"\n{'الفترة':<22}{'صفقات':>7}{'متوسط/صفقة':>10}{'ربح%':>8}{'مجموع':>10}")
    print("-"*57)
    stat(ent < split, "موسم بديل (2024Q4)")
    stat(ent >= split, "جفاف (2025)")
    stat(np.ones(len(ret), bool), "الكل")
    print("-"*57)
    print(f"الإعداد: TP+3% SL−2% عمولة {COST}% | فلتر MACD صاعد على 15m/1h/4h/1d")
    print("⚠️ عملات كبيرة سائلة (انحياز بقاء). العمولة محسوبة، الانزلاق تقديري.")
    print("\nDONE_SCALP.", flush=True)


if __name__ == "__main__":
    main()
