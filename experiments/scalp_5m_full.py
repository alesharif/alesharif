#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FULL 5m MACD zero-cross scalp, month-by-month over 2024-2025, all eligible coins.

Entry (5m): MACD line crosses above 0. Filter: MACD line rising on the last CLOSED
bar of 15m/1h/4h/1d (all). Volume gate: $2M < trailing-30d $vol < $200M at signal.
Exclude stablecoins / commodity / leveraged tokens. Exit TP+3%/SL-2%. Cost 0.25% RT.
Portfolio: start $2,000, stake = 10% of equity, max 10 concurrent, compounding.
Output: per-month return%, win-rate, $ profit, trade count.
Run: python experiments/scalp_5m_full.py
"""

from __future__ import annotations

import gc, sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

US, UE = "2023-10-01", "2026-01-01"           # universe window (warmup + test)
W0, W1 = "2024-01-01", "2026-01-01"           # test window (2024-2025)
VLO, VHI = 2e6, 2e8
START = 2000.0; STAKE = 0.10; MAXPOS = 10
TP = 0.03; SL = 0.02; COST = 0.25; MAXHOLD = 3*24*12
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP",
          "USTC","PYUSD","XUSD","EURT","BFUSD"}
COMMODITY = {"PAXG","XAUT","WBTC","WBETH","BETH"}


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()
def macd_line(c): return ema(c, 12) - ema(c, 26)


def excluded(sym):
    if not sym.endswith("USDT"):
        return True
    base = sym[:-4]
    if base in STABLE or base in COMMODITY:
        return True
    for tag in ("UP", "DOWN", "BULL", "BEAR"):              # leveraged tokens
        if base.endswith(tag):
            return True
    if base.endswith("3L") or base.endswith("3S") or base.endswith("5L") or base.endswith("5S"):
        return True
    return False


def frame_macd(df5, rule):
    g = df5.set_index(pd.to_datetime(df5["time"], unit="ms"))
    c = g["close"].resample(rule).last().dropna().to_numpy()
    if len(c) < 30:
        return None, None
    o = g["close"].resample(rule).last().dropna()
    start = np.array([ts.value // 10**6 for ts in o.index])
    return macd_line(c), start


def rising_at(m, start, ent_t):
    if m is None:
        return False
    ci = np.searchsorted(start, ent_t, side="right") - 1
    j = ci - 1
    return j >= 1 and m[j] > m[j-1]


def main():
    print("FULL 5m scalp month-by-month, 2024-2025\n", flush=True)
    # 1) universe -> candidate coins in the $2M-$200M band at some point
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(US), ms(UE), log=lambda *a: None)
    cand = {}
    for sym, df in raw.items():
        if excluded(sym):
            continue
        df = df.sort_values("time")
        g = df.set_index(pd.to_datetime(df["time"], unit="ms"))
        dv = (g["close"]*g["volume"]).resample("D").sum().dropna()
        if len(dv) < 40:
            continue
        trail = dv.rolling(30, min_periods=10).mean()
        in_band = ((trail > VLO) & (trail < VHI))
        if not in_band.any():
            continue
        dtt = np.array([ts.value // 10**6 for ts in trail.index])
        cand[sym] = (dtt, trail.to_numpy())
    del raw; gc.collect()
    coins = sorted(cand)
    MAX_COINS = 150
    if len(coins) > MAX_COINS:
        coins = coins[:MAX_COINS]
    print(f"عملات مؤهّلة (ضمن $2M-$200M، غير مستثناة): {len(cand)} — نختبر عيّنة {len(coins)}", flush=True)

    # 2) parallel prefetch 5m for candidates across the test window
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(W0, W1, freq="D")]
    print(f"تحميل 5m: {len(coins)} عملة × {len(days)} يوم (قد يطول)...", flush=True)
    t0 = time.time(); tasks = [(c, d) for c in coins for d in days]
    def fetch(cd):
        try:
            HR.load_day(cd[0], "5m", cd[1])
        except Exception:
            pass
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(fetch, tasks))
    print(f"  اكتمل التحميل في {(time.time()-t0)/60:.1f} دقيقة\n", flush=True)

    # 3) per-coin signal generation
    s0, s1 = ms(W0), ms(W1); trades = []          # (ent_t, exit_t, ret_net)
    for ci, c in enumerate(coins):
        df5 = HR.load_range(c, "5m", s0, s1)
        if df5 is None or len(df5) < 1000:
            continue
        df5 = df5.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        t5 = df5["time"].to_numpy(); c5 = df5["close"].to_numpy(float)
        h5 = df5["high"].to_numpy(float); l5 = df5["low"].to_numpy(float)
        m5 = macd_line(c5)
        m15, s15 = frame_macd(df5, "15min"); m1h, s1h = frame_macd(df5, "1h")
        m4h, s4h = frame_macd(df5, "4h"); m1d, s1d = frame_macd(df5, "1D")
        dtt, trail = cand[c]; n = len(c5)
        for i in range(30, n-1):
            if not (m5[i-1] <= 0 and m5[i] > 0):
                continue
            ent_t = int(t5[i])
            di = np.searchsorted(dtt, ent_t, side="right") - 1   # daily vol at signal
            if di < 0 or not np.isfinite(trail[di]) or not (VLO < trail[di] < VHI):
                continue
            if not (rising_at(m15, s15, ent_t) and rising_at(m1h, s1h, ent_t)
                    and rising_at(m4h, s4h, ent_t) and rising_at(m1d, s1d, ent_t)):
                continue
            P0 = c5[i]; tp = P0*(1+TP); sl = P0*(1-SL); ret = None; xt = ent_t
            end = min(i+1+MAXHOLD, n)
            for k in range(i+1, end):
                if l5[k] <= sl: ret = -SL*100 - COST; xt = int(t5[k]); break
                if h5[k] >= tp: ret = TP*100 - COST; xt = int(t5[k]); break
            if ret is None:
                ret = (c5[end-1]/P0-1)*100 - COST; xt = int(t5[end-1])
            trades.append((ent_t, xt, ret))
        del df5; gc.collect()
        if (ci+1) % 25 == 0:
            print(f"  عُولج {ci+1}/{len(coins)} عملة، إشارات حتى الآن: {len(trades)}", flush=True)

    print(f"\nإجمالي الإشارات: {len(trades)}", flush=True)
    if not trades:
        print("لا إشارات."); return
    trades.sort()
    import pickle
    pickle.dump(trades, open("/tmp/trades5m.pkl", "wb"))   # save for fast re-reporting

    # 4) INDEPENDENT per-month: each month starts fresh at $2,000, wipes, next month.
    from collections import defaultdict
    bymon = defaultdict(list)
    for et, xt, rr in trades:
        bymon[pd.Timestamp(et, unit="ms").strftime("%Y-%m")].append((et, xt, rr))

    print(f"\n{'الشهر':<9}{'رابحة':>7}{'خاسرة':>7}{'نسبة الربح':>11}{'عائد%':>9}{'ربح $':>10}")
    print("-"*53)
    tot_prof = 0.0; pos_m = 0; nm = 0
    for mon in sorted(bymon):
        mt = sorted(bymon[mon]); cash = START; op = []; nwin = nloss = 0
        for et, xt, rr in mt:
            op.sort()
            while op and op[0][0] <= et:
                xt0, payout, stake = op.pop(0); cash += payout
            equity = cash + sum(s for _, _2, s in op)
            if len(op) >= MAXPOS or cash <= 1:
                continue
            stake = min(equity*STAKE, cash); cash -= stake
            op.append((xt, stake*(1+rr/100), stake))
            if rr > 0: nwin += 1
            else: nloss += 1
        for xt, payout, stake in sorted(op):
            cash += payout
        prof = cash - START; ret = (cash/START-1)*100; ntk = nwin+nloss
        wr = nwin/ntk*100 if ntk else float("nan")
        print(f"{mon:<9}{nwin:>7}{nloss:>7}{wr:>10.0f}%{ret:>+8.1f}%{prof:>+9,.0f}$")
        tot_prof += prof; pos_m += ret > 0; nm += 1
    print("-"*53)
    print(f"أشهر موجبة: {pos_m}/{nm}   |   مجموع الربح (لو $2000 كل شهر مستقل): {tot_prof:+,.0f}$")
    print(f"الإعداد: 5m MACD0 + فلتر MACD صاعد 15m/1h/4h/1d | TP+3/SL-2 | عمولة {COST}% | 10%/صفقة | 10 متزامنة | $2000/شهر مستقل")
    print("⚠️ انحياز بقاء (عملات حالية). عمولة وانزلاق محسوبان تقديرياً.")
    print("\nDONE_FULL5M.", flush=True)


if __name__ == "__main__":
    main()
