#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raise WR: entry = 5m RSI(14) crossing UP through a threshold, KEEP the multi-TF
MACD-rising filter (15m/1h/4h/1d) + volume gate. Test thresholds 20/22/25/27/30/35/40
on ONE month (2025-01). TP+3/SL-2, cost 0.25%, $2000 independent, 10%/trade, 10 conc.
Reuses cached 5m. Run: python experiments/scalp_rsi_test.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

MONTH = "2025-01"; US, UE = "2024-11-01", "2025-02-01"
VLO, VHI = 2e6, 2e8
START = 2000.0; STAKE = 0.10; MAXPOS = 10
TP = 0.03; SL = 0.02; COST = 0.25; MAXHOLD = 3*24*12; WARMUP_D = 45
THRS = [40, 45, 50, 55, 60]
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP",
          "USTC","PYUSD","XUSD","EURT","BFUSD"}
COMMODITY = {"PAXG","XAUT","WBTC","WBETH","BETH"}


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()
def macd_line(c): return ema(c, 12) - ema(c, 26)


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def excluded(sym):
    if not sym.endswith("USDT"):
        return True
    base = sym[:-4]
    if base in STABLE or base in COMMODITY:
        return True
    for tag in ("UP", "DOWN", "BULL", "BEAR"):
        if base.endswith(tag):
            return True
    return base[-2:] in ("3L", "3S", "5L", "5S")


def frame_macd(df5, rule):
    g = df5.set_index(pd.to_datetime(df5["time"], unit="ms"))
    s = g["close"].resample(rule).last().dropna()
    if len(s) < 30:
        return None, None
    return macd_line(s.to_numpy()), np.array([ts.value // 10**6 for ts in s.index])


def rising_at(m, start, ent_t):
    if m is None:
        return False
    ci = np.searchsorted(start, ent_t, side="right") - 1; j = ci - 1
    return j >= 1 and m[j] > m[j-1]


def sim_month(trades):
    trades = sorted(trades); cash = START; op = []; nwin = nloss = 0
    for et, xt, rr in trades:
        op.sort()
        while op and op[0][0] <= et:
            _0, payout, _s = op.pop(0); cash += payout
        equity = cash + sum(s for _, _2, s in op)
        if len(op) >= MAXPOS or cash <= 1:
            continue
        stake = min(equity*STAKE, cash); cash -= stake
        op.append((xt, stake*(1+rr/100), stake)); nwin += rr > 0; nloss += rr <= 0
    for xt, payout, s in sorted(op):
        cash += payout
    return nwin, nloss, cash-START, (cash/START-1)*100


def main():
    print(f"RSI cross-up entry + multi-TF MACD filter — {MONTH}\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(US), ms(UE), log=lambda *a: None)
    cand = {}
    for sym, df in raw.items():
        if excluded(sym):
            continue
        g = df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"], unit="ms"))
        dv = (g["close"]*g["volume"]).resample("D").sum().dropna()
        if len(dv) < 20:
            continue
        trail = dv.rolling(30, min_periods=10).mean()
        if not ((trail > VLO) & (trail < VHI)).any():
            continue
        cand[sym] = (np.array([ts.value // 10**6 for ts in trail.index]), trail.to_numpy())
    del raw; gc.collect()
    coins = list(cand)
    m0 = pd.Timestamp(MONTH + "-01")
    ws = int(m0.value // 10**6); we = int((m0 + pd.offsets.MonthBegin(1)).value // 10**6)
    ld = int((m0 - pd.Timedelta(days=WARMUP_D)).value // 10**6)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(ld, unit="ms"), pd.Timestamp(we, unit="ms"), freq="D")]
    print(f"عملات: {len(coins)} | تحميل/قراءة 5m للشهر...", flush=True)
    def _f(cd):
        try: HR.load_day(cd[0], "5m", cd[1])
        except Exception: pass
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(_f, [(c, d) for c in coins for d in days]))

    cands = []   # (rprev, rcur, ent_t, xt, ret) for every rising-RSI bar passing filters
    for c in coins:
        df5 = HR.load_range(c, "5m", ld, we)
        if df5 is None or len(df5) < 500:
            continue
        df5 = df5.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        t5 = df5["time"].to_numpy(); c5 = df5["close"].to_numpy(float)
        h5 = df5["high"].to_numpy(float); l5 = df5["low"].to_numpy(float)
        r5 = rsi(c5, 14)
        m15, s15 = frame_macd(df5, "15min"); m1h, s1h = frame_macd(df5, "1h")
        m4h, s4h = frame_macd(df5, "4h"); m1d, s1d = frame_macd(df5, "1D")
        dtt, trail = cand[c]; n = len(c5)
        for i in range(20, n-1):
            ent_t = int(t5[i])
            if ent_t < ws or ent_t >= we:
                continue
            if not (r5[i] > r5[i-1]):                      # RSI rising
                continue
            di = np.searchsorted(dtt, ent_t, side="right") - 1
            if di < 0 or not (VLO < trail[di] < VHI):
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
            cands.append((r5[i-1], r5[i], ent_t, xt, ret))
        del df5; gc.collect()

    print(f"إشارات مرشّحة (RSI صاعد + الفلاتر): {len(cands)}\n")
    print(f"{'عتبة RSI':<10}{'صفقات':>7}{'رابحة':>7}{'خاسرة':>7}{'WR':>6}{'عائد%':>9}{'ربح$':>9}")
    print("-"*55)
    for T in THRS:
        sel = [(et, xt, rr) for rp, rc, et, xt, rr in cands if rp <= T < rc]   # crossed up through T
        nwin, nloss, prof, ret = sim_month(sel)
        ntk = nwin+nloss; wr = nwin/ntk*100 if ntk else 0
        print(f"عبور {T:<5}{len(sel):>7}{nwin:>7}{nloss:>7}{wr:>5.0f}%{ret:>+8.1f}%{prof:>+8,.0f}$")
    print("-"*55)
    print("الهدف: WR فوق ~45% (عتبة التعادل بعد العمولة). MACD-صفر السابق كان 40% في هذا الشهر.")
    print("\nDONE_RSITEST.", flush=True)


if __name__ == "__main__":
    main()
