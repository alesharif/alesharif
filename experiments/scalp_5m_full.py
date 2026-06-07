#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FULL 5m MACD zero-cross scalp, processed MONTH-BY-MONTH over ALL eligible coins,
2024-2025. Each month is downloaded, processed, reported INDEPENDENTLY ($2,000 fresh),
then wiped -> next month. Streams results so progress is visible and interruption-safe.

Entry (5m): MACD line crosses above 0. Filter: MACD line rising on the last CLOSED bar
of 15m/1h/4h/1d (all). Volume gate: $2M < trailing-30d $vol < $200M at signal. Exclude
stablecoins/commodity/leveraged tokens. Exit TP+3%/SL-2%. Cost 0.25% RT. Portfolio per
month: start $2,000, stake 10% equity, max 10 concurrent. Run: python experiments/scalp_5m_full.py
"""

from __future__ import annotations

import gc, sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

US, UE = "2023-10-01", "2026-01-01"
VLO, VHI = 2e6, 2e8
START = 2000.0; STAKE = 0.10; MAXPOS = 10
TP = 0.03; SL = 0.02; COST = 0.25; MAXHOLD = 3*24*12; WARMUP_D = 45
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
    for tag in ("UP", "DOWN", "BULL", "BEAR"):
        if base.endswith(tag):
            return True
    if base[-2:] in ("3L", "3S", "5L", "5S"):
        return True
    return False


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


def month_trades(coins, cand, ld, ws, we):
    """signals entering [ws,we) for all coins; returns list of (ent_t,xt,rr)."""
    out = []
    for c in coins:
        df5 = HR.load_range(c, "5m", ld, we)
        if df5 is None or len(df5) < 500:
            continue
        df5 = df5.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        t5 = df5["time"].to_numpy(); c5 = df5["close"].to_numpy(float)
        h5 = df5["high"].to_numpy(float); l5 = df5["low"].to_numpy(float)
        m5 = macd_line(c5)
        m15, s15 = frame_macd(df5, "15min"); m1h, s1h = frame_macd(df5, "1h")
        m4h, s4h = frame_macd(df5, "4h"); m1d, s1d = frame_macd(df5, "1D")
        dtt, trail = cand[c]; n = len(c5)
        for i in range(30, n-1):
            ent_t = int(t5[i])
            if ent_t < ws or ent_t >= we:
                continue
            if not (m5[i-1] <= 0 and m5[i] > 0):
                continue
            di = np.searchsorted(dtt, ent_t, side="right") - 1
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
            out.append((ent_t, xt, ret))
        del df5; gc.collect()
    return out


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
        op.append((xt, stake*(1+rr/100), stake))
        nwin += rr > 0; nloss += rr <= 0
    for xt, payout, s in sorted(op):
        cash += payout
    return nwin, nloss, cash-START, (cash/START-1)*100


def main():
    print("FULL 5m scalp, month-by-month, ALL eligible coins\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(US), ms(UE), log=lambda *a: None)
    cand = {}
    for sym, df in raw.items():
        if excluded(sym):
            continue
        g = df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"], unit="ms"))
        dv = (g["close"]*g["volume"]).resample("D").sum().dropna()
        if len(dv) < 40:
            continue
        trail = dv.rolling(30, min_periods=10).mean()
        if not ((trail > VLO) & (trail < VHI)).any():
            continue
        cand[sym] = (np.array([ts.value // 10**6 for ts in trail.index]), trail.to_numpy())
    del raw; gc.collect()
    print(f"عملات مؤهّلة (ضمن $2M-$200M): {len(cand)}\n", flush=True)

    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    print(f"{'الشهر':<9}{'رابحة':>7}{'خاسرة':>7}{'نسبة الربح':>11}{'عائد%':>9}{'ربح $':>10}{'عملات':>7}", flush=True)
    print("-"*60, flush=True)
    pos_m = 0; tot = 0.0; rows = []
    for m0 in months:
        ws = int(m0.value // 10**6); we = int((m0 + pd.offsets.MonthBegin(1)).value // 10**6)
        ld = int((m0 - pd.Timedelta(days=WARMUP_D)).value // 10**6)
        # eligible coins this month
        elig = []
        for c, (dtt, trail) in cand.items():
            lo = np.searchsorted(dtt, ws); hi = np.searchsorted(dtt, we)
            seg = trail[lo:hi]
            if len(seg) and np.nanmax(np.where(np.isfinite(seg), seg, 0)) > VLO and np.nanmin(np.where(np.isfinite(seg), seg, 1e18)) < VHI:
                elig.append(c)
        # parallel download this month's window for eligible coins
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(ld, unit="ms"), pd.Timestamp(we, unit="ms"), freq="D")]
        def _fetch(cd):
            try:
                HR.load_day(cd[0], "5m", cd[1])
            except Exception:
                pass
        with ThreadPoolExecutor(max_workers=24) as ex:
            list(ex.map(_fetch, [(c, d) for c in elig for d in days]))
        tr = month_trades(elig, cand, ld, ws, we)
        nwin, nloss, prof, ret = sim_month(tr)
        mon = m0.strftime("%Y-%m"); ntk = nwin+nloss; wr = nwin/ntk*100 if ntk else 0
        print(f"{mon:<9}{nwin:>7}{nloss:>7}{wr:>10.0f}%{ret:>+8.1f}%{prof:>+9,.0f}${len(elig):>7}", flush=True)
        pos_m += ret > 0; tot += prof; rows.append((mon, nwin, nloss, ret, prof))
        gc.collect()
    print("-"*60, flush=True)
    print(f"أشهر موجبة: {pos_m}/{len(rows)}  |  مجموع الربح (لو $2000 كل شهر مستقل): {tot:+,.0f}$", flush=True)
    print(f"الإعداد: 5m MACD0 + فلتر MACD صاعد 15m/1h/4h/1d | TP+3/SL-2 | عمولة {COST}% | 10%/صفقة | 10 متزامنة | $2000/شهر مستقل")
    print("⚠️ انحياز بقاء. عمولة وانزلاق تقديريان.")
    print("\nDONE_FULL5M.", flush=True)


if __name__ == "__main__":
    main()
