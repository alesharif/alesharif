#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the top-3 EMA-cross pairs (150/200, 100/200, 35/50) across all months
2024-2025. 5m entry = fast EMA crosses above slow EMA + big-frame MACD-rising filter
(15m/1h/4h/1d) + vol gate. TP+3/SL-2, cost 0.25%, $2000 independent, 10%/trade, 10 conc.
Streams per-month WR/return for each pair. Reuses cached 5m.
Run: python experiments/scalp_ema_validate.py
"""

from __future__ import annotations

import gc, sys
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
PAIRS = [(150, 200), (100, 200), (35, 50)]
EMAS = sorted({p for pr in PAIRS for p in pr})
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


def sim(trades):
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
    ntk = nwin+nloss
    return ntk, (nwin/ntk*100 if ntk else 0), (cash/START-1)*100, cash-START


def month_pairsigs(coins, cand, ld, ws, we):
    res = {p: [] for p in PAIRS}
    for c in coins:
        df5 = HR.load_range(c, "5m", ld, we)
        if df5 is None or len(df5) < 500:
            continue
        df5 = df5.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        t5 = df5["time"].to_numpy(); c5 = df5["close"].to_numpy(float)
        h5 = df5["high"].to_numpy(float); l5 = df5["low"].to_numpy(float)
        E = {p: ema(c5, p) for p in EMAS}
        m15, s15 = frame_macd(df5, "15min"); m1h, s1h = frame_macd(df5, "1h")
        m4h, s4h = frame_macd(df5, "4h"); m1d, s1d = frame_macd(df5, "1D")
        dtt, trail = cand[c]; n = len(c5); outcome = {}
        def out(i):
            if i in outcome:
                return outcome[i]
            ent_t = int(t5[i]); di = np.searchsorted(dtt, ent_t, side="right") - 1
            if di < 0 or not (VLO < trail[di] < VHI):
                outcome[i] = None; return None
            if not (rising_at(m15, s15, ent_t) and rising_at(m1h, s1h, ent_t)
                    and rising_at(m4h, s4h, ent_t) and rising_at(m1d, s1d, ent_t)):
                outcome[i] = None; return None
            P0 = c5[i]; tp = P0*(1+TP); sl = P0*(1-SL); ret = None; xt = ent_t
            end = min(i+1+MAXHOLD, n)
            for k in range(i+1, end):
                if l5[k] <= sl: ret = -SL*100 - COST; xt = int(t5[k]); break
                if h5[k] >= tp: ret = TP*100 - COST; xt = int(t5[k]); break
            if ret is None:
                ret = (c5[end-1]/P0-1)*100 - COST; xt = int(t5[end-1])
            outcome[i] = (ent_t, xt, ret); return outcome[i]
        for f, s in PAIRS:
            cf = E[f]; cs = E[s]
            idx = np.nonzero((cf[:-1] <= cs[:-1]) & (cf[1:] > cs[1:]))[0] + 1
            for i in idx:
                if i < 20 or i >= n-1:
                    continue
                et = int(t5[i])
                if et < ws or et >= we:
                    continue
                o = out(int(i))
                if o is not None:
                    res[(f, s)].append(o)
        del df5, E, outcome; gc.collect()
    return res


def main():
    print("Validate top-3 EMA pairs across all months 2024-2025\n", flush=True)
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
    print(f"عملات: {len(cand)}\n", flush=True)
    hdr = f"{'الشهر':<9}"
    for f, s in PAIRS:
        hdr += f"{'E'+str(f)+'/'+str(s)+'_WR':>11}{'عائد':>8}"
    print(hdr, flush=True); print("-"*len(hdr.encode('ascii','ignore'))*0 + "-"*58, flush=True)
    tot = {p: 0.0 for p in PAIRS}; pos = {p: 0 for p in PAIRS}; nm = 0
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    for m0 in months:
        ws = int(m0.value // 10**6); we = int((m0 + pd.offsets.MonthBegin(1)).value // 10**6)
        ld = int((m0 - pd.Timedelta(days=WARMUP_D)).value // 10**6)
        elig = []
        for c, (dtt, trail) in cand.items():
            lo = np.searchsorted(dtt, ws); hi = np.searchsorted(dtt, we); seg = trail[lo:hi]
            if len(seg) and np.nanmax(np.where(np.isfinite(seg), seg, 0)) > VLO and np.nanmin(np.where(np.isfinite(seg), seg, 1e18)) < VHI:
                elig.append(c)
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(ld, unit="ms"), pd.Timestamp(we, unit="ms"), freq="D")]
        def _f(cd):
            try: HR.load_day(cd[0], "5m", cd[1])
            except Exception: pass
        with ThreadPoolExecutor(max_workers=24) as ex:
            list(ex.map(_f, [(c, d) for c in elig for d in days]))
        res = month_pairsigs(elig, cand, ld, ws, we)
        line = f"{m0.strftime('%Y-%m'):<9}"
        for p in PAIRS:
            ntk, wr, ret, prof = sim(res[p])
            line += f"{wr:>9.0f}%{ret:>+7.1f}%"
            tot[p] += prof; pos[p] += ret > 0
        print(line, flush=True); nm += 1; gc.collect()
    print("-"*58, flush=True)
    for p in PAIRS:
        print(f"EMA{p[0]}/{p[1]}: أشهر موجبة {pos[p]}/{nm} | مجموع ربح {tot[p]:+,.0f}$")
    print("\nDONE_EMAVAL.", flush=True)


if __name__ == "__main__":
    main()
