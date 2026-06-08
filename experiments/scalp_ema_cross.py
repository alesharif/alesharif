#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Systematic 5m EMA-crossover scan: for every fast<slow pair from
[5,9,12,15,20,25,35,50,100,150,200], entry = fast EMA crosses ABOVE slow EMA.
Vol gate $2M-$200M, TP+3/SL-2, cost 0.25%, $2000 independent, 10%/trade, 10 conc.
One month (2025-01). Reports WR/return/trades per pair, sorted. Reuses cached 5m.
Run: python experiments/scalp_ema_cross.py
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
EMAS = [5, 9, 12, 15, 20, 25, 35, 50, 100, 150, 200]
STABLE = {"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP",
          "USTC","PYUSD","XUSD","EURT","BFUSD"}
COMMODITY = {"PAXG","XAUT","WBTC","WBETH","BETH"}


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


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
    ntk = nwin+nloss
    return ntk, (nwin/ntk*100 if ntk else 0), (cash/START-1)*100, cash-START


def main():
    print(f"5m EMA crossover scan (55 pairs) — {MONTH}\n", flush=True)
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
    print(f"عملات: {len(coins)}", flush=True)
    def _f(cd):
        try: HR.load_day(cd[0], "5m", cd[1])
        except Exception: pass
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(_f, [(c, d) for c in coins for d in days]))

    pairs = [(EMAS[a], EMAS[b]) for a in range(len(EMAS)) for b in range(a+1, len(EMAS))]
    ptr = {p: [] for p in pairs}      # pair -> list of (ent_t, xt, ret)
    for c in coins:
        df5 = HR.load_range(c, "5m", ld, we)
        if df5 is None or len(df5) < 500:
            continue
        df5 = df5.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        t5 = df5["time"].to_numpy(); c5 = df5["close"].to_numpy(float)
        h5 = df5["high"].to_numpy(float); l5 = df5["low"].to_numpy(float)
        E = {p: ema(c5, p) for p in EMAS}
        dtt, trail = cand[c]; n = len(c5)
        outcome = {}                  # bar i -> (ent_t, xt, ret) cached
        def out(i):
            if i in outcome:
                return outcome[i]
            ent_t = int(t5[i]); di = np.searchsorted(dtt, ent_t, side="right") - 1
            if di < 0 or not (VLO < trail[di] < VHI):
                outcome[i] = None; return None
            P0 = c5[i]; tp = P0*(1+TP); sl = P0*(1-SL); ret = None; xt = ent_t
            end = min(i+1+MAXHOLD, n)
            for k in range(i+1, end):
                if l5[k] <= sl: ret = -SL*100 - COST; xt = int(t5[k]); break
                if h5[k] >= tp: ret = TP*100 - COST; xt = int(t5[k]); break
            if ret is None:
                ret = (c5[end-1]/P0-1)*100 - COST; xt = int(t5[end-1])
            outcome[i] = (ent_t, xt, ret); return outcome[i]
        for f, s in pairs:
            cf = E[f]; cs = E[s]
            cross = (cf[:-1] <= cs[:-1]) & (cf[1:] > cs[1:])
            idx = np.nonzero(cross)[0] + 1
            lst = ptr[(f, s)]
            for i in idx:
                if i < 20 or i >= n-1:
                    continue
                et = int(t5[i])
                if et < ws or et >= we:
                    continue
                o = out(int(i))
                if o is not None:
                    lst.append(o)
        del df5, E, outcome; gc.collect()

    rows = []
    for p in pairs:
        ntk, wr, ret, prof = sim_month(ptr[p])
        rows.append((p, ntk, wr, ret, prof))
    rows.sort(key=lambda r: -r[3])     # by return desc
    print(f"\n{'الزوج (سريع/بطيء)':<18}{'صفقات':>7}{'WR':>6}{'عائد%':>9}{'ربح$':>9}")
    print("-"*50)
    for (f, s), ntk, wr, ret, prof in rows:
        print(f"EMA{f}/EMA{s:<10}{ntk:>7}{wr:>5.0f}%{ret:>+8.1f}%{prof:>+8,.0f}$")
    print("-"*50)
    print("الأفضل في الأعلى (مرتّب حسب العائد). شهر واحد فقط — نوسّع الأفضل لاحقاً.")
    print("\nDONE_EMA.", flush=True)


if __name__ == "__main__":
    main()
