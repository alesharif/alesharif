#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIAGNOSE the root cause of losing months (stop guessing filters).

Developed system (squeeze + EMA200 + liquid + SL10 + TP+100%). For EVERY month we
measure the OUTCOME ANATOMY: how many trades hit TP(+100), how many hit the stop,
how many timed out, and the single BEST trade that month. Then we split months
into POSITIVE vs NEGATIVE and compare — to find what actually drives a losing
month: is it (a) NO big winner appeared, (b) a high fakeout/stop rate, or (c)
something else? In-sample. Run:  python experiments/diagnose_months.py
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
    print("DIAGNOSE losing months — outcome anatomy (developed system)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    rows = []          # (month, net_ret, category)  category in {tp,stop,timeout}
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
            cat = None; r = None; j = j0
            while j < len(T) and T[j] <= end_t:
                if L[j] <= sl: cat, r = "stop", -SL*100; break
                if H[j] >= tp: cat, r = "tp", TP*100; break
                j += 1
            if cat is None:
                cat, r = "timeout", (C[min(j, len(T)-1)]/ent_px-1)*100
            nr = r - 1.0 - (1.0 if cat == "stop" else 0.0)
            rows.append((pd.Timestamp(ent_t, unit="ms").strftime("%Y-%m"), nr, cat))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    # per-month anatomy
    mo = {}
    for mk, nr, cat in rows:
        mo.setdefault(mk, []).append((nr, cat))
    pos_stats, neg_stats = [], []
    print(f"{'month':<9}{'n':>5}{'tp%':>6}{'stop%':>7}{'to%':>6}{'maxWin':>8}{'mean':>8}")
    for mk in sorted(mo):
        items = mo[mk]; n = len(items)
        nets = np.array([x[0] for x in items]); cats = [x[1] for x in items]
        tpp = cats.count("tp")/n*100; stp = cats.count("stop")/n*100; top = cats.count("timeout")/n*100
        mx = nets.max(); mn = nets.mean()
        tag = pos_stats if mn > 0 else neg_stats
        tag.append((n, tpp, stp, top, mx, mn))
        print(f"{mk:<9}{n:>5}{tpp:>5.0f}%{stp:>6.0f}%{top:>5.0f}%{mx:>+7.0f}%{mn:>+7.1f}%")

    def avg(stats, k):
        return np.mean([s[k] for s in stats]) if stats else float("nan")
    print(f"\n##### POSITIVE months ({len(pos_stats)}) vs NEGATIVE months ({len(neg_stats)}) #####")
    print(f"{'metric':<16}{'POSITIVE':>10}{'NEGATIVE':>10}")
    for k, name in [(1, "tp-hit %"), (2, "stop %"), (3, "timeout %"),
                    (4, "best trade %"), (0, "n trades")]:
        print(f"{name:<16}{avg(pos_stats, k):>+9.1f}{avg(neg_stats, k):>+10.1f}")
    print("\nالفرق الأكبر بين العمودين = السبب الجذري للأشهر الخاسرة.")
    print("\nDONE_DIAG.", flush=True)


if __name__ == "__main__":
    main()
