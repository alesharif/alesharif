#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OUT-OF-SAMPLE validation of the final drought MR system. SAME rules, no tuning,
on independent windows. If +8%/-12% was real (not 2025 luck) it should hold on the
2022-2023 bear (a different drought) and not blow up in 2024's alt-season.

System (frozen): daily RSI(14) crosses up from <=25 + lower-wick>=0.4 + liquid;
exit TP+15/SL-15 (60d); risk = max 2 concurrent positions @ 10% equity, rest cash.
Run:  python experiments/mr_oos_validate.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000; DAY = 86400000; MWIN = 60*24*3600*1000
S, E = "2021-06-01", "2026-06-01"
START = 10_000.0; SLOTS = 2; ALLOC = 0.10; COST = 0.01; STOP_SLIP = 0.01
TP, SL = 0.15, 0.15; OS = 25
WINDOWS = [("2022-2023 هبوط (OOS)", "2022-09-01", "2024-01-01"),
           ("2024 موسم بديل (OOS)", "2024-01-01", "2025-01-01"),
           ("2025 جفاف (مرجع)", "2025-01-01", "2026-01-01"),
           ("2026 حتى الآن (OOS)", "2026-01-01", "2026-06-01")]


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def main():
    print("OOS validation of frozen drought MR system (RSI25+wick, max 2 concurrent)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    allsig = []      # (ent_t, exit_t, rr)
    for sym, df in raw.items():
        df = df.sort_values("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="ms")
        g = df.set_index("dt")
        d = pd.DataFrame({"o": g["open"].resample("D").first(), "h": g["high"].resample("D").max(),
                          "l": g["low"].resample("D").min(), "c": g["close"].resample("D").last(),
                          "v": g["volume"].resample("D").sum(), "t": g["time"].resample("D").last()}).dropna()
        c = d["c"].to_numpy(); n = len(c)
        if n < 130:
            df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
        o = d["o"].to_numpy(); lo = d["l"].to_numpy(); hi = d["h"].to_numpy(); v = d["v"].to_numpy(); t = d["t"].to_numpy(); dv = c*v
        r = rsi(c, 14)
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(100, n):
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            if not (r[i-1] <= OS and r[i] > OS):
                continue
            rng = max(hi[i]-lo[i], 1e-12)
            if (min(o[i], c[i])-lo[i])/rng < 0.4:
                continue
            ent_t = int(t[i]); ent_px = float(c[i])
            j0 = np.searchsorted(T, ent_t, side="right"); j1 = np.searchsorted(T, ent_t+MWIN, side="right")
            rr = None; xt = None
            for k in range(j0, j1):
                if L[k] <= ent_px*(1-SL): rr = -SL-COST-STOP_SLIP; xt = int(T[k]); break
                if H[k] >= ent_px*(1+TP): rr = TP-COST; xt = int(T[k]); break
            if rr is None:
                ke = min(j1, len(C)-1); rr = (C[ke]/ent_px-1)-COST; xt = int(T[ke])
            allsig.append((ent_t, xt, rr))
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()
    allsig.sort(key=lambda x: x[0])

    def sim(w0, w1):
        equity = START; op = []; eqt = []; eqv = []; nt = 0; wins = 0
        for et, xt, rr in allsig:
            op.sort()
            while op and op[0][0] <= et:
                a, pnl = op.pop(0); equity += pnl; eqt.append(a); eqv.append(equity)
            if not (w0 <= et < w1) or len(op) >= SLOTS:
                continue
            stake = equity*ALLOC; equity -= stake; op.append((xt, stake*(1+rr))); nt += 1; wins += 1 if rr > 0 else 0
        for xt, pnl in sorted(op):
            equity += pnl; eqt.append(xt); eqv.append(equity)
        if not eqv or nt == 0:
            return None
        eqt = np.array(eqt); eqv = np.array(eqv); ordr = np.argsort(eqt); eqt = eqt[ordr]; eqv = eqv[ordr]
        s = pd.Series(eqv, index=pd.to_datetime(eqt, unit="ms")).resample("ME").last().dropna()
        rets = s.pct_change().dropna()*100
        peak = np.maximum.accumulate(eqv); dd = ((eqv-peak)/peak).min()*100
        yrs = max((w1-w0)/(365.25*DAY), 1e-9)
        ann = ((eqv[-1]/START)**(1/yrs)-1)*100
        return (eqv[-1]/START-1)*100, ann, dd, nt, wins/nt*100, (rets > 0).mean()*100 if len(rets) else float("nan")

    print(f"{'النافذة':<24}{'عائد%':>8}{'سنوي%':>8}{'تراجع':>8}{'صفقات':>7}{'win%':>6}{'موجب%':>7}")
    print("-"*68)
    for lab, a, b in WINDOWS:
        res = sim(ms(a), ms(b))
        if res is None:
            print(f"{lab:<24}{'لا صفقات':>20}"); continue
        ret, ann, dd, nt, wr, pos = res
        print(f"{lab:<24}{ret:>+7.0f}%{ann:>+7.0f}%{dd:>+7.0f}%{nt:>7}{wr:>5.0f}%{pos:>6.0f}%")
    print("-"*68)
    print("الحكم: إن صمدت 2022-2023 قرب +8%/−12% فالنظام حقيقي؛ إن انهارت فكان حظ 2025.")
    print("⚠️ متفائل بانحياز البقاء (مخفّف بالسيولة، غير مُلغى).")
    print("\nDONE_OOS.", flush=True)


if __name__ == "__main__":
    main()
