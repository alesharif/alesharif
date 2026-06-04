#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User's averaging-down idea vs the simple stop, on breakout entries.
SIMPLE : $400 at entry, TP +100% (sell at 2x), SL -10%. (90d)
DCA-DOWN: $200 at entry, +$30 at each -10% step down to -70% (no stop), TP +100%
          on the INITIAL price (sell all at 2x P0). (180d)
Return measured on the $410 committed budget. Per regime FULL/2024/2025.
NOTE: heavy survivorship bias inflates DCA (dead coins that go to -100% are absent).
Run:  python experiments/dca_down.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

LIQ_MIN = 300_000
S, E, SF = "2023-08-01", "2026-06-01", "2024-01-01"
BUDGET = 410.0


def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp()*1000)
def ema(a, n): return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy()


def main():
    print("Averaging-DOWN vs simple stop (breakout entries)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(S), ms(E), log=lambda *a: None)
    SIM = {"full": [], 2024: [], 2025: []}; DCA = {"full": [], 2024: [], 2025: []}
    dca_tp = 0; dca_deep = 0; ntot = 0
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
        T = df["time"].to_numpy(); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
        for i in range(200, n-1):
            if int(t[i]) < ms(SF):
                continue
            if not (np.isfinite(q25[i-1]) and bbw[i-1] <= q25[i-1] and bbw[i] > bbw[i-1]
                    and hist[i] > hist[i-1] > hist[i-2] and macd[i] > macd[i-1]
                    and abs(e20[i]-e50[i])/c[i] < 0.03 and c[i] > o[i]
                    and c[i] > e200[i] and i >= 99 and c[i] >= c[i-99]):
                continue
            if np.nanmean(dv[max(0, i-30):i]) <= LIQ_MIN:
                continue
            P0 = c[i]; ent_t = int(t[i]); yr = int(pd.Timestamp(ent_t, unit="ms").year)
            j0 = np.searchsorted(T, ent_t, side="right")
            # SIMPLE: $400, TP 2x / SL 0.9, 90d
            j1 = np.searchsorted(T, ent_t+90*86400000, side="right")
            sret = None
            for j in range(j0, min(j1, len(T))):
                if L[j] <= P0*0.9:
                    sret = -10.0; break
                if H[j] >= P0*2:
                    sret = +100.0; break
            if sret is None:
                sret = (C[min(j1, len(T)-1)]/P0 - 1)*100
            SIM["full"].append(sret)
            if yr in (2024, 2025): SIM[yr].append(sret)
            # DCA-DOWN: $200 + 7x$30, TP 2x P0, 180d, no stop
            j2 = np.searchsorted(T, ent_t+180*86400000, side="right")
            invested = 200.0; coins = 200.0/P0; hit = set(); exitp = None
            for j in range(j0, min(j2, len(T))):
                for k in range(1, 8):
                    lvl = P0*(1-0.1*k)
                    if k not in hit and L[j] <= lvl:
                        invested += 30.0; coins += 30.0/lvl; hit.add(k)
                if H[j] >= P0*2:
                    exitp = P0*2; break
            if exitp is None:
                exitp = C[min(j2, len(T)-1)]
            pnl = coins*exitp - invested
            dret = pnl/BUDGET*100
            DCA["full"].append(dret)
            if yr in (2024, 2025): DCA[yr].append(dret)
            ntot += 1
            if exitp >= P0*2: dca_tp += 1
            if len(hit) == 7 and exitp < P0: dca_deep += 1
        df.drop(columns=["dt"], inplace=True, errors="ignore")
    del raw; gc.collect()

    def m(st, k):
        a = np.array(st[k]); return a.mean() if len(a) else float("nan")
    print(f"{'method':<14}{'n':>7}{'FULL':>9}{'2024':>9}{'2025':>9}")
    print("-" * 48)
    print(f"{'SIMPLE (stop)':<14}{len(SIM['full']):>7}{m(SIM,'full'):>+8.1f}%{m(SIM,2024):>+8.1f}%{m(SIM,2025):>+8.1f}%")
    print(f"{'DCA-DOWN':<14}{len(DCA['full']):>7}{m(DCA,'full'):>+8.1f}%{m(DCA,2024):>+8.1f}%{m(DCA,2025):>+8.1f}%")
    print(f"\nDCA: بلغت الهدف (2x): {dca_tp/ntot*100:.0f}%   |   نشرت كل الـ7 وبقيت خاسرة: {dca_deep/ntot*100:.0f}%")
    print("⚠️ DCA متفائل بانحياز البقاء (العملات الميتة −100% غائبة). الواقع أسوأ.")
    print("\nDONE_DCA.", flush=True)


if __name__ == "__main__":
    main()
