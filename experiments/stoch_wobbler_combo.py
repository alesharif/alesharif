#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COMBINED: monthly stochastic gate + Wobbler daily entry (user's full workflow).

The user's real workflow: the MONTHLY screener flags a coin when monthly
stochastic is a deep-oversold turn (%K>%D & %D<=10); THEN he watches that coin on
the daily for the Wobbler turn-up to enter. So we gate the Wobbler daily entries:
only take them while the coin is within GATE_MONTHS of a monthly oversold-turn.
SL 10%, TP +50% first-touch. We run BOTH in-sample (2022-2026) and 2024 OOS, so
we see immediately whether the monthly gate creates a durable edge.

Run:  python experiments/stoch_wobbler_combo.py [2024]
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402

OOS = "2024" in sys.argv
START = "2023-09-01" if OOS else "2022-06-01"
END = "2025-01-01" if OOS else "2026-06-01"
SIG_FROM = int(pd.Timestamp("2024-01-01" if OOS else START, tz="UTC").timestamp()*1000)
DS = int(pd.Timestamp(START, tz="UTC").timestamp()*1000)
DE = int(pd.Timestamp(END, tz="UTC").timestamp()*1000)
SL = 0.10; TP = 0.50; WINDOW_MS = 90*24*3600*1000
GATE_MONTHS = 3
MONTH_MS = 31*24*3600*1000


def resample(df, rule):
    g = df.set_index("dt")
    return pd.DataFrame({"h": g["high"].resample(rule).max(),
                         "l": g["low"].resample(rule).min(),
                         "c": g["close"].resample(rule).last(),
                         "t": g["time"].resample(rule).last()}).dropna()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    return 100 - 100/(1 + ru/np.where(rd == 0, 1e-9, rd))


def stoch1433(h, l, c):
    ll = pd.Series(l).rolling(14).min(); hh = pd.Series(h).rolling(14).max()
    rawk = 100*(c-ll)/(hh-ll).replace(0, np.nan)
    k = rawk.rolling(3).mean(); d = k.rolling(3).mean()
    return k.to_numpy(), d.to_numpy()


def main():
    tag = "2024 OOS" if OOS else "in-sample 2022-2026"
    print(f"COMBO: monthly stoch gate + Wobbler daily entry ({tag})\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), DS, DE, log=lambda *a: None)

    def collect(use_gate):
        outs = []
        for sym, df in raw.items():
            df = df.sort_values("time").reset_index(drop=True)
            df["dt"] = pd.to_datetime(df["time"], unit="ms")
            mo = resample(df, "ME"); d = resample(df, "D")
            if len(mo) < 18 or len(d) < 110:
                df.drop(columns=["dt"], inplace=True, errors="ignore"); continue
            # monthly stoch gate: deep-oversold turn month-close times
            mk, md = stoch1433(mo["h"].to_numpy(), mo["l"].to_numpy(), mo["c"].to_numpy())
            mt = mo["t"].to_numpy()
            gate_times = [int(mt[i]) for i in range(16, len(mk))
                          if np.isfinite(mk[i]) and np.isfinite(md[i]) and mk[i] > md[i] and md[i] <= 10]
            gate_times = np.array(sorted(gate_times))
            # Wobbler daily entries (SMA of RSI 54/81, cross up from low zone)
            dc = d["c"].to_numpy(); dt = d["t"].to_numpy()
            r = rsi(dc, 14)
            maf = pd.Series(r).rolling(54).mean().to_numpy(); mas = pd.Series(r).rolling(81).mean().to_numpy()
            t = df["time"].to_numpy(); H = df["high"].to_numpy(float)
            L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
            for i in range(82, len(dc)):
                cross = (maf[i] > mas[i]) and (maf[i-1] <= mas[i-1]) and (mas[i] < 45)
                if not (cross and np.isfinite(mas[i]) and int(dt[i]) >= SIG_FROM):
                    continue
                if use_gate:
                    if len(gate_times) == 0:
                        continue
                    p = np.searchsorted(gate_times, int(dt[i]), side="right")
                    if p == 0 or (int(dt[i]) - gate_times[p-1]) > GATE_MONTHS*MONTH_MS:
                        continue
                ent_t, ent_px = int(dt[i]), float(dc[i])
                j0 = np.searchsorted(t, ent_t)
                if j0 >= len(t):
                    continue
                tp_px = ent_px*(1+TP); sl_px = ent_px*(1-SL); end_t = ent_t+WINDOW_MS
                tagx = None; j = j0
                while j < len(t) and t[j] <= end_t:
                    if L[j] <= sl_px: tagx = ("stop", -SL*100); break
                    if H[j] >= tp_px: tagx = ("tp", TP*100); break
                    j += 1
                if tagx is None:
                    tagx = ("to", (C[min(j, len(t)-1)]/ent_px-1)*100)
                outs.append(tagx)
            df.drop(columns=["dt"], inplace=True, errors="ignore")
        return outs

    for use_gate, name in [(False, "Wobbler ONLY (no monthly gate)"),
                           (True, "Wobbler + MONTHLY stoch gate")]:
        o = collect(use_gate); n = len(o)
        if n == 0:
            print(f"### {name}: no signals\n"); continue
        tp = sum(1 for k, _ in o if k == "tp")
        def exp(cost, ss):
            return np.mean([v - (ss if k == "stop" else 0) - cost for k, v in o])
        print(f"### {name}: {n} signals, TP+50% hit {tp/n*100:.0f}% ###")
        print(f"   @fee0.2%: {exp(0.2,0):+.1f}%   @realistic(1%+1%): {exp(1.0,1.0):+.1f}%   "
              f"@harsh(2%+1.5%): {exp(2.0,1.5):+.1f}%\n", flush=True)
    del raw; gc.collect()
    print("نقارن: هل البوّابة الشهرية ترفع التوقّع وتنجّيه خارج العيّنة؟")
    print("\nDONE_COMBO.", flush=True)


if __name__ == "__main__":
    main()
