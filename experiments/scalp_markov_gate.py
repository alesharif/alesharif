#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Markov regime GATE on the 5m scalp (from Roan's framework in the uploaded file).
Daily regime labels (20d return, +/-5%) -> walk-forward 3x3 transition matrix ->
signal = P(Bull next)-P(Bear next) from current state. Gate trades to days where the
signal is bullish (>0). Two gates: PER-COIN (coin's own regime) and BTC-MARKET.
Test on one drought month (2025-01). Compare baseline vs each gate. No lookahead.
Run: python experiments/scalp_markov_gate.py
"""

from __future__ import annotations

import gc, sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

MONTH = "2025-01"; US, UE = "2022-06-01", "2025-02-01"   # long daily history for the matrix
VLO, VHI = 2e6, 2e8
START = 2000.0; STAKE = 0.10; MAXPOS = 10
TP = 0.03; SL = 0.02; COST = 0.25; MAXHOLD = 3*24*12; WARMUP_D = 45
MK_WIN = 20; MK_THR = 0.05; MK_MIN = 252                 # Markov: 20d return, +/-5%, min train
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


# ---- Markov regime (from the uploaded file) ----
def label_regimes(close: np.ndarray, window=MK_WIN, thr=MK_THR) -> np.ndarray:
    """0=Bear, 1=Sideways, 2=Bull from rolling `window` return."""
    lab = np.ones(len(close), dtype=int)            # default Sideways
    rr = np.full(len(close), np.nan)
    rr[window:] = close[window:] / close[:-window] - 1.0
    lab[rr > thr] = 2
    lab[rr < -thr] = 0
    lab[:window] = 1
    return lab


def transition_matrix(lab: np.ndarray) -> np.ndarray:
    counts = np.zeros((3, 3))
    for i in range(len(lab) - 1):
        counts[lab[i], lab[i+1]] += 1
    rs = counts.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    return counts / rs


def markov_sig_by_day(close: np.ndarray):
    """Walk-forward signal per daily index d (labels up to d, state=lab[d]).
    signal = P[state,Bull]-P[state,Bear]. INCREMENTAL counting -> O(n)."""
    lab = label_regimes(close); n = len(lab); sig = np.full(n, np.nan)
    counts = np.zeros((3, 3))
    for d in range(1, n):
        counts[lab[d-1], lab[d]] += 1                # transition into day d
        if d >= MK_MIN:
            row = counts[lab[d]]; tot = row.sum()
            if tot > 0:
                sig[d] = (row[2] - row[0]) / tot
    return sig


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


def main():
    print(f"Markov regime gate on 5m scalp — {MONTH} (drought)\n", flush=True)
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ms(US), ms(UE), log=lambda *a: None)
    cand = {}; dclose = {}
    btc_t = btc_sig = None
    for sym, df in raw.items():
        g = df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"], unit="ms"))
        dd = pd.DataFrame({"c": g["close"].resample("D").last(), "v": (g["close"]*g["volume"]).resample("D").sum()}).dropna()
        if sym == "BTCUSDT" and len(dd) > MK_MIN+10:
            btc_t = np.array([ts.value // 10**6 for ts in dd.index]); btc_sig = markov_sig_by_day(dd["c"].to_numpy())
        if excluded(sym) or len(dd) < 40:
            continue
        trail = dd["v"].rolling(30, min_periods=10).mean()
        if not ((trail > VLO) & (trail < VHI)).any():
            continue
        dtt = np.array([ts.value // 10**6 for ts in dd.index])
        cand[sym] = (dtt, trail.to_numpy())
        dclose[sym] = (dtt, dd["c"].to_numpy())
    del raw; gc.collect()
    coins = list(cand)
    m0 = pd.Timestamp(MONTH + "-01")
    ws = int(m0.value // 10**6); we = int((m0 + pd.offsets.MonthBegin(1)).value // 10**6)
    ld = int((m0 - pd.Timedelta(days=WARMUP_D)).value // 10**6)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(ld, unit="ms"), pd.Timestamp(we, unit="ms"), freq="D")]
    print(f"عملات: {len(coins)} | BTC markov متاح: {btc_sig is not None}", flush=True)
    def _f(cd):
        try: HR.load_day(cd[0], "5m", cd[1])
        except Exception: pass
    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(_f, [(c, d) for c in coins for d in days]))

    def btc_bull(ent_t):
        if btc_sig is None:
            return True
        i = np.searchsorted(btc_t, ent_t, side="right") - 1
        return i >= 0 and np.isfinite(btc_sig[i]) and btc_sig[i] > 0

    base = []; per_coin = []; btc_gate = []
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
        dtt, trail = cand[c]; cdt, cc = dclose[c]; csig = markov_sig_by_day(cc); n = len(c5)
        for i in range(20, n-1):
            ent_t = int(t5[i])
            if ent_t < ws or ent_t >= we:
                continue
            if not (m5[i-1] <= 0 and m5[i] > 0):
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
            tr = (ent_t, xt, ret); base.append(tr)
            ci = np.searchsorted(cdt, ent_t, side="right") - 1     # coin's last completed day
            if ci >= 0 and np.isfinite(csig[ci]) and csig[ci] > 0:
                per_coin.append(tr)
            if btc_bull(ent_t):
                btc_gate.append(tr)
        del df5; gc.collect()

    print(f"\n{'الإعداد':<22}{'صفقات':>7}{'WR':>6}{'عائد%':>9}{'ربح$':>9}")
    print("-"*53)
    for lab, tr in [("أساس (بلا بوّابة)", base), ("+ بوّابة ماركوف للعملة", per_coin), ("+ بوّابة ماركوف BTC", btc_gate)]:
        ntk, wr, ret, prof = sim(tr)
        print(f"{lab:<22}{ntk:>7}{wr:>5.0f}%{ret:>+8.1f}%{prof:>+8,.0f}$")
    print("-"*53)
    print("الهدف: هل بوّابة ماركوف تكشف ضعف الجفاف وتقلّص الخسارة/ترفع WR؟")
    print("\nDONE_MARKOV.", flush=True)


if __name__ == "__main__":
    main()
