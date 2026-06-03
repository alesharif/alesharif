#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ORDER-FLOW at peaks — does executed order flow (CVD) LEAD the price top?

The resting order book (depth/walls/unfilled orders) is NOT available
historically — only live. But EXECUTED order flow IS (Binance aggTrades archive).
From aggTrades we derive taker-buy vs taker-sell volume and CVD (cumulative
volume delta). For each real pump (our taken trades with MFE>=10%) we:
  1) find the price peak time (from 1m highs),
  2) load aggTrades from entry to peak+60m, build CVD over 5-min buckets,
  3) check whether CVD peaks BEFORE the price peak (bearish divergence = a
     LEADING top signal) or only coincides with it (no edge over price).

If CVD reliably tops X minutes before price -> an order-flow exit could beat the
EMA. If it only coincides -> order flow adds nothing to the exit.

aggTrades cached to data/cache/aggtrades/. Run:  python experiments/order_flow_peaks.py
"""

from __future__ import annotations

import gc, io, os, sys, urllib.request, zipfile, socket
socket.setdefaulttimeout(60)
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from binance_sim import paper_bot_strategy as S          # noqa: E402
from binance_sim import portfolio_backtest as PB          # noqa: E402
from binance_sim import pit_universe as PIT               # noqa: E402
from binance_sim import hires_data as HR                  # noqa: E402

MAX_CONC = 8
FOUR = 4 * 3600 * 1000
HOLD_H = 48
FEAR = 1.15
ATR_MIN = 0.05
ADX_MIN = 40.0
MFE_MIN = 10.0                       # only analyze real pumps (peak >= +10%)
AGG_DIR = "data/cache/aggtrades"

MONTHS = {"2026-04": ("2026-04-01", "2026-05-01"),
          "2026-05": ("2026-05-01", "2026-06-01")}     # 2 rich months first


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def load_aggtrades(sym, start_ms, end_ms):
    """Return (time_ms, qty, is_sell) arrays for [start,end]. Cached per day."""
    os.makedirs(AGG_DIR, exist_ok=True)
    days = pd.date_range(pd.Timestamp(start_ms, unit="ms").floor("D"),
                         pd.Timestamp(end_ms, unit="ms").floor("D"), freq="D")
    T = []; Q = []; SELL = []
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        cf = f"{AGG_DIR}/{sym}-{ds}.npz"
        if os.path.exists(cf):
            try:
                z = np.load(cf); T.append(z["t"]); Q.append(z["q"]); SELL.append(z["s"]); continue
            except Exception:
                pass
        url = (f"https://data.binance.vision/data/spot/daily/aggTrades/"
               f"{sym}/{sym}-aggTrades-{ds}.zip")
        try:
            raw = urllib.request.urlopen(url, timeout=60).read()
            zf = zipfile.ZipFile(io.BytesIO(raw))
            txt = zf.read(zf.namelist()[0]).decode()
        except Exception:
            np.savez(cf, t=np.array([]), q=np.array([]), s=np.array([])); continue
        t = []; q = []; s = []
        for ln in txt.strip().split("\n"):
            p = ln.split(",")
            if len(p) < 7:
                continue
            t.append(int(p[5])); q.append(float(p[2]))
            s.append(1 if p[6].strip().lower() == "true" else 0)  # buyer maker => taker SELL
        t = np.array(t, dtype=np.int64); q = np.array(q); s = np.array(s, dtype=np.int8)
        np.savez(cf, t=t, q=q, s=s)
        T.append(t); Q.append(q); SELL.append(s)
    if not T:
        return None
    t = np.concatenate(T); q = np.concatenate(Q); s = np.concatenate(SELL)
    if len(t) == 0:
        return None
    # normalize microsecond timestamps (2025+) to ms
    if t.max() > 1e14:
        t = t // 1000
    m = (t >= start_ms) & (t <= end_ms)
    return t[m], q[m], s[m]


def reconstruct_trades(mname, s, e):
    start, end = parse(s), parse(e)
    ff = start - PB.WARMUP_BARS * FOUR
    raw = PIT.prefetch_universe(PIT.list_all_usdt_symbols(), ff, end, log=lambda *a: None)
    ps = {sym: S.attach_entry_signal(S.compute_features(df.copy())).set_index("close_time")
          for sym, df in raw.items()}
    times = sorted({int(t) for df in ps.values() for t in df.index if start <= t <= end})
    sr = PB.build_stable_ratio(ff, end)
    sblock = {t: (np.isfinite(v) and v > FEAR) for t, v in sr.items()}
    trades = []; open_until = {}
    for t in times:
        open_until = {sy: u for sy, u in open_until.items() if u > t}
        if sblock.get(t, False) or len(open_until) >= MAX_CONC:
            continue
        cands = []
        for sym, df in ps.items():
            if sym in open_until or t not in df.index:
                continue
            row = df.loc[t]
            if not bool(row["entry_signal"]):
                continue
            price = float(row["close"]); atr = float(row["atr"]); adx = float(row["adx"])
            if atr / price < ATR_MIN or adx < ADX_MIN:
                continue
            cands.append((sym, price, float(row["vol_pit"])))
        cands.sort(key=lambda x: x[2], reverse=True)
        for sym, price, _ in cands[:MAX_CONC - len(open_until)]:
            open_until[sym] = t + HOLD_H * 3600 * 1000
            trades.append((sym, t, price))
    del ps, raw; gc.collect()
    return trades


def main():
    leads = []          # minutes CVD-peak leads price-peak (>0 = leads/divergence)
    npump = 0
    print("ORDER-FLOW at peaks — does CVD lead the price top?\n", flush=True)
    for mname, (s, e) in MONTHS.items():
        trades = reconstruct_trades(mname, s, e)
        print(f"  {mname}: {len(trades)} trades; scanning aggTrades for pumps (MFE>={MFE_MIN}%)...", flush=True)
        for sym, t0, ep in trades:
            d = HR.load_range(sym, "1m", t0 + 1, t0 + HOLD_H * 3600 * 1000)
            if d is None or len(d) < 60:
                continue
            hi = d["high"].to_numpy(float); tm = d["time"].to_numpy()
            mfe = (hi.max() / ep - 1) * 100
            if mfe < MFE_MIN:
                continue
            npump += 1
            peak_t = int(tm[int(np.argmax(hi))])
            ag = load_aggtrades(sym, t0, peak_t + 60 * 60 * 1000)
            if ag is None or len(ag[0]) < 50:
                continue
            at, aq, asell = ag
            signed = np.where(asell == 1, -aq, aq)        # +buy taker, -sell taker
            # 5-min CVD buckets
            bucket = (at - t0) // (5 * 60 * 1000)
            order = np.argsort(bucket)
            b = bucket[order]; sv = signed[order]
            ub = np.unique(b)
            cvd = np.cumsum([sv[b == k].sum() for k in ub])
            bt = t0 + ub * 5 * 60 * 1000                  # bucket start times
            cvd_peak_t = int(bt[int(np.argmax(cvd))])
            lead_min = (peak_t - cvd_peak_t) / 60000.0    # >0 = CVD peaked before price
            leads.append(lead_min)
        gc.collect()

    if not leads:
        print("no pumps with aggTrades."); return
    L = np.array(leads)
    print(f"\n##### ORDER-FLOW vs PRICE PEAK ({len(L)} pumps, MFE>={MFE_MIN}%) #####")
    print(f"median CVD lead over price peak: {np.median(L):+.0f} min")
    print(f"mean:   {L.mean():+.0f} min")
    print(f"share where CVD peaked BEFORE price (divergence): {(L > 5).mean()*100:.0f}%")
    print(f"share where CVD ≈ coincides (±5min):              {(np.abs(L)<=5).mean()*100:.0f}%")
    print(f"share where CVD peaked AFTER price:               {(L < -5).mean()*100:.0f}%")
    print("\nصفر تقريباً للوسيط = التدفّق يرافق القمة (لا يسبقها) → لا أفضلية على السعر")
    print("موجب كبير = التدفّق يسبق القمة → إشارة خروج مبكّرة مفيدة")
    print("\nDONE_ORDERFLOW.", flush=True)


if __name__ == "__main__":
    main()
