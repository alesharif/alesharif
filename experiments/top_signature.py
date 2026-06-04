#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REVERSE signal-mining: what RECURS at real tops (vs mid-trend)?

The user's method (inverted from everything before): instead of ASSUMING a
signal and testing it forward, we look at every REAL reversal-from-top and ask
the data which indicators recur there — AND, critically, how often the SAME
indicator fires mid-trend where we should NOT exit (the false-alarm rate the
user asked for: "نسبة ظهورها الصحيح ونسبة ظهورها الخاطئ").

For each real pump (taken trade, MFE>=MFE_MIN%) on 1h candles we label moments:
  * TOP   (🔴): the peak candle after which price drops >= DROP% and never
                reclaims the high within the window  -> a true reversal.
  * HOLD  (🟢): mid-uptrend candles whose forward return over FWD_H is still
                >= CONT% (price keeps climbing) -> "do NOT exit here" (control).

At every labelled candle we compute ~16 boolean signals, then for each signal:
  recall = P(signal | TOP)     -> نسبة الظهور الصحيح
  fpr    = P(signal | HOLD)    -> نسبة الظهور الخاطئ (false alarm)
  lift   = recall / fpr        -> how much more it marks a top than a hold
  prec   = TP/(TP+FP) at the sampled hold:top ratio
A good "top mark" = high recall AND low fpr (lift >> 1).

Run:  python experiments/top_signature.py
"""

from __future__ import annotations

import gc, sys
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
MFE_MIN = 10.0          # only analyze real pumps (peak >= +10%)

DROP = 8.0             # a TOP must be followed by >= this % drawdown
FWD_H = 6              # forward window (hours) used for labels
CONT = 3.0            # a HOLD (control) keeps climbing >= this % over FWD_H
WARM_H = 220          # 1h warmup bars before entry for indicators

_ALL_MONTHS = {"2025-12": ("2025-12-01", "2026-01-01"),
               "2026-01": ("2026-01-01", "2026-02-01"),
               "2026-02": ("2026-02-01", "2026-03-01"),
               "2026-03": ("2026-03-01", "2026-04-01"),
               "2026-04": ("2026-04-01", "2026-05-01"),
               "2026-05": ("2026-05-01", "2026-06-01")}
# subset selectable via CLI:  python experiments/top_signature.py 2026-04 2026-05
MONTHS = {k: _ALL_MONTHS[k] for k in (sys.argv[1:] or _ALL_MONTHS)}


def parse(d):
    return int(pd.Timestamp(d, tz="UTC").timestamp() * 1000)


def reconstruct_trades(s, e):
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


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    rs = ru / np.where(rd == 0, 1e-9, rd)
    return 100 - 100 / (1 + rs)


def compute_signals(df):
    """Return a dict name->bool-array over all candles of df (1h)."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    n = len(c)
    rsi = _rsi(c, 14)
    ema9 = _ema(c, 9); ema20 = _ema(c, 20); ema50 = _ema(c, 50)
    ema12 = _ema(c, 12); ema26 = _ema(c, 26)
    macd = ema12 - ema26; sig = _ema(macd, 9); hist = macd - sig
    sma20 = pd.Series(c).rolling(20).mean().to_numpy()
    std20 = pd.Series(c).rolling(20).std().to_numpy()
    upper = sma20 + 2 * std20; lower = sma20 - 2 * std20
    bbpb = (c - lower) / np.where((upper - lower) == 0, 1e-9, upper - lower)
    volma = pd.Series(v).rolling(20).mean().to_numpy()
    rng = np.where((h - l) == 0, 1e-9, h - l)
    uwick = (h - np.maximum(o, c)) / rng
    roc6 = np.concatenate([np.zeros(6), c[6:] / c[:-6] - 1])

    def sh(a, k):  # shift forward by k (a[i-k]); fill with nan
        out = np.full_like(a, np.nan, dtype=float); out[k:] = a[:-k]; return out

    prev_h = sh(h, 1); prev_l = sh(l, 1); prev_c = sh(c, 1); prev_o = sh(o, 1)
    green = c > o

    S_ = {}
    S_["rsi>70"] = rsi > 70
    S_["rsi>78"] = rsi > 78
    # bearish RSI divergence: price higher-high vs 5 bars ago, RSI lower-high
    hh = c > sh(c, 5); rsi_lower = rsi < sh(rsi, 5)
    S_["rsi_bear_div"] = hh & rsi_lower & (rsi > 60)
    S_["close<ema9"] = c < ema9
    S_["close<ema20"] = c < ema20           # our current exit
    S_["ext>15%ema20"] = (c - ema20) / ema20 > 0.15
    S_["bb_%b>1"] = bbpb > 1.0
    S_["uwick>0.5"] = uwick > 0.5            # rejection wick (user's idea)
    S_["bear_engulf"] = (c < o) & (prev_c > prev_o) & (c < prev_o) & (o >= prev_c)
    S_["red_after_3green"] = (c < o) & sh(green, 1).astype(bool) & sh(green, 2).astype(bool) & sh(green, 3).astype(bool)
    S_["vol_climax>3x"] = v > 3 * volma
    # volume divergence: new 5-bar high price but volume below its 5-bar-ago value
    S_["vol_div"] = hh & (v < sh(v, 5))
    S_["macd_hist<0"] = hist < 0
    S_["macd_hist_flip"] = (hist < 0) & (sh(hist, 1) >= 0)
    S_["roc_decel"] = (roc6 < sh(roc6, 1)) & (sh(roc6, 1) > 0.05)
    S_["lower_high"] = (h < prev_h) & (sh(h, 1) > sh(h, 2))
    S_["close<prev_low"] = c < prev_l
    S_["3 lower closes"] = (c < prev_c) & (prev_c < sh(c, 2)) & (sh(c, 2) < sh(c, 3))
    return S_, n


def main():
    print("REVERSE signal-mining: what recurs at REAL tops vs mid-trend?\n", flush=True)
    print(f"  rules: TOP=peak then drop>={DROP}% & no reclaim; "
          f"HOLD=mid-trend, +{CONT}% more over next {FWD_H}h\n", flush=True)

    # store full boolean rows so we can do combos + per-month validation
    names = None
    top_rows = []; hold_rows = []
    top_mon = []; hold_mon = []
    n_pump = 0

    for mname, (s, e) in MONTHS.items():
        trades = reconstruct_trades(s, e)
        print(f"  {mname}: {len(trades)} trades; labelling pumps...", flush=True)
        for sym, t0, ep in trades:
            df = HR.load_range(sym, "1h", t0 - WARM_H * 3600 * 1000,
                               t0 + HOLD_H * 3600 * 1000)
            if df is None or len(df) < WARM_H + 10:
                continue
            tm = df["time"].to_numpy()
            hi = df["high"].to_numpy(float)
            cl = df["close"].to_numpy(float)
            # episode window indices (entry .. entry+HOLD)
            epi = np.where((tm >= t0) & (tm <= t0 + HOLD_H * 3600 * 1000))[0]
            if len(epi) < 8:
                continue
            ep_hi = hi[epi]
            mfe = (ep_hi.max() / ep - 1) * 100
            if mfe < MFE_MIN:
                continue
            n_pump += 1
            sigs, _ = compute_signals(df)
            if names is None:
                names = list(sigs.keys())
            peak_i = epi[int(np.argmax(ep_hi))]
            peak_px = hi[peak_i]
            fwd = max(1, FWD_H)
            # ---- label each candle in episode up to peak ----
            for i in epi:
                if i <= 5 or i + fwd >= len(cl):
                    continue
                fwd_lo = cl[i + 1:i + 1 + fwd].min() if i + 1 < len(cl) else cl[i]
                fwd_hi = hi[i + 1:i + 1 + fwd].max() if i + 1 < len(cl) else hi[i]
                # TOP: this candle is the peak (or within 0.5% of it) AND big drop follows AND no reclaim
                is_peak = hi[i] >= peak_px * 0.995
                drops = (1 - fwd_lo / hi[i]) * 100 >= DROP
                no_reclaim = hi[i + 1:].max() < hi[i] * 1.005 if i + 1 < len(hi) else True
                if is_peak and drops and no_reclaim:
                    label = "TOP"
                elif (fwd_hi / cl[i] - 1) * 100 >= CONT and i < peak_i:
                    label = "HOLD"
                else:
                    continue
                vec = [bool(sigs[k][i]) and np.isfinite(sigs[k][i]) for k in names]
                if label == "TOP":
                    top_rows.append(vec); top_mon.append(mname)
                else:
                    hold_rows.append(vec); hold_mon.append(mname)
        gc.collect()

    if not top_rows or not hold_rows:
        print("not enough labelled points."); return

    TM = np.array(top_rows, dtype=bool)        # (n_top, k)
    HM = np.array(hold_rows, dtype=bool)       # (n_hold, k)
    tmon = np.array(top_mon); hmon = np.array(hold_mon)
    n_top, n_hold = len(TM), len(HM)
    base = n_top / (n_top + n_hold) * 100

    recall = TM.mean(0); fpr = HM.mean(0)
    prec = TM.sum(0) / np.maximum(TM.sum(0) + HM.sum(0), 1)
    lift = recall / np.maximum(fpr, 1e-6)

    order = np.argsort(-lift)
    print(f"\n##### TOP SIGNATURE — {n_pump} pumps, {n_top} TOP candles, "
          f"{n_hold} HOLD candles (base rate {base:.0f}%) #####")
    print(f"{'signal':<18}{'recall':>8}{'fpr':>8}{'lift':>7}{'prec':>7}")
    print(f"{'(الإشارة)':<18}{'صحيح':>8}{'خاطئ':>8}{'×':>7}{'دقة':>7}")
    print("-" * 50)
    for j in order:
        print(f"{names[j]:<18}{recall[j]*100:>7.0f}%{fpr[j]*100:>7.0f}%"
              f"{lift[j]:>6.1f}x{prec[j]*100:>6.0f}%")

    # ---- best 2- and 3-signal AND-combinations (precision via lift) ----
    import itertools
    k = len(names)
    combos = []
    for r in (2, 3):
        for cc in itertools.combinations(range(k), r):
            tc = np.all(TM[:, cc], axis=1); hc = np.all(HM[:, cc], axis=1)
            rec = tc.mean()
            if rec < 0.12:                      # must catch >=12% of tops to matter
                continue
            fp = hc.mean(); lf = rec / max(fp, 1e-6)
            pr = tc.sum() / max(tc.sum() + hc.sum(), 1)
            combos.append((lf, rec, fp, pr, cc))
    combos.sort(reverse=True)
    print(f"\n##### BEST COMBINATIONS (AND), recall>=12% #####")
    print(f"{'combo':<40}{'recall':>8}{'fpr':>7}{'lift':>7}{'prec':>7}")
    for lf, rec, fp, pr, cc in combos[:12]:
        label = " + ".join(names[i] for i in cc)
        print(f"{label:<40}{rec*100:>7.0f}%{fp*100:>6.0f}%{lf:>6.1f}x{pr*100:>6.0f}%")

    # ---- per-month validation of the top-5 single marks (overfit guard) ----
    print(f"\n##### PER-MONTH lift of top-5 singles (does it hold out-of-sample?) #####")
    months = sorted(set(top_mon) | set(hold_mon))
    head = "signal".ljust(18) + "".join(m[-2:].rjust(7) for m in months)
    print(head)
    for j in order[:5]:
        cells = ""
        for m in months:
            tm_m = TM[tmon == m, j]; hm_m = HM[hmon == m, j]
            if len(tm_m) == 0 or len(hm_m) == 0:
                cells += "   -  "; continue
            lf = tm_m.mean() / max(hm_m.mean(), 1e-6)
            cells += f"{lf:>6.1f}x"
        print(names[j].ljust(18) + cells)

    print("\nالمفيد = recall عالٍ + fpr منخفض (lift كبير). "
          "lift~1 يعني الإشارة تظهر في الاستمرار بنفس قدر القمة → عديمة الفائدة.")
    print("ثبات lift عبر الأشهر = إشارة حقيقية؛ تذبذبه = صدفة/إفراط بالعيّنة.")
    print("\nDONE_TOPSIG.", flush=True)


if __name__ == "__main__":
    main()
