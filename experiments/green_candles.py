#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distribution of GREEN & RED 5m candles by % move, all coins WITH cached data, 2025-01.
Reads cached CSVs directly (no network) to avoid dead-coin timeouts."""
import os, glob, numpy as np, pandas as pd

HIRES = "data/cache/hires"
MONTH = "2025-01"
files = glob.glob(os.path.join(HIRES, f"*-5m-{MONTH}-*.csv"))
coins = {}
for f in files:
    base = os.path.basename(f).split("-5m-")[0]
    coins.setdefault(base, []).append(f)
print(f"عملات لها بيانات 5m مخزّنة في {MONTH}: {len(coins)}", flush=True)

edges = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 1e9]
labels = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.75", "0.75-1",
          "1-1.5", "1.5-2", "2-3", "3-5", "5-10", ">10"]
gcnt = np.zeros(len(labels)); rcnt = np.zeros(len(labels))
tot = 0; green = 0; red = 0; nc = 0
for base, fs in coins.items():
    parts = []
    for f in sorted(fs):
        try:
            parts.append(pd.read_csv(f, usecols=["open", "close"]))
        except Exception:
            pass
    if not parts:
        continue
    df = pd.concat(parts, ignore_index=True)
    o = df["open"].to_numpy(float); cl = df["close"].to_numpy(float)
    m = np.isfinite(o) & np.isfinite(cl) & (o > 0); o = o[m]; cl = cl[m]
    if len(o) < 50:
        continue
    nc += 1; tot += len(o)
    gm = cl > o; rm = cl < o; green += int(gm.sum()); red += int(rm.sum())
    gp = (cl[gm] - o[gm]) / o[gm] * 100
    rp = (o[rm] - cl[rm]) / o[rm] * 100
    for i in (np.digitize(gp, edges) - 1):
        if 0 <= i < len(labels):
            gcnt[i] += 1
    for i in (np.digitize(rp, edges) - 1):
        if 0 <= i < len(labels):
            rcnt[i] += 1
print(f"عملات معالَجة: {nc} | إجمالي الشموع: {tot:,} | خضراء {green:,} ({green/tot*100:.0f}%) | حمراء {red:,} ({red/tot*100:.0f}%)\n")


def table(cnt, total, title):
    print(f"##### {title} #####")
    print(f"{'النسبة %':<12}{'العدد':>14}{'% منها':>10}{'تراكمي':>10}")
    print("-" * 46)
    cum = 0
    for i, lab in enumerate(labels):
        sh = cnt[i] / total * 100 if total else 0; cum += sh
        print(f"{lab:<12}{int(cnt[i]):>14,}{sh:>9.2f}%{cum:>9.1f}%")
    print()


table(gcnt, green, "الشموع الخضراء (نسبة الصعود)")
table(rcnt, red, "الشموع الحمراء (نسبة الهبوط)")
print("DONE_GREEN.")
