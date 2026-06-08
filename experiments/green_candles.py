#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distribution of GREEN 5m candles (close>open) by % rise, all coins, one month (2025-01)."""
import gc, sys, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,".")
from binance_sim import pit_universe as PIT
from binance_sim import hires_data as HR
MONTH="2025-01"; STABLE={"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP","USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM={"PAXG","XAUT","WBTC","WBETH","BETH"}
def exc(s):
    if not s.endswith("USDT"): return True
    b=s[:-4]
    if b in STABLE or b in COMM: return True
    if any(b.endswith(t) for t in("UP","DOWN","BULL","BEAR")): return True
    return b[-2:] in("3L","3S","5L","5S")
coins=[s for s in PIT.list_all_usdt_symbols() if not exc(s)]
m0=pd.Timestamp(MONTH+"-01"); ws=int(m0.value//10**6); we=int((m0+pd.offsets.MonthBegin(1)).value//10**6)
days=[d.strftime("%Y-%m-%d") for d in pd.date_range(m0,pd.Timestamp(we,unit="ms"),freq="D")]
print(f"عملات: {len(coins)} | تحميل 5m للشهر...",flush=True)
def _f(cd):
    try: HR.load_day(cd[0],"5m",cd[1])
    except: pass
with ThreadPoolExecutor(max_workers=24) as ex: list(ex.map(_f,[(c,d) for c in coins for d in days]))
edges=[0,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5,10,1e9]
labels=["0-0.1","0.1-0.2","0.2-0.3","0.3-0.5","0.5-0.75","0.75-1","1-1.5","1.5-2","2-3","3-5","5-10",">10"]
cnt=np.zeros(len(labels)); tot_candles=0; green=0; ncoins=0
for c in coins:
    df=HR.load_range(c,"5m",ws,we)
    if df is None or len(df)<100: continue
    df=df.drop_duplicates("time")
    o=df["open"].to_numpy(float); cl=df["close"].to_numpy(float)
    tot_candles+=len(o); ncoins+=1
    g=cl>o; green+=int(g.sum())
    pct=(cl[g]-o[g])/o[g]*100
    idx=np.digitize(pct,edges)-1
    for i in idx:
        if 0<=i<len(labels): cnt[i]+=1
gc.collect()
print(f"\nعملات بها بيانات: {ncoins} | إجمالي الشموع: {tot_candles:,} | خضراء: {green:,} ({green/tot_candles*100:.0f}%)\n")
print(f"{'نسبة الصعود %':<12}{'عدد الشموع':>14}{'% من الخضراء':>14}{'تراكمي %':>12}")
print("-"*52)
cum=0
for i,lab in enumerate(labels):
    sh=cnt[i]/green*100; cum+=sh
    print(f"{lab:<12}{int(cnt[i]):>14,}{sh:>13.2f}%{cum:>11.1f}%")
print("-"*52)
print(f"وسيط صعود الشمعة الخضراء ~ يُقرأ من الجدول التراكمي عند 50%")
print("DONE_GREEN.")
