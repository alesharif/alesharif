#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAILY EMA35 x EMA49 crossover on ALL coins, ~1 year (2024-06..2025-06). Long on cross
above, exit on cross below. cost 0.2% RT. Aggregates vs buy&hold."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from binance_sim import pit_universe as PIT
S,E="2023-06-01","2025-06-01"; TS="2024-06-01"; COST=0.20; LIQ=3e5
STABLE={"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP","USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM={"PAXG","XAUT","WBTC","WBETH","BETH"}
def exc(s):
    if not s.endswith("USDT"): return True
    b=s[:-4]
    if b in STABLE or b in COMM: return True
    if any(b.endswith(t) for t in("UP","DOWN","BULL","BEAR")): return True
    return b[-2:] in("3L","3S","5L","5S")
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
def ema(a,n): return pd.Series(a).ewm(span=n,adjust=False).mean().to_numpy()
raw=PIT.prefetch_universe(PIT.list_all_usdt_symbols(),ms(S),ms(E),log=lambda*a:None)
ts=ms(TS); res=[]
for sym,df in raw.items():
    if exc(sym): continue
    g=df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"],unit="ms"))
    d=pd.DataFrame({"c":g["close"].resample("D").last(),"v":(g["close"]*g["volume"]).resample("D").sum(),"t":g["time"].resample("D").last()}).dropna()
    c=d["c"].to_numpy(float); t=d["t"].to_numpy(); v=d["v"].to_numpy(); n=len(c)
    if n<300 or np.nanmedian(v)<LIQ: continue
    e35=ema(c,210); e49=ema(c,252)
    pos=False; entry=0.0; eq=1.0; tr=[]
    for i in range(260,n):
        if int(t[i])<ts: continue
        if not pos and e35[i-1]<=e49[i-1] and e35[i]>e49[i]: pos=True; entry=c[i]
        elif pos and e35[i-1]>=e49[i-1] and e35[i]<e49[i]:
            r=(c[i]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r); pos=False
    if pos:
        r=(c[-1]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r)
    mask=t>=ts; cc=c[mask]
    if len(cc)<2: continue
    bh=(cc[-1]/cc[0]-1)*100; wr=(np.array(tr)>0).mean()*100 if tr else 0
    res.append(((eq-1)*100,bh,len(tr),wr))
del raw
R=np.array([x[0] for x in res]); B=np.array([x[1] for x in res]); WRa=np.array([x[3] for x in res]); NT=np.array([x[2] for x in res])
print(f"\nعملات: {len(res)} | يومي EMA210×EMA252 | 2024-06→2025-06\n")
print("##### الاستراتيجية #####")
print(f"  متوسط العائد/عملة: {R.mean():+.0f}%   الوسيط: {np.median(R):+.0f}%")
print(f"  عملات رابحة: {(R>0).mean()*100:.0f}%   تتفوّق على الاحتفاظ: {(R>B).mean()*100:.0f}%")
print(f"  متوسط WR: {np.nanmean(WRa):.0f}%   متوسط الصفقات/عملة: {NT.mean():.1f}")
print(f"##### الاحتفاظ: متوسط {B.mean():+.0f}%  الوسيط {np.median(B):+.0f}% #####")
print("DONE_ALL210.")
