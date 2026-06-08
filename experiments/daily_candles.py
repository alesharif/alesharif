#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distribution of GREEN & RED DAILY candles by % move, all coins, 2024-2025."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from binance_sim import pit_universe as PIT
S,E="2024-01-01","2026-01-01"
STABLE={"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP","USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM={"PAXG","XAUT","WBTC","WBETH","BETH"}
def exc(s):
    if not s.endswith("USDT"): return True
    b=s[:-4]
    if b in STABLE or b in COMM: return True
    if any(b.endswith(t) for t in("UP","DOWN","BULL","BEAR")): return True
    return b[-2:] in("3L","3S","5L","5S")
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
raw=PIT.prefetch_universe(PIT.list_all_usdt_symbols(),ms(S),ms(E),log=lambda*a:None)
edges=[0,1,2,3,5,7,10,15,20,30,1e9]
labels=["0-1","1-2","2-3","3-5","5-7","7-10","10-15","15-20","20-30",">30"]
g=np.zeros(len(labels)); r=np.zeros(len(labels)); tot=0; green=0; red=0; nc=0
for sym,df in raw.items():
    if exc(sym): continue
    gg=df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"],unit="ms"))
    d=pd.DataFrame({"o":gg["open"].resample("D").first(),"c":gg["close"].resample("D").last()}).dropna()
    d=d[(d.index>=pd.Timestamp(S))&(d.index<pd.Timestamp(E))]
    if len(d)<30: continue
    o=d["o"].to_numpy(float); c=d["c"].to_numpy(float); m=(o>0)
    o=o[m]; c=c[m]; nc+=1; tot+=len(o)
    gm=c>o; rm=c<o; green+=int(gm.sum()); red+=int(rm.sum())
    for i in (np.digitize((c[gm]-o[gm])/o[gm]*100,edges)-1):
        if 0<=i<len(labels): g[i]+=1
    for i in (np.digitize((o[rm]-c[rm])/o[rm]*100,edges)-1):
        if 0<=i<len(labels): r[i]+=1
del raw
print(f"عملات: {nc} | شموع يومية: {tot:,} | خضراء {green:,} ({green/tot*100:.0f}%) | حمراء {red:,} ({red/tot*100:.0f}%)\n")
def table(cnt,total,title):
    print(f"##### {title} #####"); print(f"{'النسبة %':<10}{'العدد':>12}{'% منها':>10}{'تراكمي':>10}"); print("-"*42); cum=0
    for i,lab in enumerate(labels):
        sh=cnt[i]/total*100; cum+=sh; print(f"{lab:<10}{int(cnt[i]):>12,}{sh:>9.1f}%{cum:>9.1f}%")
    print()
table(g,green,"الشموع اليومية الخضراء (صعود)")
table(r,red,"الشموع اليومية الحمراء (هبوط)")
print("DONE_DAILY.")
