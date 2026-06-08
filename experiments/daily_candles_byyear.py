#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GREEN & RED daily candle distribution by % move, all coins, 2024 and 2025 SEPARATELY."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from binance_sim import pit_universe as PIT
STABLE={"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP","USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM={"PAXG","XAUT","WBTC","WBETH","BETH"}
def exc(s):
    if not s.endswith("USDT"): return True
    b=s[:-4]
    if b in STABLE or b in COMM: return True
    if any(b.endswith(t) for t in("UP","DOWN","BULL","BEAR")): return True
    return b[-2:] in("3L","3S","5L","5S")
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
raw=PIT.prefetch_universe(PIT.list_all_usdt_symbols(),ms("2024-01-01"),ms("2026-01-01"),log=lambda*a:None)
edges=[0,1,2,3,5,7,10,15,20,30,1e9]
labels=["0-1","1-2","2-3","3-5","5-7","7-10","10-15","15-20","20-30",">30"]
acc={2024:{"g":np.zeros(len(labels)),"r":np.zeros(len(labels)),"green":0,"red":0,"tot":0,"nc":set()},
     2025:{"g":np.zeros(len(labels)),"r":np.zeros(len(labels)),"green":0,"red":0,"tot":0,"nc":set()}}
for sym,df in raw.items():
    if exc(sym): continue
    gg=df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"],unit="ms"))
    d=pd.DataFrame({"o":gg["open"].resample("D").first(),"c":gg["close"].resample("D").last()}).dropna()
    for yr in (2024,2025):
        dy=d[(d.index>=pd.Timestamp(f"{yr}-01-01"))&(d.index<pd.Timestamp(f"{yr+1}-01-01"))]
        if len(dy)<20: continue
        o=dy["o"].to_numpy(float); c=dy["c"].to_numpy(float); m=(o>0); o=o[m]; c=c[m]
        A=acc[yr]; A["nc"].add(sym); A["tot"]+=len(o)
        gm=c>o; rm=c<o; A["green"]+=int(gm.sum()); A["red"]+=int(rm.sum())
        for i in (np.digitize((c[gm]-o[gm])/o[gm]*100,edges)-1):
            if 0<=i<len(labels): A["g"][i]+=1
        for i in (np.digitize((o[rm]-c[rm])/o[rm]*100,edges)-1):
            if 0<=i<len(labels): A["r"][i]+=1
del raw
def table(cnt,total,title):
    print(f"  {title}"); print(f"  {'النسبة':<9}{'العدد':>11}{'%':>8}{'تراكمي':>9}"); cum=0
    for i,lab in enumerate(labels):
        sh=cnt[i]/total*100 if total else 0; cum+=sh; print(f"  {lab:<9}{int(cnt[i]):>11,}{sh:>7.1f}%{cum:>8.1f}%")
    print()
for yr in (2024,2025):
    A=acc[yr]
    print(f"\n========== {yr} ==========")
    print(f"عملات: {len(A['nc'])} | شموع: {A['tot']:,} | خضراء {A['green']:,} ({A['green']/A['tot']*100:.0f}%) | حمراء {A['red']:,} ({A['red']/A['tot']*100:.0f}%)\n")
    table(A["g"],A["green"],"خضراء (صعود)")
    table(A["r"],A["red"],"حمراء (هبوط)")
print("DONE_BYYEAR.")
