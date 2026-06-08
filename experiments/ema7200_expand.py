#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand experiment-2 (EMA7200 +5% entry, exit on close<EMA7200, NO TP) to the
top-N liquid band coins ($2M-$200M). Downloads full year 5m. cost 0.2% RT. vs buy&hold."""
import sys, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,".")
from binance_sim import pit_universe as PIT
from binance_sim import hires_data as HR
S,E="2024-06-01","2025-06-01"; PERIOD=7200; COST=0.20; BUF=0.05; N=150
VLO,VHI=2e6,2e8
STABLE={"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP","USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM={"PAXG","XAUT","WBTC","WBETH","BETH"}
def exc(s):
    if not s.endswith("USDT"): return True
    b=s[:-4]
    if b in STABLE or b in COMM: return True
    if any(b.endswith(t) for t in("UP","DOWN","BULL","BEAR")): return True
    return b[-2:] in("3L","3S","5L","5S")
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
raw=PIT.prefetch_universe(PIT.list_all_usdt_symbols(),ms("2024-03-01"),ms("2025-07-01"),log=lambda*a:None)
vol={}
for sym,df in raw.items():
    if exc(sym): continue
    g=df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"],unit="ms"))
    dv=(g["close"]*g["volume"]).resample("D").sum().dropna()
    if len(dv)<100: continue
    md=dv.median()
    if VLO<md<VHI: vol[sym]=md
del raw
coins=sorted(vol,key=lambda s:-vol[s])[:N]
print(f"عملات النطاق: {len(vol)} | نختبر أكبر {len(coins)} سيولةً",flush=True)
days=[d.strftime("%Y-%m-%d") for d in pd.date_range(S,E,freq="D")]
print("تحميل سنة 5m...",flush=True); t0=time.time()
def _f(cd):
    try: HR.load_day(cd[0],"5m",cd[1])
    except: pass
with ThreadPoolExecutor(max_workers=24) as ex: list(ex.map(_f,[(c,d) for c in coins for d in days]))
print(f"تحميل {(time.time()-t0)/60:.1f}د",flush=True)
res=[]
for c in coins:
    df=HR.load_range(c,"5m",ms(S),ms(E))
    if df is None or len(df)<PERIOD+2000: continue
    cl=df.drop_duplicates("time").sort_values("time")["close"].to_numpy(float); cl=cl[np.isfinite(cl)&(cl>0)]
    if len(cl)<PERIOD+2000: continue
    ema=pd.Series(cl).ewm(span=PERIOD,adjust=False).mean().to_numpy()
    pos=False; entry=0.0; eq=1.0; tr=[]; n=len(cl)
    for i in range(PERIOD,n):
        if not pos and cl[i]>=ema[i]*(1+BUF): pos=True; entry=cl[i]
        elif pos and cl[i]<ema[i]:
            r=(cl[i]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r); pos=False
    if pos:
        r=(cl[-1]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r)
    bh=(cl[-1]/cl[0]-1)*100; wr=(np.array(tr)>0).mean()*100 if tr else 0
    res.append((c,(eq-1)*100,bh,len(tr),wr))
Sx=np.array([x[1] for x in res]); Bx=np.array([x[2] for x in res])
print(f"\nعملات مُختبَرة: {len(res)} | الإعداد: EMA7200 دخول +5% خروج تحت EMA (بلا TP)\n")
print("##### الاستراتيجية #####")
print(f"  متوسط العائد/عملة: {Sx.mean():+.0f}%   الوسيط: {np.median(Sx):+.0f}%")
print(f"  عملات رابحة: {(Sx>0).mean()*100:.0f}%   تتفوّق على الاحتفاظ: {(Sx>Bx).mean()*100:.0f}%")
print(f"  متوسط WR: {np.mean([x[4] for x in res]):.0f}%")
print(f"##### الاحتفاظ: متوسط {Bx.mean():+.0f}%  الوسيط {np.median(Bx):+.0f}% #####")
print("\nأفضل 10:")
for c,s,b,nt,w in sorted(res,key=lambda x:-x[1])[:10]:
    print(f"  {c:<11}{s:+6.0f}%  احتفاظ {b:+5.0f}%  ({nt}ص، WR{w:.0f}%)")
print("DONE_EXPAND.")
