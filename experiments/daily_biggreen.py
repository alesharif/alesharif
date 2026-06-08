#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Big-green-day continuation: enter on a daily close up >=THR while close>EMA50 (uptrend)
+ liquid; exit on trailing 20% from peak close (no TP). Daily, all coins. Compare a BULL
window (2023-10..2024-06) vs BEAR (2024-06..2025-06). Entry thresholds +7/+10/+15%."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from binance_sim import pit_universe as PIT
COST=0.20; TRAIL=0.20; LIQ=3e5
THRS=[0.07,0.10,0.15]
WINS=[("صاعدة 2023-10→2024-06","2023-10-01","2024-06-01"),
      ("هابطة 2024-06→2025-06","2024-06-01","2025-06-01")]
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
raw=PIT.prefetch_universe(PIT.list_all_usdt_symbols(),ms("2023-07-01"),ms("2025-06-01"),log=lambda*a:None)
data={}
for sym,df in raw.items():
    if exc(sym): continue
    g=df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"],unit="ms"))
    d=pd.DataFrame({"c":g["close"].resample("D").last(),"v":(g["close"]*g["volume"]).resample("D").sum(),"t":g["time"].resample("D").last()}).dropna()
    if len(d)<80: continue
    data[sym]=(d["c"].to_numpy(float),d["v"].to_numpy(float),d["t"].to_numpy())
del raw
print(f"عملات: {len(data)} | شراء يوم أخضر كبير + خروج تريل 20%\n",flush=True)

def run(c,v,t,e50,thr,ws,we):
    n=len(c); pos=False; entry=0.0; peak=0.0; eq=1.0; tr=[]
    for i in range(50,n):
        if int(t[i])<ws or int(t[i])>=we: 
            if pos and int(t[i])>=we: pass
            else: continue
        if not pos:
            ret1=c[i]/c[i-1]-1
            if ret1>=thr and c[i]>e50[i] and np.nanmean(v[max(0,i-30):i])>LIQ:
                pos=True; entry=c[i]; peak=c[i]
        else:
            peak=max(peak,c[i])
            if c[i]<=peak*(1-TRAIL):
                r=(c[i]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r); pos=False
    if pos:
        r=(c[-1]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r)
    wr=(np.array(tr)>0).mean()*100 if tr else 0
    return (eq-1)*100,len(tr),wr

for wlab,a,b in WINS:
    ws,we=ms(a),ms(b)
    print(f"===== {wlab} =====")
    print(f"{'عتبة اليوم':<12}{'متوسط':>8}{'وسيط':>8}{'رابحة%':>8}{'WR':>6}{'صفقات':>8}")
    for thr in THRS:
        R=[];N=[];W=[]
        for sym,(c,v,t) in data.items():
            e50=ema(c,50)
            ret,nt,wr=run(c,v,t,e50,thr,ws,we)
            if nt>0: R.append(ret);N.append(nt);W.append(wr)
        R=np.array(R)
        if len(R):
            print(f"+{int(thr*100)}%{'':<9}{R.mean():>+7.0f}%{np.median(R):>+7.0f}%{(R>0).mean()*100:>7.0f}%{np.mean(W):>5.0f}%{np.mean(N):>7.1f}  (عملات نشطة {len(R)})")
    print()
print("DONE_BG.")
