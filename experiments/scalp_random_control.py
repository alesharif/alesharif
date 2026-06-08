#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CONTROL: random 5m entries (no indicator) with the SAME +3%/-2% bracket + cost,
same coins/month (2025-01). If indicators add no edge, random WR ~= 40% (=SL/(TP+SL))
and net ~= -cost. Proves the ~40-45% ceiling is bracket geometry + efficiency, not signal.
"""
import gc, sys, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
from binance_sim import pit_universe as PIT
from binance_sim import hires_data as HR
MONTH="2025-01"; US,UE="2024-11-01","2025-02-01"; VLO,VHI=2e6,2e8
TP=0.03; SL=0.02; COST=0.25; MAXHOLD=3*24*12; WARMUP_D=45; rng=np.random.default_rng(7)
STABLE={"USDC","FDUSD","TUSD","USDP","DAI","BUSD","USDD","EUR","EURI","AEUR","GBP","USTC","PYUSD","XUSD","EURT","BFUSD"}
COMM={"PAXG","XAUT","WBTC","WBETH","BETH"}
def msf(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
def exc(s):
    if not s.endswith("USDT"): return True
    b=s[:-4]
    if b in STABLE or b in COMM: return True
    if any(b.endswith(t) for t in("UP","DOWN","BULL","BEAR")): return True
    return b[-2:] in("3L","3S","5L","5S")
raw=PIT.prefetch_universe(PIT.list_all_usdt_symbols(),msf(US),msf(UE),log=lambda*a:None)
cand={}
for sym,df in raw.items():
    if exc(sym): continue
    g=df.sort_values("time").set_index(pd.to_datetime(df.sort_values("time")["time"],unit="ms"))
    dv=(g["close"]*g["volume"]).resample("D").sum().dropna()
    if len(dv)<20: continue
    tr=dv.rolling(30,min_periods=10).mean()
    if not ((tr>VLO)&(tr<VHI)).any(): continue
    cand[sym]=(np.array([t.value//10**6 for t in tr.index]),tr.to_numpy())
del raw; gc.collect()
coins=list(cand)
m0=pd.Timestamp(MONTH+"-01"); ws=int(m0.value//10**6); we=int((m0+pd.offsets.MonthBegin(1)).value//10**6)
ld=int((m0-pd.Timedelta(days=WARMUP_D)).value//10**6)
days=[d.strftime("%Y-%m-%d") for d in pd.date_range(pd.Timestamp(ld,unit="ms"),pd.Timestamp(we,unit="ms"),freq="D")]
def _f(cd):
    try: HR.load_day(cd[0],"5m",cd[1])
    except: pass
with ThreadPoolExecutor(max_workers=24) as ex: list(ex.map(_f,[(c,d) for c in coins for d in days]))
rets=[]
for c in coins:
    df5=HR.load_range(c,"5m",ws,we)
    if df5 is None or len(df5)<300: continue
    df5=df5.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    t5=df5["time"].to_numpy(); c5=df5["close"].to_numpy(float); h5=df5["high"].to_numpy(float); l5=df5["low"].to_numpy(float)
    n=len(c5); dtt,trail=cand[c]
    # 30 random entries per coin
    if n<MAXHOLD+5: continue
    idxs=rng.integers(0,n-MAXHOLD-1,size=30)
    for i in idxs:
        ent_t=int(t5[i]); di=np.searchsorted(dtt,ent_t,side="right")-1
        if di<0 or not (VLO<trail[di]<VHI): continue
        P0=c5[i]; tp=P0*(1+TP); sl=P0*(1-SL); ret=None; end=min(i+1+MAXHOLD,n)
        for k in range(i+1,end):
            if l5[k]<=sl: ret=-SL*100-COST; break
            if h5[k]>=tp: ret=TP*100-COST; break
        if ret is None: ret=(c5[end-1]/P0-1)*100-COST
        rets.append(ret)
    del df5; gc.collect()
r=np.array(rets); wr=(r>0).mean()*100
print(f"\nدخول عشوائي بحت — {MONTH}: {len(r)} صفقة")
print(f"  نسبة الربح (WR): {wr:.0f}%   (المتوقّع نظرياً ~40%)")
print(f"  متوسط العائد/صفقة: {r.mean():+.2f}%   (المتوقّع ~ -العمولة)")
print("DONE_RAND.")
