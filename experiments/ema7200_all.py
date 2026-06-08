#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA7200 +5% momentum entry / exit-below-EMA, on ALL coins with cached 5m data,
2024-06..2025-06. Reads cache directly (no network). Aggregates across coins."""
import os, glob, sys, datetime as dt, numpy as np, pandas as pd
HIRES="data/cache/hires"; PERIOD=7200; COST=0.20; BUF=0.05
LO=dt.date(2024,6,1); HI=dt.date(2025,6,1)
files=glob.glob(os.path.join(HIRES,"*-5m-*.csv"))
bycoin={}
for f in files:
    bn=os.path.basename(f)
    try:
        base,rest=bn.split("-5m-"); d=dt.date.fromisoformat(rest[:10].replace(".csv",""))
    except: continue
    if LO<=d<HI: bycoin.setdefault(base,[]).append((d,f))
print(f"عملات لها 5m مخزّن في النطاق: {len(bycoin)}",flush=True)
res=[]   # (coin, strat_ret, bh_ret, ntrades, wr)
for base,fs in bycoin.items():
    fs.sort()
    if len(fs)<200: continue                 # تحتاج ~سنة من الأيام
    parts=[]
    for _,f in fs:
        try: parts.append(pd.read_csv(f,usecols=["time","close"]))
        except: pass
    if not parts: continue
    df=pd.concat(parts,ignore_index=True).drop_duplicates("time").sort_values("time")
    c=df["close"].to_numpy(float); c=c[np.isfinite(c)&(c>0)]; n=len(c)
    if n<PERIOD+2000: continue
    ema=pd.Series(c).ewm(span=PERIOD,adjust=False).mean().to_numpy()
    pos=False; entry=0.0; eq=1.0; tr=[]
    for i in range(PERIOD,n):
        if not pos and c[i]>=ema[i]*(1+BUF): pos=True; entry=c[i]
        elif pos and c[i]<ema[i]:
            r=(c[i]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r); pos=False
    if pos:
        r=(c[-1]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r)
    bh=(c[-1]/c[0]-1)*100; wr=(np.array(tr)>0).mean()*100 if tr else 0
    res.append((base,(eq-1)*100,bh,len(tr),wr))
res=[x for x in res if x[3]>0]
S=np.array([x[1] for x in res]); B=np.array([x[2] for x in res]); WR=np.array([x[4] for x in res])
print(f"\nعملات مُختبَرة: {len(res)}\n")
print("##### الاستراتيجية (EMA7200 + دخول 5%) #####")
print(f"  متوسط العائد/عملة: {S.mean():+.0f}%   |   الوسيط: {np.median(S):+.0f}%")
print(f"  عملات رابحة: {(S>0).mean()*100:.0f}%   |   تتفوّق على الاحتفاظ: {(S>B).mean()*100:.0f}%")
print(f"  متوسط نسبة الربح للصفقات: {WR.mean():.0f}%")
print("##### الاحتفاظ بالعملة (buy&hold) #####")
print(f"  متوسط العائد/عملة: {B.mean():+.0f}%   |   الوسيط: {np.median(B):+.0f}%")
print("\n##### أفضل 8 عملات بالاستراتيجية #####")
for coin,s,b,nt,w in sorted(res,key=lambda x:-x[1])[:8]:
    print(f"  {coin:<12} استراتيجية {s:+6.0f}%  احتفاظ {b:+6.0f}%  ({nt} صفقة، WR {w:.0f}%)")
print("\n##### أسوأ 5 #####")
for coin,s,b,nt,w in sorted(res,key=lambda x:x[1])[:5]:
    print(f"  {coin:<12} استراتيجية {s:+6.0f}%  احتفاظ {b:+6.0f}%")
print("DONE_EALL.")
