#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAILY EMA35 x EMA49 crossover, one coin, ~1 year. Long when EMA35 crosses above
EMA49, exit on cross below. cost 0.2% RT. vs buy&hold."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from binance_sim import pit_universe as PIT
COIN=sys.argv[1] if len(sys.argv)>1 else "INJUSDT"
S,E="2024-03-01","2025-06-01"; TS="2024-06-01"; COST=0.20
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
def ema(a,n): return pd.Series(a).ewm(span=n,adjust=False).mean().to_numpy()
raw=PIT.prefetch_universe([COIN],ms(S),ms(E),log=lambda*a:None)
df=raw[COIN].sort_values("time")
g=df.set_index(pd.to_datetime(df["time"],unit="ms"))
d=pd.DataFrame({"c":g["close"].resample("D").last(),"t":g["time"].resample("D").last()}).dropna()
c=d["c"].to_numpy(float); t=d["t"].to_numpy(); n=len(c)
e35=ema(c,35); e49=ema(c,49)
ts=ms(TS)
pos=False; entry=0.0; eq=1.0; tr=[]; ei=None
for i in range(50,n):
    if int(t[i])<ts: continue
    if not pos and e35[i-1]<=e49[i-1] and e35[i]>e49[i]:
        pos=True; entry=c[i]
    elif pos and e35[i-1]>=e49[i-1] and e35[i]<e49[i]:
        r=(c[i]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r); pos=False
if pos:
    r=(c[-1]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r)
# buy&hold over the test window
mask=t>=ts; cc=c[mask]; bh=(cc[-1]/cc[0]-1)*100 if len(cc)>1 else 0
trr=np.array(tr); wr=(trr>0).mean()*100 if len(trr) else 0
print(f"##### {COIN} | يومي | تقاطع EMA35×EMA49 | سنة #####")
print(f"  عدد الصفقات: {len(trr)}")
print(f"  نسبة الربح: {wr:.0f}%")
print(f"  متوسط الصفقة: {trr.mean() if len(trr) else 0:+.2f}%")
print(f"  العائد الكلي (مُركّب): {(eq-1)*100:+.0f}%")
print(f"  للمقارنة — احتفاظ: {bh:+.0f}%")
if len(trr): print(f"  أفضل/أسوأ: {trr.max():+.1f}% / {trr.min():+.1f}%")
print(f"  تفاصيل الصفقات: {[round(x,1) for x in tr]}")
print("DONE_D3549.")
