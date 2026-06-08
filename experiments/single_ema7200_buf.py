#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same EMA7200 system on one coin, but ENTER only after price is +5% ABOVE EMA7200
(momentum confirmation), exit when price < EMA7200. 5m, 1yr, cost 0.2% RT. vs buy&hold."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from binance_sim import hires_data as HR
COIN=sys.argv[1] if len(sys.argv)>1 else "INJUSDT"
S,E="2024-06-01","2025-06-01"; PERIOD=7200; COST=0.20; BUF=0.05
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
df=HR.load_range(COIN,"5m",ms(S),ms(E))   # cached from previous run
df=df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
c=df["close"].to_numpy(float); n=len(c)
ema=pd.Series(c).ewm(span=PERIOD,adjust=False).mean().to_numpy()
pos=False; entry=0.0; eq=1.0; trades=[]
for i in range(PERIOD,n):
    if not pos and c[i]>=ema[i]*(1+BUF):          # دخول: السعر +5% فوق EMA7200
        pos=True; entry=c[i]
    elif pos and c[i]<ema[i]:                      # خروج: السعر تحت EMA7200
        r=(c[i]/entry-1)*100-COST; eq*=(1+r/100); trades.append(r); pos=False
if pos:
    r=(c[-1]/entry-1)*100-COST; eq*=(1+r/100); trades.append(r)
tr=np.array(trades); bh=(c[-1]/c[0]-1)*100; wr=(tr>0).mean()*100 if len(tr) else 0
print(f"##### {COIN} | دخول +5% فوق EMA7200، خروج تحته (سنة، 5m) #####")
print(f"  عدد الصفقات: {len(tr)}")
print(f"  نسبة الربح: {wr:.0f}%")
print(f"  متوسط الصفقة: {tr.mean() if len(tr) else 0:+.2f}%")
print(f"  العائد الكلي (مُركّب): {(eq-1)*100:+.0f}%")
print(f"  للمقارنة — احتفاظ بالعملة: {bh:+.0f}%")
print(f"  للمقارنة — دخول عند التقاطع (بلا 5%): −33% (240 صفقة، WR 5%)")
if len(tr): print(f"  أفضل/أسوأ صفقة: {tr.max():+.1f}% / {tr.min():+.1f}%")
print("DONE_BUF.")
