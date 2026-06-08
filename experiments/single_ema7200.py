#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick single-coin test: 5m, long when price crosses ABOVE EMA(7200), exit when it
crosses below. Spot long-only, cost 0.2% round trip. 1 year. vs buy&hold."""
import sys, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,".")
from binance_sim import hires_data as HR
COIN=sys.argv[1] if len(sys.argv)>1 else "INJUSDT"
S,E="2024-06-01","2025-06-01"; PERIOD=7200; COST=0.20
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
days=[d.strftime("%Y-%m-%d") for d in pd.date_range(S,E,freq="D")]
print(f"{COIN} | تحميل 5m لسنة ({len(days)} يوم)...",flush=True); t0=time.time()
def _f(d):
    try: HR.load_day(COIN,"5m",d)
    except: pass
with ThreadPoolExecutor(max_workers=24) as ex: list(ex.map(_f,days))
df=HR.load_range(COIN,"5m",ms(S),ms(E))
if df is None or len(df)<PERIOD+100:
    print("بيانات غير كافية"); sys.exit()
df=df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
c=df["close"].to_numpy(float); n=len(c)
ema=pd.Series(c).ewm(span=PERIOD,adjust=False).mean().to_numpy()
print(f"تحميل {time.time()-t0:.0f}ث | شموع 5m: {n:,} (~{n/288:.0f} يوم)\n")
# long-only in/out on EMA7200 cross
pos=False; entry=0.0; eq=1.0; trades=[]; 
for i in range(PERIOD,n):
    if not pos and c[i-1]<=ema[i-1] and c[i]>ema[i]:
        pos=True; entry=c[i]
    elif pos and c[i-1]>=ema[i-1] and c[i]<ema[i]:
        r=(c[i]/entry-1)*100-COST; eq*=(1+r/100); trades.append(r); pos=False
if pos:
    r=(c[-1]/entry-1)*100-COST; eq*=(1+r/100); trades.append(r)
tr=np.array(trades); bh=(c[-1]/c[0]-1)*100
wr=(tr>0).mean()*100 if len(tr) else 0
print(f"##### النتيجة على {COIN} (سنة، 5m، عبور EMA7200) #####")
print(f"  عدد الصفقات: {len(tr)}")
print(f"  نسبة الربح: {wr:.0f}%")
print(f"  متوسط الصفقة: {tr.mean() if len(tr) else 0:+.2f}%")
print(f"  العائد الكلي (مُركّب): {(eq-1)*100:+.0f}%")
print(f"  للمقارنة — احتفاظ بالعملة: {bh:+.0f}%")
if len(tr): print(f"  أفضل/أسوأ صفقة: {tr.max():+.1f}% / {tr.min():+.1f}%")
print("DONE_E7200.")
