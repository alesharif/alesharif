#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA7200 +5% entry / exit-below, on ~45 liquid coins, 1yr 5m. Aggregates vs buy&hold."""
import sys, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,".")
from binance_sim import hires_data as HR
S,E="2024-06-01","2025-06-01"; PERIOD=7200; COST=0.20; BUF=0.10
COINS=["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT",
"LINKUSDT","DOTUSDT","LTCUSDT","TRXUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","INJUSDT",
"SUIUSDT","SEIUSDT","TIAUSDT","FILUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT","RUNEUSDT","ALGOUSDT",
"ICPUSDT","FETUSDT","GALAUSDT","SANDUSDT","MANAUSDT","AXSUSDT","GRTUSDT","LDOUSDT","IMXUSDT",
"STXUSDT","CRVUSDT","COMPUSDT","MKRUSDT","SNXUSDT","DYDXUSDT","ENAUSDT","WLDUSDT","JUPUSDT","ORDIUSDT"]
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
days=[d.strftime("%Y-%m-%d") for d in pd.date_range(S,E,freq="D")]
print(f"تحميل 5m لـ{len(COINS)} عملة × {len(days)} يوم...",flush=True); t0=time.time()
def _f(cd):
    try: HR.load_day(cd[0],"5m",cd[1])
    except: pass
with ThreadPoolExecutor(max_workers=24) as ex: list(ex.map(_f,[(c,d) for c in COINS for d in days]))
print(f"تحميل {(time.time()-t0)/60:.1f}د\n",flush=True)
res=[]
for c in COINS:
    df=HR.load_range(c,"5m",ms(S),ms(E))
    if df is None or len(df)<PERIOD+2000: continue
    df=df.drop_duplicates("time").sort_values("time"); cl=df["close"].to_numpy(float)
    cl=cl[np.isfinite(cl)&(cl>0)]; n=len(cl)
    ema=pd.Series(cl).ewm(span=PERIOD,adjust=False).mean().to_numpy()
    pos=False; entry=0.0; eq=1.0; tr=[]
    for i in range(PERIOD,n):
        if not pos and cl[i]>=ema[i]*(1+BUF): pos=True; entry=cl[i]
        elif pos and cl[i]<ema[i]:
            r=(cl[i]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r); pos=False
    if pos:
        r=(cl[-1]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r)
    bh=(cl[-1]/cl[0]-1)*100; wr=(np.array(tr)>0).mean()*100 if tr else 0
    res.append((c,(eq-1)*100,bh,len(tr),wr))
Sx=np.array([x[1] for x in res]); Bx=np.array([x[2] for x in res])
print(f"عملات مُختبَرة: {len(res)}\n")
print("##### الاستراتيجية (EMA7200 + دخول +10%) #####")
print(f"  متوسط العائد/عملة: {Sx.mean():+.0f}%   الوسيط: {np.median(Sx):+.0f}%")
print(f"  عملات رابحة: {(Sx>0).mean()*100:.0f}%   تتفوّق على الاحتفاظ: {(Sx>Bx).mean()*100:.0f}%")
print("##### الاحتفاظ #####")
print(f"  متوسط: {Bx.mean():+.0f}%   الوسيط: {np.median(Bx):+.0f}%\n")
print(f"{'العملة':<10}{'استراتيجية':>11}{'احتفاظ':>9}{'صفقات':>7}{'WR':>6}")
for c,s,b,nt,w in sorted(res,key=lambda x:-x[1]):
    print(f"{c:<10}{s:>+10.0f}%{b:>+8.0f}%{nt:>7}{w:>5.0f}%")
print("DONE_ESAMPLE.")
