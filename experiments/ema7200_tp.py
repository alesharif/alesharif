#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EMA7200 with TAKE-PROFIT +10% (exit at +10% OR cross below EMA7200). 45 coins, 1yr 5m.
Two entries: (A) cross above EMA7200, (B) +5% above EMA7200. cost 0.2% RT. vs buy&hold."""
import sys, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,".")
from binance_sim import hires_data as HR
S,E="2024-06-01","2025-06-01"; PERIOD=7200; COST=0.20; TP=0.10
COINS=["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT",
"LINKUSDT","DOTUSDT","LTCUSDT","TRXUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","INJUSDT",
"SUIUSDT","SEIUSDT","TIAUSDT","FILUSDT","ATOMUSDT","UNIUSDT","AAVEUSDT","RUNEUSDT","ALGOUSDT",
"ICPUSDT","FETUSDT","GALAUSDT","SANDUSDT","MANAUSDT","AXSUSDT","GRTUSDT","LDOUSDT","IMXUSDT",
"STXUSDT","CRVUSDT","COMPUSDT","MKRUSDT","SNXUSDT","DYDXUSDT","ENAUSDT","WLDUSDT","JUPUSDT","ORDIUSDT"]
def ms(s): return int(pd.Timestamp(s,tz="UTC").timestamp()*1000)
days=[d.strftime("%Y-%m-%d") for d in pd.date_range(S,E,freq="D")]
def _f(cd):
    try: HR.load_day(cd[0],"5m",cd[1])
    except: pass
with ThreadPoolExecutor(max_workers=24) as ex: list(ex.map(_f,[(c,d) for c in COINS for d in days]))
def backtest(cl,ema,buf):
    n=len(cl); pos=False; entry=0.0; eq=1.0; tr=[]
    for i in range(PERIOD,n):
        if not pos:
            if (buf==0 and cl[i-1]<=ema[i-1] and cl[i]>ema[i]) or (buf>0 and cl[i]>=ema[i]*(1+buf)):
                pos=True; entry=cl[i]
        else:
            if cl[i]>=entry*(1+TP):      # TP +10%
                r=TP*100-COST; eq*=(1+r/100); tr.append(r); pos=False
            elif cl[i]<ema[i]:            # exit on cross below EMA7200
                r=(cl[i]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r); pos=False
    if pos:
        r=(cl[-1]/entry-1)*100-COST; eq*=(1+r/100); tr.append(r)
    wr=(np.array(tr)>0).mean()*100 if tr else 0
    return (eq-1)*100, len(tr), wr
A=[]; B=[]; BH=[]
for c in COINS:
    df=HR.load_range(c,"5m",ms(S),ms(E))
    if df is None or len(df)<PERIOD+2000: continue
    cl=df.drop_duplicates("time").sort_values("time")["close"].to_numpy(float); cl=cl[np.isfinite(cl)&(cl>0)]
    if len(cl)<PERIOD+2000: continue
    ema=pd.Series(cl).ewm(span=PERIOD,adjust=False).mean().to_numpy()
    A.append((c,)+backtest(cl,ema,0.0)); B.append((c,)+backtest(cl,ema,0.05)); BH.append((c,(cl[-1]/cl[0]-1)*100))
def summ(R,lab):
    r=np.array([x[1] for x in R]); print(f"{lab}: متوسط {r.mean():+.0f}% | وسيط {np.median(r):+.0f}% | رابحة {(r>0).mean()*100:.0f}% | متوسط WR {np.mean([x[3] for x in R]):.0f}%")
print(f"\nعملات: {len(A)} | TP=+10%\n")
summ(A,"(أ) دخول عند التقاطع + TP10")
summ(B,"(ب) دخول +5% فوق EMA + TP10")
bh=np.array([x[1] for x in BH]); print(f"الاحتفاظ: متوسط {bh.mean():+.0f}% | وسيط {np.median(bh):+.0f}%")
print(f"\n{'العملة':<10}{'أ(تقاطع)':>10}{'ب(+5%)':>9}{'احتفاظ':>9}")
for i,c in enumerate([x[0] for x in A]):
    print(f"{c:<10}{A[i][1]:>+9.0f}%{B[i][1]:>+8.0f}%{BH[i][1]:>+8.0f}%")
print("DONE_TP.")
