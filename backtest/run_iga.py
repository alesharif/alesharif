import sys
from backtest.intraday import IntradayData, run_iga, run_intraday, _fmt, TEST_YEAR, MAKER_FEE, TAKER_FEE
import json
LIQUID = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT','ADAUSDT',
'AVAXUSDT','LINKUSDT','DOTUSDT','MATICUSDT','LTCUSDT','TRXUSDT','ATOMUSDT','UNIUSDT',
'NEARUSDT','APTUSDT','FILUSDT','ARBUSDT','OPUSDT','INJUSDT','SUIUSDT','SEIUSDT',
'TIAUSDT','RUNEUSDT','AAVEUSDT','FTMUSDT','ALGOUSDT','SANDUSDT','GALAUSDT',
'ICPUSDT','HBARUSDT','VETUSDT','GRTUSDT','EGLDUSDT','THETAUSDT','AXSUSDT','EOSUSDT',
'FLOWUSDT','CHZUSDT','MANAUSDT','XTZUSDT','CRVUSDT','LDOUSDT','ENSUSDT','DYDXUSDT',
'1INCHUSDT','COMPUSDT','ZECUSDT','SNXUSDT']
print(f"GA لحظي: maker={MAKER_FEE*100:.3f}% taker={TAKER_FEE*100:.3f}% | {len(LIQUID)} عملة", flush=True)
idata = IntradayData(top_n=50, req_universe=LIQUID)
print("-"*70, flush=True)
bg, br = run_iga(idata, pop=18, gens=30, patience=10, seed=7)
print("\n"+"="*70, flush=True)
print("أفضل تركيبة لحظية: "+json.dumps(bg, ensure_ascii=False), flush=True)
for y in [2023,2024]:
    print(_fmt(y,'train',br['train'][y]), flush=True)
print(_fmt(TEST_YEAR,'TEST',run_intraday(idata,TEST_YEAR,bg)), flush=True)
print("="*70, flush=True)
