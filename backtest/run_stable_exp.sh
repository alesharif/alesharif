#!/bin/bash
cd /home/user/alesharif
run() { python3 -m backtest.experiments --exp "$1" --workers 4 >> backtest/output/exp_stable.log 2>&1; }
echo "[$(date +%H:%M:%S)] بدء تجارب الفلتر المستقر E7..E10" > backtest/output/exp_stable.log
run E7 & run E8 & wait
echo "[$(date +%H:%M:%S)] موجة 1 (E7,E8) انتهت" >> backtest/output/exp_stable.log
run E9 & run E10 & wait
echo "[$(date +%H:%M:%S)] === انتهت كل تجارب الفلتر المستقر ===" >> backtest/output/exp_stable.log
