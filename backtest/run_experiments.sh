#!/bin/bash
cd /home/user/alesharif
mkdir -p backtest/output
run() { python3 -m backtest.experiments --exp "$1" --workers 4 >> backtest/output/exp_all.log 2>&1; }
echo "[$(date +%H:%M:%S)] بدء التجارب E1..E6" > backtest/output/exp_all.log
# الموجة 1: E1 E2 E3
run E1 & run E2 & run E3 & wait
echo "[$(date +%H:%M:%S)] انتهت الموجة 1" >> backtest/output/exp_all.log
# الموجة 2: E4 E5 E6
run E4 & run E5 & run E6 & wait
echo "[$(date +%H:%M:%S)] === كل التجارب انتهت ===" >> backtest/output/exp_all.log
