#!/bin/bash
cd /home/user/alesharif
run() { python3 -m backtest.experiments --exp "$1" --workers 4 >> backtest/output/exp_quality.log 2>&1; }
echo "[$(date +%H:%M:%S)] بدء تجارب ATR/ADX E11..E14" > backtest/output/exp_quality.log
run E11 & run E12 & wait
echo "[$(date +%H:%M:%S)] موجة 1 (E11,E12) انتهت" >> backtest/output/exp_quality.log
run E13 & run E14 & wait
echo "[$(date +%H:%M:%S)] === انتهت كل تجارب ATR/ADX ===" >> backtest/output/exp_quality.log
