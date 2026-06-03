#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_atrexit.log
run() { python3 -m backtest.experiments --exp "$1" --year 2025 --workers 4 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] بدء تجارب الخروج الديناميكي F1..F4 (2025)" > "$LOG"
run F1 & run F2 & wait
echo "[$(date +%H:%M:%S)] موجة 1 (F1,F2) تم" >> "$LOG"
run F3 & run F4 & wait
echo "[$(date +%H:%M:%S)] === انتهت تجارب الخروج الديناميكي ===" >> "$LOG"
