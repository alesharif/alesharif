#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_atrexit_2024.log
run() { python3 -m backtest.experiments --exp "$1" --year 2024 --workers 4 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] الخروج الديناميكي على 2024: F2, F3" > "$LOG"
run F2 & run F3 & wait
echo "[$(date +%H:%M:%S)] === تم ===" >> "$LOG"
