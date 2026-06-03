#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_2024.log
run() { python3 -m backtest.experiments --exp "$1" --year 2024 --workers 6 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] بدء اختبار 2024 تسلسلي: BASE -> E11 -> E13" > "$LOG"
run BASE; echo "[$(date +%H:%M:%S)] BASE تم" >> "$LOG"
run E11;  echo "[$(date +%H:%M:%S)] E11 تم" >> "$LOG"
run E13;  echo "[$(date +%H:%M:%S)] === انتهى اختبار 2024 ===" >> "$LOG"
