#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_revalidate.log
run() { python3 -m backtest.experiments --exp "$1" --year "$2" --workers 6 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] إعادة تحقّق كاملة بعد إصلاح الإحماء" > "$LOG"
for Y in 2024 2023; do
  for E in BASE E13 G4; do
    run "$E" "$Y"; echo "[$(date +%H:%M:%S)] $E-$Y تم" >> "$LOG"
  done
done
echo "[$(date +%H:%M:%S)] === انتهت إعادة التحقّق ===" >> "$LOG"
