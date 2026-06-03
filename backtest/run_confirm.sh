#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_confirm.log
run() { python3 -m backtest.experiments --exp "$1" --year "$2" --workers 6 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] تأكيد 2023 + تشديد القوة النسبية (G5/G6)" > "$LOG"
run G4 2023; echo "[$(date +%H:%M:%S)] G4-2023 تم" >> "$LOG"
run G5 2024; echo "[$(date +%H:%M:%S)] G5-2024 تم" >> "$LOG"
run G5 2025; echo "[$(date +%H:%M:%S)] G5-2025 تم" >> "$LOG"
run G5 2023; echo "[$(date +%H:%M:%S)] G5-2023 تم" >> "$LOG"
run G6 2024; echo "[$(date +%H:%M:%S)] G6-2024 تم" >> "$LOG"
run G6 2025; echo "[$(date +%H:%M:%S)] G6-2025 تم" >> "$LOG"
echo "[$(date +%H:%M:%S)] === انتهى الكل ===" >> "$LOG"
