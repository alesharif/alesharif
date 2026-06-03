#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_breakout_2024.log
run() { python3 -m backtest.experiments --exp "$1" --year 2024 --workers 6 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] الاختراق على 2024: G2 ثم G4 (تسلسلي)" > "$LOG"
run G2; echo "[$(date +%H:%M:%S)] G2 تم" >> "$LOG"
run G4; echo "[$(date +%H:%M:%S)] === تم ===" >> "$LOG"
