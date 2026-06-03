#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_breakout.log
run() { python3 -m backtest.experiments --exp "$1" --year 2025 --workers 6 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] إشارة الاختراق على 2025: G1 G2 G3 G4 (تسلسلي)" > "$LOG"
run G1; echo "[$(date +%H:%M:%S)] G1 تم" >> "$LOG"
run G2; echo "[$(date +%H:%M:%S)] G2 تم" >> "$LOG"
run G3; echo "[$(date +%H:%M:%S)] G3 تم" >> "$LOG"
run G4; echo "[$(date +%H:%M:%S)] === تم G1..G4 ===" >> "$LOG"
