#!/bin/bash
cd /home/user/alesharif
LOG=backtest/output/exp_ls.log
run() { python3 -m backtest.experiments --exp "$1" --year "$2" --workers 6 >> "$LOG" 2>&1; }
echo "[$(date +%H:%M:%S)] Long/Short على 2025: H1 H2 H3 (تسلسلي)" > "$LOG"
run H1 2025; echo "[$(date +%H:%M:%S)] H1-2025 تم" >> "$LOG"
run H2 2025; echo "[$(date +%H:%M:%S)] H2-2025 تم" >> "$LOG"
run H3 2025; echo "[$(date +%H:%M:%S)] === H*-2025 تم ===" >> "$LOG"
