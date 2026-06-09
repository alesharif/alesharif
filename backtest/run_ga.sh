#!/bin/bash
cd /home/user/alesharif
python3 -m backtest.optimizer --pop 14 --gen 6 --cap 100 --seed 42 > backtest/output/ga_run.log 2>&1
