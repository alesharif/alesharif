#!/bin/bash
cd /home/user/alesharif
python3 -m backtest.optimizer --pop 16 --gen 40 --cap 0 --patience 10 --target-ret 35 --target-dd 30 --seed 7 > backtest/output/ga_mtf.log 2>&1
