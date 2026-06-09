#!/bin/bash
cd /home/user/alesharif
python3 -m backtest.optimizer --objective return --pop 16 --gen 40 --cap 0 \
  --patience 10 --target-ret 35 --target-dd 50 --seed 11 > backtest/output/ga_ret.log 2>&1
