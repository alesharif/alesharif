#!/bin/bash
# Keep 1s cache under a size cap; delete oldest 1s files if disk free < 5GB.
while true; do
  FREE_KB=$(df --output=avail / | tail -1)
  if [ "$FREE_KB" -lt 5000000 ]; then
    # delete oldest 400 1s files to reclaim space
    ls -1t /home/user/alesharif/data/cache/hires/*-1s-*.csv 2>/dev/null | tail -400 | xargs rm -f 2>/dev/null
  fi
  sleep 60
done
