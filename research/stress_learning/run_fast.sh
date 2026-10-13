#!/bin/bash
# Fast sleep battery (fast_learning.py). Launch with:
#   setsid nohup bash run_fast.sh < /dev/null > /dev/null 2>&1 &
export HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTHONUNBUFFERED=1
cd /home/leonardo/projects/LLM/bytebrain/research/stress_learning
OUT=${OUT:-/media/llm-data/universe-runs/sleep-fast-20261013/full}
PY=/home/leonardo/projects/LLM/.venv-rocm/bin/python
mkdir -p $OUT
run() { nice -n 5 $PY fast_learning.py --seed $1 --out $OUT "${@:2}" >> $OUT/seed$1.log 2>&1; }
run 0 & run 1 & wait
run 2
echo DONE >> $OUT/run.log
