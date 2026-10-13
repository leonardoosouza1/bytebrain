"""Print the latest row per (seed, arm) from the live progress file."""

import glob
import json
import sys

PATHS = sorted(glob.glob("/media/llm-data/universe-runs/stress-learning-20261013/full/seed*.log"))

latest = {}
lines = [l for p in (sys.argv[1:] or PATHS) for l in open(p)]
for line in lines:
    line = line.strip()
    if not line.startswith("{"):
        if "Error" in line or "Traceback" in line:
            print("LOG:", line)
        continue
    r = json.loads(line)
    if "arm" in r:
        latest[(r["seed"], r["arm"])] = r

print(f"{'seed':>4} {'arm':>7} {'question':>9} {'steps':>6} {'consults':>8} "
      f"{'heldout':>8} {'trainpar':>8} {'served':>7} {'secs':>6}")
for (seed, arm), r in sorted(latest.items()):
    print(f"{seed:>4} {arm:>7} {r['t']:>9} {r['steps']:>6} {r['consults']:>8} "
          f"{r['acc_heldout']:>8.3f} {r['acc_train']:>8.3f} {r['served_acc']:>7.3f} "
          f"{r['elapsed_s']:>6}")
