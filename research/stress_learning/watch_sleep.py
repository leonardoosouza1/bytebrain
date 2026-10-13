"""Print the latest row per (seed, arm) of the sleep battery."""

import glob
import json
import sys

PATHS = sorted(glob.glob("/media/llm-data/universe-runs/sleep-learning-20261013/full/seed*.log"))

latest = {}
for line in (l.strip() for p in (sys.argv[1:] or PATHS) for l in open(p)):
    if not line.startswith("{"):
        if "Error" in line or "Traceback" in line:
            print("LOG:", line)
        continue
    r = json.loads(line)
    if "arm" in r:
        latest[(r["seed"], r["arm"])] = r

print(f"{'seed':>4} {'arm':>13} {'question':>8} {'wake':>5} {'sleep':>5} {'itemgrad':>8} "
      f"{'relearn':>7} {'heldout':>7} {'served':>6} {'secs':>6}")
for (seed, arm), r in sorted(latest.items()):
    print(f"{seed:>4} {arm:>13} {r['t']:>8} {r['wake_steps']:>5} {r['sleep_steps']:>5} "
          f"{r['item_grads']:>8} {r['relearn_events']:>7} {r['acc_heldout']:>7.3f} "
          f"{r['served_acc']:>6.3f} {r['elapsed_s']:>6}")
