"""Log GPU power/temperature every second; kill PID if junction stays above a limit.

Usage: gpu_guard.py PID LOG [LIMIT_C=100] [HOLD_S=5]
"""
import glob
import os
import signal
import sys
import time

pid, log = int(sys.argv[1]), sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
hold = int(sys.argv[4]) if len(sys.argv) > 4 else 5
dev = "/sys/class/drm/card1/device"
h = glob.glob(dev + "/hwmon/hwmon*")[0]


def rd(f):
    with open(f) as fh:
        return fh.read().strip()


over = 0
with open(log, "a") as out:
    while os.path.exists(f"/proc/{pid}"):
        busy = rd(dev + "/gpu_busy_percent")
        watts = int(rd(h + "/power1_average")) / 1e6
        edge, junc = int(rd(h + "/temp1_input")) // 1000, int(rd(h + "/temp2_input")) // 1000
        out.write(f"{time.time():.0f} busy={busy} W={watts:.0f} edge={edge} junction={junc}\n")
        out.flush()
        over = over + 1 if junc > limit else 0
        if over >= hold:
            out.write(f"GUARD: junction {junc}C > {limit}C for {hold}s, killing {pid}\n")
            os.kill(pid, signal.SIGTERM)
            break
        time.sleep(1)
