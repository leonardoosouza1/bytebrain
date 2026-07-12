#!/bin/bash
# OS-level watchdog (crontab, independent of Claude). SOLE relauncher of the
# supervisor — if the supervisor died and the battery isn't all done, restart it.
# Uses pidfiles only (no pgrep/pkill of "train" → no self-kill).
cd /home/leonardo/projects/LLM/bytebrain || exit 0
[ -f ckpt_ovn_top8q4/DONE ] && exit 0                 # all 3 experiments finished
SP=$(cat /tmp/byte_supervisor.pid 2>/dev/null)
if [ -z "$SP" ] || ! kill -0 "$SP" 2>/dev/null; then
  setsid nohup bash research/overnight_supervisor.sh > /tmp/byte_supervisor.log 2>&1 < /dev/null &
  echo "$(date '+%m-%d %H:%M') · OS-watchdog: supervisor estava morto → relançado" >> overnight_journal_neuron.md
fi
