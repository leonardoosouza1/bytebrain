#!/bin/bash
# Detached supervisor for the SCALE-REAL battery. Waits for the current coherence
# battery to free the single GPU, then runs scale-real; restarts on crash (resume),
# stops at ALLDONE. Survives crashes + closed session (setsid nohup).
cd /home/leonardo/projects/LLM/bytebrain
echo $$ > /tmp/scale_real_sup.pid
# wait until the current coherence battery finishes (frees the GPU)
while [ ! -f ckpt_coh_scaletop/ALLDONE ]; do sleep 60; done
echo "$(date '+%m-%d %H:%M') · scale-real sup: current battery done, GPU free → starting" >> research/coherence_journal.md
for i in $(seq 1 200); do                   # hard cap on restarts (anti-runaway)
  bash research/scale_real_battery.sh >> /tmp/scale_real.log 2>&1
  if [ -f ckpt_scale_86m/ALLDONE ]; then break; fi
  echo "$(date '+%m-%d %H:%M') · scale-real sup: battery exited, restarting (resume) [#$i]" >> research/coherence_journal.md
  sleep 30
done
echo "$(date '+%m-%d %H:%M') · scale-real sup: DONE/stop" >> research/coherence_journal.md
rm -f /tmp/scale_real_sup.pid
