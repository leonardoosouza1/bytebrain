#!/bin/bash
# Detached supervisor for the coherence battery. Restarts on crash (battery resumes
# from ckpt.pt), stops when ALLDONE. The "seguro" layer — survives crashes and a
# closed session (launched via setsid nohup).
cd /home/leonardo/projects/LLM/bytebrain
echo $$ > /tmp/coh_supervisor.pid
for i in $(seq 1 200); do                  # hard cap on restarts (anti-runaway)
  bash research/coherence_battery.sh >> /tmp/coh_battery.log 2>&1
  if [ -f ckpt_coh_scaletop/ALLDONE ]; then break; fi
  echo "$(date '+%m-%d %H:%M') · supervisor: battery exited, restarting (resume) [#$i]" >> research/coherence_journal.md
  sleep 30
done
echo "$(date '+%m-%d %H:%M') · supervisor: DONE/stop" >> research/coherence_journal.md
rm -f /tmp/coh_supervisor.pid
