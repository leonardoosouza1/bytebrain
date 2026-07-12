#!/bin/bash
# Detached supervisor: keeps the byte-neuron battery alive overnight INDEPENDENT
# of Claude. Restarts it on crash (the battery resumes from ckpt.pt), stops when
# all experiments are DONE. This is the "seguro" layer — training progresses even
# if the laptop/Claude is closed.
cd /home/leonardo/projects/LLM/bytebrain
echo $$ > /tmp/byte_supervisor.pid
for i in $(seq 1 200); do                     # hard cap on restarts (anti-runaway)
  bash research/overnight_battery.sh >> /tmp/byte_battery.log 2>&1
  if [ -f ckpt_ovn_top8q4/DONE ]; then break; fi   # all 3 experiments finished
  echo "$(date '+%m-%d %H:%M') · supervisor: battery exited, restarting (resume) [#$i]" >> overnight_journal_neuron.md
  sleep 30
done
echo "$(date '+%m-%d %H:%M') · supervisor: DONE/stop" >> overnight_journal_neuron.md
rm -f /tmp/byte_supervisor.pid
