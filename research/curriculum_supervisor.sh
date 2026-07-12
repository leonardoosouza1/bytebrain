#!/bin/bash
# Waits for the distill2 fine-tune to free the GPU, then runs the curriculum battery.
# Restart-on-crash, detached (setsid). Resumable.
cd /home/leonardo/projects/LLM/bytebrain
echo $$ > /tmp/cur_sup.pid
while [ ! -f ckpt_distill2/DONE ]; do sleep 60; done
echo "$(date '+%m-%d %H:%M') · curric sup: distill2 done, GPU free → starting" >> research/coherence_journal.md
for i in $(seq 1 100); do
  bash research/curriculum_battery.sh >> /tmp/cur_battery.log 2>&1
  [ -f ckpt_cur_grow/ALLDONE ] && break
  echo "$(date '+%m-%d %H:%M') · curric sup: restart (resume) [#$i]" >> research/coherence_journal.md
  sleep 30
done
echo "$(date '+%m-%d %H:%M') · curric sup: DONE/stop" >> research/coherence_journal.md
rm -f /tmp/cur_sup.pid
