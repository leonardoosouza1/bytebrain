#!/bin/bash
# Bateria noturna 2 — roda as 3 frentes SEQUENCIAL (1 GPU, sem concorrência), tolera falha por script.
cd /home/leonardo/projects/LLM
source .venv-rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
J=/home/leonardo/projects/LLM/bytebrain/research/overnight_night2_journal.md
run() {
  echo -e "\n\n========== RUN $1  ($(date '+%H:%M:%S')) ==========" >> "$J"
  python "$1" >> "$J" 2>&1 && echo ">>> $1 OK ($(date '+%H:%M:%S'))" >> "$J" \
                            || echo "!!! $1 FALHOU ($(date '+%H:%M:%S')) — ver acima" >> "$J"
}
echo "# BATERIA NOTURNA 2 — início $(date)" > "$J"
run bytebrain/research/marco112_fact_forest.py
run make-shorts-video/iara_pinboard/marco113_night_image.py
run bytebrain/research/marco114_wisdom_distill.py
echo -e "\n# FIM DA BATERIA — $(date)" >> "$J"
