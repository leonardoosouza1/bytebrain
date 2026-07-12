#!/bin/bash
# BATERIA 6 — cura o reversal curse + semente de raciocínio (swing fora-da-caixa). Resumível, tolera falha.
cd /home/leonardo/projects/LLM
source .venv-rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/leonardo/projects/LLM/bytebrain/research:$PYTHONPATH
J=/home/leonardo/projects/LLM/bytebrain/research/overnight_night6_journal.md
run(){
  echo -e "\n\n========== RUN $1 ($(date '+%H:%M:%S')) ==========" >> "$J"
  python "$1" >> "$J" 2>&1 && echo ">>> $1 OK ($(date '+%H:%M:%S'))" >> "$J" \
                           || echo "!!! $1 FALHOU ($(date '+%H:%M:%S'))" >> "$J"
}
echo -e "\n# BATERIA 6 — reversal cure + raciocínio — (re)início $(date)" >> "$J"
run bytebrain/research/marco133_reversal_cure.py
run bytebrain/research/marco134_reasoning_seed.py
echo -e "\n# FIM BATERIA 6 — $(date)" >> "$J"
