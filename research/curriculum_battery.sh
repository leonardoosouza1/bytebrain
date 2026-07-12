#!/bin/bash
# Tests Leonardo's thesis: "it's not scale, it's WHAT it sees and HOW MUCH" →
# curriculum (short→long sequences) vs fixed, same model/steps, from scratch.
# If curriculum wins coherence at equal compute, data-ordering > raw scale.
set -u
cd /home/leonardo/projects/LLM/bytebrain
PY=/home/leonardo/projects/LLM/make-shorts-video/.venv-rocm/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
echo $$ > /tmp/cur_battery.pid
J=research/coherence_journal.md
COMMON="--corpus data/pt_big.txt --dim 384 --layers 8 --heads 6 --ctx 512 --batch 32 --amp --max-steps 8000 --warmup 500 --decay-steps 8000 --val-every 1000 --seed 0"
run(){ name=$1; dir=$2; shift 2
  [ -f "$dir/DONE" ] && { echo "[skip] $name"; return 0; }
  echo "=== $name $(date '+%m-%d %H:%M') ==="
  $PY research/train_graph.py $COMMON "$@" --ckpt-dir "$dir" || { echo "$name CRASH rc=$?"; exit 1; }
  best=$($PY -c "import torch;print(round(torch.load('$dir/ckpt.pt',map_location='cpu',weights_only=False).get('best',9),4))" 2>/dev/null)
  touch "$dir/DONE"; echo "$(date '+%m-%d %H:%M') · $name · best=$best · DONE" >> "$J"
}
echo "" >> "$J"; echo "### bateria CURRÍCULO curto→longo (tese: dados>escala) $(date '+%m-%d %H:%M')" >> "$J"
run "cur-fixed" ckpt_cur_fixed
run "cur-grow"  ckpt_cur_grow  --curriculum 1 --cur-start 64 --cur-steps 4800
touch ckpt_cur_grow/ALLDONE
echo "$(date '+%m-%d %H:%M') · ### CURRÍCULO COMPLETO" >> "$J"
rm -f /tmp/cur_battery.pid
