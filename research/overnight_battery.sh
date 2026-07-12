#!/bin/bash
# Overnight byte-neuron battery — idempotent + resumable + crash-safe.
# Each experiment resumes from ckpt.pt; a DONE marker skips finished ones.
# On crash it exits non-zero so the hourly cron relaunches it (and it resumes
# exactly where it stopped). No pgrep/pkill of "train" patterns (avoids self-kill).
set -u
cd /home/leonardo/projects/LLM/bytebrain
PY=/home/leonardo/projects/LLM/make-shorts-video/.venv-rocm/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
echo $$ > /tmp/byte_battery.pid
J=overnight_journal_neuron.md
C="--corpus data/pt_big.txt --dim 384 --layers 8 --heads 6 --ctx 1024 --batch 12 --amp --max-steps 40000 --warmup 1000 --decay-steps 40000 --val-every 1000 --seed 0"

run () {                                  # run NAME DIR [extra args...]
  name=$1; dir=$2; shift 2
  if [ -f "$dir/DONE" ]; then echo "[skip] $name DONE"; return 0; fi
  echo "=== $name START $(date '+%m-%d %H:%M') ==="
  $PY research/train_graph.py $C --ckpt-dir "$dir" "$@" || { echo "$name CRASHED rc=$?"; exit 1; }
  # reached max_steps cleanly -> mark DONE + log best
  best=$($PY -c "import torch;print(round(torch.load('$dir/ckpt.pt',map_location='cpu',weights_only=False).get('best',9),4))" 2>/dev/null)
  touch "$dir/DONE"
  echo "$(date '+%m-%d %H:%M') · $name · best_val_bpb=$best · DONE" >> "$J"
}

echo "" >> "$J"; echo "### bateria byte-neuron (ctx1024, 14.8M, foco COESÃO) $(date '+%m-%d %H:%M')" >> "$J"
run "dense-L1024"     ckpt_ovn_dense                            # flagship coesão (prioridade)
run "top8-L1024"      ckpt_ovn_top8    --topk 8                 # esparsidade segura qualidade em L longo?
run "top8+4bit-L1024" ckpt_ovn_top8q4  --topk 8 --quant-bits 4 # mais leve
echo "$(date '+%m-%d %H:%M') · ### BATERIA COMPLETA" >> "$J"
rm -f /tmp/byte_battery.pid
