#!/bin/bash
# SCALE-REAL battery — pushes the two PROVEN coherence levers (size + context),
# afforded on 12GB by gradient checkpointing. After ruling out architecture/objective.
#   40M-ctx1024 : 40M (dim640/8L) + LONG context — the two winning levers TOGETHER
#                 (until now only tested separately: 40M@ctx512 and 14.8M@ctx1024).
#   86M-ctx512  : a REAL 86M (dim768/12L) trained to convergence (ckpt_big2 was step 500).
# Resumable + crash-safe (resume from ckpt.pt; DONE markers skip finished).
# Judged by coherence_metric.py over >=5 prompts (n=1 is noise).
set -u
cd /home/leonardo/projects/LLM/bytebrain
PY=/home/leonardo/projects/LLM/make-shorts-video/.venv-rocm/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
echo $$ > /tmp/scale_real.pid
J=research/coherence_journal.md

run () {                                    # run NAME DIR [extra args...]
  name=$1; dir=$2; shift 2
  if [ -f "$dir/DONE" ]; then echo "[skip] $name DONE"; return 0; fi
  echo "=== $name START $(date '+%m-%d %H:%M') ==="
  $PY research/train_graph.py --corpus data/pt_big.txt --amp --grad-ckpt 1 --seed 0 \
     --val-every 1000 "$@" --ckpt-dir "$dir" || { echo "$name CRASHED rc=$?"; exit 1; }
  best=$($PY -c "import torch;print(round(torch.load('$dir/ckpt.pt',map_location='cpu',weights_only=False).get('best',9),4))" 2>/dev/null)
  touch "$dir/DONE"
  echo "$(date '+%m-%d %H:%M') · $name · best_val_bpb=$best · DONE" >> "$J"
}

echo "" >> "$J"; echo "### bateria SCALE-REAL (grad-ckpt, lever=tamanho+contexto) $(date '+%m-%d %H:%M')" >> "$J"
run "40M-ctx1024" ckpt_scale_40m_c1k --dim 640 --layers 8  --heads 10 --ctx 1024 --batch 12 --max-steps 14000 --warmup 1000 --decay-steps 14000
run "86M-ctx512"  ckpt_scale_86m     --dim 768 --layers 12 --heads 12 --ctx 512  --batch 12 --max-steps 18000 --warmup 1000 --decay-steps 18000
touch ckpt_scale_86m/ALLDONE
echo "$(date '+%m-%d %H:%M') · ### BATERIA SCALE-REAL COMPLETA" >> "$J"
rm -f /tmp/scale_real.pid
