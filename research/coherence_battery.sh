#!/bin/bash
# COHERENCE battery — attacks topic-drift with the RIGHT levers (objective + scale),
# after ruling out architecture (top-k graph / mtp / Hebbian memory all failed coesão).
#   base-512   : 14.8M control (dim384/8L/ctx512)
#   topic-512  : same + topic-lookahead loss (NEW objective: predict gist of future)
#   scale-40M  : 40.5M (dim640/8L/ctx512) — the PROVEN lever (40M>14.8M coesão)
#   scale+topic: both best levers together
# Resumable + crash-safe (each resumes from ckpt.pt; DONE markers skip finished).
# Judged later by coherence_metric.py (continuidade/ancoragem), NOT bpb.
set -u
cd /home/leonardo/projects/LLM/bytebrain
PY=/home/leonardo/projects/LLM/make-shorts-video/.venv-rocm/bin/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
echo $$ > /tmp/coh_battery.pid
J=research/coherence_journal.md
COMMON="--corpus data/pt_big.txt --ctx 512 --amp --max-steps 14000 --warmup 800 --decay-steps 14000 --val-every 1000 --seed 0"
SMALL="--dim 384 --layers 8 --heads 6 --batch 32"
BIG="--dim 640 --layers 8 --heads 10 --batch 16"

run () {                                   # run NAME DIR [extra args...]
  name=$1; dir=$2; shift 2
  if [ -f "$dir/DONE" ]; then echo "[skip] $name DONE"; return 0; fi
  echo "=== $name START $(date '+%m-%d %H:%M') ==="
  $PY research/train_graph.py $COMMON "$@" --ckpt-dir "$dir" || { echo "$name CRASHED rc=$?"; exit 1; }
  best=$($PY -c "import torch;print(round(torch.load('$dir/ckpt.pt',map_location='cpu',weights_only=False).get('best',9),4))" 2>/dev/null)
  touch "$dir/DONE"
  echo "$(date '+%m-%d %H:%M') · $name · best_val_bpb=$best · DONE" >> "$J"
}

echo "" >> "$J"; echo "### bateria COESÃO (ctx512, lever=objetivo+escala) $(date '+%m-%d %H:%M')" >> "$J"
run "base-512"        ckpt_coh_base     $SMALL
run "topic-512"       ckpt_coh_topic    $SMALL --topic-loss 0.5
run "scale-40M"       ckpt_coh_scale    $BIG
run "scale-40M-topic" ckpt_coh_scaletop $BIG --topic-loss 0.5
touch ckpt_coh_scaletop/ALLDONE
echo "$(date '+%m-%d %H:%M') · ### BATERIA COESÃO COMPLETA" >> "$J"
rm -f /tmp/coh_battery.pid
