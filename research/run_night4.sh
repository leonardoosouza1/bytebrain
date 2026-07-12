#!/bin/bash
# BATERIA 4 — atacar os GAPS (generalização, regra-em-semente, álgebra, roteador) + empurrar solo grande.
# Resumível (skip-if-JSON), tolera falha, auto-planeja ao final. Math-trunk primeiro, Qwen3-4B-trunk por último.
cd /home/leonardo/projects/LLM
source .venv-rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/leonardo/projects/LLM/bytebrain/research:$PYTHONPATH
J=/home/leonardo/projects/LLM/bytebrain/research/overnight_night4_journal.md
run(){
  echo -e "\n\n========== RUN $1 ($(date '+%H:%M:%S')) ==========" >> "$J"
  python "$1" >> "$J" 2>&1 && echo ">>> $1 OK ($(date '+%H:%M:%S'))" >> "$J" \
                           || echo "!!! $1 FALHOU ($(date '+%H:%M:%S'))" >> "$J"
}
echo -e "\n# BATERIA 4 — GAPS & FORA DA CAIXA — (re)início $(date)" >> "$J"
run bytebrain/research/marco123_generalize_multiphrase.py
run bytebrain/research/marco129_forest_router.py
run bytebrain/research/marco128_seed_algebra.py
run bytebrain/research/marco127_rule_seed.py
run bytebrain/research/marco126_qwen3_trunk.py
echo -e "\n# FIM BATERIA 4 — $(date)" >> "$J"

cat > /home/leonardo/projects/LLM/bytebrain/research/planned_next4.md <<'PLAN'
# Bateria 5 planejada (auto-gerado) — depende dos achados da 4

Se M123 (multi-formulação) GENERALIZOU:
  - M130: escalar formulações (5-10) + testar generalização p/ pergunta de FORMA totalmente nova (não-template).
  - M131: semente de CONCEITO — treinar fato + suas consequências, testar inferência simples.
Se M127 (regra) pegou a REGRA:
  - M132: regras compostas (2 passos), regras com memória, e "biblioteca de regras" (várias numa floresta).
Se M128 (álgebra) foi LINEAR:
  - M133: floresta comprimida por fusão (trocar N árvores compatíveis por 1 fundida); medir perda.
Se M126 (Qwen3-tronco) foi o solo mais barato:
  - M134: lei de escala do solo — hidden 1536→2048→3072→4096: bytes/fato int4 K=1 vs dimensão (ajustar curva).
  - M135: destilar de N modelos p/ o tronco Qwen3-4B grande (armazém cross-model no melhor solo).
Sempre: multi-seed onde diferença < 5% (variância ROCm ±2), honestidade, dump incremental.
PLAN
echo "auto-plano bateria 5 escrito em planned_next4.md" >> "$J"
echo -e "\n# CONCLUÍDO + PLANEJADO — $(date)" >> "$J"
