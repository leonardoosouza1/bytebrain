#!/bin/bash
# BATERIA 5 — empurra os ganhos da 4: generalização profunda, biblioteca de regras, armazém cross-model
# no solo grande. Resumível, tolera falha, auto-planeja. Math-trunk primeiro, Qwen3-trunk por último.
cd /home/leonardo/projects/LLM
source .venv-rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/leonardo/projects/LLM/bytebrain/research:$PYTHONPATH
J=/home/leonardo/projects/LLM/bytebrain/research/overnight_night5_journal.md
run(){
  echo -e "\n\n========== RUN $1 ($(date '+%H:%M:%S')) ==========" >> "$J"
  python "$1" >> "$J" 2>&1 && echo ">>> $1 OK ($(date '+%H:%M:%S'))" >> "$J" \
                           || echo "!!! $1 FALHOU ($(date '+%H:%M:%S'))" >> "$J"
}
echo -e "\n# BATERIA 5 — empurrar os ganhos — (re)início $(date)" >> "$J"
run bytebrain/research/marco130_deep_generalize.py
run bytebrain/research/marco131_rule_library.py
run bytebrain/research/marco132_crossmodel_bigsoil.py
echo -e "\n# FIM BATERIA 5 — $(date)" >> "$J"
cat > /home/leonardo/projects/LLM/bytebrain/research/planned_next5.md <<'PLAN'
# Bateria 6 planejada (auto)
- Se M130 generalizou p/ pergunta mas sofreu reversal: testar seed treinada em AMBAS direções (cura o reversal curse?).
- Se M131 biblioteca de regras funcionou: escalar p/ 10+ regras + regras encadeadas de 3 passos; regra que CHAMA outra.
- Se M132 armazém barato: destilar TODAS as 50 perguntas dos 3 modelos no Qwen3-4B + medir teto do armazém no solo grande.
- Novo: SEMENTE HIERÁRQUICA (galho→sub-galho): uma seed-mãe geral + seeds-filhas que refinam, roteadas por profundidade.
- Novo: SEMENTE DE RACIOCÍNIO — treinar seed em cadeias "porque X, então Y"; testar se generaliza o passo lógico.
PLAN
echo "auto-plano bateria 6 em planned_next5.md" >> "$J"
