#!/bin/bash
# BATERIA NOTURNA 3 — campanha CROSS-MODEL (criar árvores do conhecimento de vários modelos + herança).
# Sequencial, 1 GPU, tolera falha por script, dump incremental. Ao final AUTO-PLANEJA mais.
cd /home/leonardo/projects/LLM
source .venv-rocm/bin/activate
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/leonardo/projects/LLM/bytebrain/research:$PYTHONPATH
J=/home/leonardo/projects/LLM/bytebrain/research/overnight_night3_journal.md
run(){
  echo -e "\n\n========== RUN $1 ($(date '+%H:%M:%S')) ==========" >> "$J"
  python "$1" >> "$J" 2>&1 && echo ">>> $1 OK ($(date '+%H:%M:%S'))" >> "$J" \
                           || echo "!!! $1 FALHOU ($(date '+%H:%M:%S'))" >> "$J"
}
echo -e "\n# BATERIA NOTURNA 3 — CROSS-MODEL — (re)início $(date)" >> "$J"
run bytebrain/research/marco117_crossmodel_distill.py
run bytebrain/research/marco118_multiteacher_forest.py
run bytebrain/research/marco119_crossmodel_heredity.py
run bytebrain/research/marco120_trunk_soil.py
run bytebrain/research/marco121_generalization.py
run bytebrain/research/marco122_stream_noforget.py
echo -e "\n# FIM DA BATERIA 3 — $(date)" >> "$J"

# ---- AUTO-PLANEJAMENTO: fila de próximos experimentos (Leonardo: "se termina planeje mais") ----
cat > /home/leonardo/projects/LLM/bytebrain/research/planned_next.md <<'PLAN'
# Próximos experimentos planejados (auto-gerado pela bateria 3)

Concluídos nesta noite: M117 (destilação cross-model), M118 (floresta multi-professor/variedade),
M119 (herança cross-model), M120 (melhor solo), M121 (generalização), M122 (stream/não-esquecimento).

## Fila (ordem de valor)
1. **M123 Roteador da floresta** — plugar o roteador territorial: a floresta escolhe a árvore por query
   (fecha o gap "cobertura oráculo → cobertura real"). Medir acerto do roteador vs oráculo.
2. **M124 Fusão de sementes** — mesclar 2 árvores (média/soma dos vetores) cobre a união? Ou interfere?
   (a "poda por fusão": trocar N árvores por 1 quando compatíveis.)
3. **M125 Professor-conselho** — quando professores DISCORDAM, plantar a resposta por VOTO da maioria
   vs a de um só; medir se o conselho cross-model destila mais fatos CORRETOS (validar contra gabarito).
4. **M126 Mutação ótima** — varrer escala de ruído do broto (0.01–0.3): qual acelera mais a filha?
   (afina o "reproduzir com variação".)
5. **M127 Árvore de árvores** — sub-nichos recursivos: uma filha que brota netas; medir profundidade útil.
6. **M128 Solo grande** — repetir M120 com Qwen3-4B como TRONCO (gguf→torch): tronco maior = solo mais
   barato por byte? (tese sabedoria=andaime em escala.)
7. **M129 Transferência de semente** — semente plantada no tronco A recuperada no tronco B da MESMA
   família/arch (Qwen2.5 variantes) — a semente é portável entre modelos irmãos?

## Método (sempre)
- Otimizador estável (lr0.1 cosine + grad-clip), alvo 1-token, quant per-grupo, dump incremental.
- Honestidade: sem número fake; multi-seed onde a diferença for < ~5% (variância ROCm ±2).
PLAN
echo "auto-plano escrito em bytebrain/research/planned_next.md" >> "$J"
echo -e "\n# TUDO CONCLUÍDO + PLANEJADO — $(date)" >> "$J"
