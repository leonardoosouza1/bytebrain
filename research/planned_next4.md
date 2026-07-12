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
