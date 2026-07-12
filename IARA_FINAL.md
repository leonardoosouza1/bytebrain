# IARA FINAL — leve, inteligente, rápida, byte-nativa

> Fechamento do PROGRAMA IARA (2026-07-12). Runtime: `research/iara_final.py`.
> Diário completo (WS1→Fase 3b): `research/PROGRAMA_JOURNAL.md`.

## O que é

IARA (Inferência Autônoma com Roteamento Adaptativo) **não é um LLM maior**. É um sistema de
**órgãos alinhados** onde o modelo pesado **semeia** o conhecimento uma vez, e um conjunto
leve/rápido **serve** — auditável, byte-nativo e que não alucina.

```
pergunta
   │
   ├─ ÓRGÃO-BYTE   corrige TYPO por distância de edição em bytes (cobre as 75 entidades, sem modelo pesado)
   │
   ├─ NAVEGAÇÃO    água no grafo:  direto · agregado (aresta inversa) · multi-hop (região ∩ língua)
   │      │
   │      ▼
   │   MEMÓRIA     grafo semeado (139 arestas do 3B por auto-consistência + backbone geográfico) = 289 arestas, 7.6 KB
   │
   ├─ GERMINADOR   transformer BYTE com MBP (3.5M): germina a falta sob demanda e cristaliza a aresta (semente)
   │
   └─ VERIFICADOR  abstém em entidade fora do grafo (diz "não sei") — não alucina
```

Aprende com o uso: a sinapse fortalece no reuso (Hebbian) e um **conceito nasce** por co-ativação
(ex.: `capital@south america` vira um bundle instantâneo depois de regado).

## Os saltos desta versão

| Salto | O quê | Resultado honesto |
|---|---|---|
| **MTP → MBP** | germinador byte com K=4 cabeças prevendo os próximos 4 **bytes** | lookahead 97/96/96/95% → **~3.7× menos passos** (especulativo) |
| **Órgãos falam byte** | typo corrigido no substrato byte, cobrindo todas as entidades | **byte degrada -5pp** (95→90%) vs **token-3B -31pp** (93→63%) sob typo = ~6× mais robusto (head-to-head ao vivo, N=59) |
| **Memória semeada** | o 3B semeia o grafo 1× offline; runtime responde instantâneo | fato direto **~1.8 ms** (vs 7B no chat = 0.3 tok/s) |
| **Não alucina** | verificador abstém em entidade desconhecida | abstenção 2/2, 0 alucinação nos falsos |
| **Backbone objetivo** | continente/região são fato-terra (src=`geo`); capitais vêm do modelo | agregado e multi-hop 2/2 |

## Números (bateria + escala)

- Capacidade global da bateria: **100%** — fato direto 4/4 · typo 4/4 · agregado 2/2 · multi-hop 2/2 · abstenção 2/2
- Validação em escala (N=61 capitais reais): **limpo 95% · com typo 90%**
- Head-to-head ao vivo sob typo (N=59): **byte-IARA 95→90% (-5pp) vs token-3B 93→63% (-31pp)** = byte ~6× mais robusto
- MBP especulativo: **~3.7×** menos passos de geração
- Peso: **7.6 KB** de conhecimento + **3.5M** de germinador byte (vs 7B fp16 = 14 GB)
- Latência: grafo **instantâneo (~1.8 ms)**; germinador byte só na falta

## Onde substitui o LLM tradicional

Não é "um chatbot melhor". Ganha exatamente onde o LLM monolítico é fraco:

1. **Fatos auditáveis e instantâneos.** Consulta com fonte (`[3B]`/`[geo]`/`[byte-germ]`) em ~ms, sem
   rodar bilhões de params. Substitui LLM em busca factual, lookup estruturado, FAQ, roteamento.
2. **Custo/latência.** 7.6 KB + 3.5M cabem em qualquer edge; o modelo pesado só é chamado 1× para semear.
3. **Robustez a ruído de entrada.** Byte-nativo aguenta typo (90%) onde o BPE quebra — bom para entrada
   humana suja, OCR, logs, terminais.
4. **Não alucina por construção.** Abstém em vez de inventar — crítico onde uma resposta errada custa caro.
5. **Aprende sem re-treino.** Germina a falta, cristaliza a aresta, faz nascer conceito — o conhecimento
   cresce com o uso, não com um novo fine-tune.

**Limite honesto:** IARA é **seletor/roteador**, não gerador de texto longo. Para redação aberta e
raciocínio livre, o transformer generativo ainda ganha — a IARA o **usa como semeador**, não o substitui
nesse eixo. O produto é o casamento: pesado semeia, leve serve.

## Próximo passo natural (motor)

Portar o loop `iara_final.py` para o `iara-engine` (Rust): grafo em RAM + órgão-byte (edit-distance) +
germinador byte-MBP como kernel — servido no endpoint OpenAI-compat. É a IARA final rodando no nosso motor.
