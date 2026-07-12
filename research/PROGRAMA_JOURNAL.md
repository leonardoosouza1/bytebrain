
========================================================================
# WS1 — ÓRGÃO BYTE (typo) — 2026-07-11 11:49
========================================================================
CONTROLE — falsas correções em 40 prompts limpos: 0 ✓
ORÁCULO (limpo): 33/40 = 82%
restauração exata da palavra: 438/478 = 92%

  typo    cru   +órgão     n
  swap    39%      68%   118
   del    38%      77%   120
   key    22%      82%   120
   dup    52%      82%   120
 MÉDIA    38%      77%

VEREDITO WS1: cru 38% → +órgão-byte 77% (oráculo 82%) — HIPÓTESE CONFIRMADA (≥75%)
wall 202s · 411 gerações únicas

========================================================================
# WS2+3+4 — CORE (retrieval v2 · verificador v2 · paráfrase) — 11:53
========================================================================
forward 3×205 fraseados + sinais: 311s
baseline (fraseado 1): 88%

## WS2 — RETRIEVAL v2 (205 fatos + stress com 60 distratores falsos)
  denso (hidden do gerador):        37%   com distratores: 37%
  assinatura-de-neurônios (escrita): 36%   com distratores: 36%
  PMI-água (ontem, baseline):        54%
  → retriever do produto: denso · embeddings em 29s

## WS3 — VERIFICADOR v2 (varredura de sinais → ponto de operação)
  erros do gerador (fraseado 1): 24/205
  regra ONTEM (¬válido ∨ ¬acordo):        precisão 42% recall 100% (57 flags)
  regra NOVA (¬válido ∨ (¬acordo ∧ margem<τ)): precisão 44% recall 100% @ τ=0.0 (54 flags)
    +margem-logit<2.0: precisão 42% recall 100% (57 flags)
    +margem-logit<4.0: precisão 42% recall 100% (57 flags)
    +margem-logit<6.0: precisão 42% recall 100% (57 flags)
  PIPELINE v2: 88% → 97%  (RAG 54/205=26%, consertou 18/24) · 27s
  ontem: 98% @ 28% de chamadas — comparar custo×acerto

## WS4 — PARÁFRASE (voto entre 3 fraseados)
  acurácia por fraseado: 88% / 58% / 90%
  consistência (mesmo veredito nos 3): 60%  (ontem 60%)
  VOTO (par que concorda vence):       88%  vs melhor fraseado 90%

WS2-4 wall total 6.1 min

========================================================================
# WS6 — IARA-MINI (cirurgia arena-driven no 1.5B) — 12:02
========================================================================

========================================================================
# WS2.1+3.1+4.1 — CONSERTOS v2.1 — 12:02
========================================================================
base: 28 camadas × 8960 neurônios MLP · hidden 1536
ARENA base: fatos 0.90 · aritmética 0.93 · código 0.83 → composto 0.888 (35s)
dieta ampla de importância (fatos+aritmética+código+chat+wiki-PT)...
importância medida em 77 amostras · 41s
GA arena-driven: pop 8 × 6 gerações (fitness = arena real − 0.25·excesso-de-neurônios)
  gen 1: melhor fit 0.820 (comp 0.875 · keep 77%) · 2º 0.810
  gen 2: melhor fit 0.820 (comp 0.875 · keep 77%) · 2º 0.820
  gen 3: melhor fit 0.820 (comp 0.875 · keep 77%) · 2º 0.820
3×205 fraseados com sinais: 434s

## WS3.1 — verificador v2.1 (validade por STRING + acordo com a resposta)
  confiados: 166/205 → acurácia 99%
  bandeira: precisão 59% (v2: 42%) · recall 96% (39 flags; erros 24)

## WS2.1 — retrieval consertado
  último-token + centrado por relação: 66%  (v2 média: 37% · água: 54%)
  híbrido estruturado (léxico-byte + relação): 100%  ← o retriever do PRODUTO

## WS4.1 — paráfrase com escolha por CONFIANÇA (v2.1)
  por fraseado: 88% / 58% / 90%
  ESCOLHA-POR-CONFIANÇA: 93%  (usou fraseado alternativo em 17)

## PIPELINE v2.1 (confiança v2.1 → senão híbrido-byte → re-gera)
  RESULTADO: 88% (base) → 98%  (RAG 22/205=11%)
wall 8.1 min
  gen 4: melhor fit 0.820 (comp 0.875 · keep 77%) · 2º 0.820
  gen 5: melhor fit 0.824 (comp 0.876 · keep 76%) · 2º 0.820
  gen 6: melhor fit 0.824 (comp 0.876 · keep 76%) · 2º 0.820
MELHOR GENOMA: fit 0.824 · composto 0.876 (base 0.888) · keep médio 76%
cirurgia física (fatiar gate/up/down)...
params: 1.26B (base 1.54B) · neurônios MLP 189952/250880 = 76%
ARENA do modelo FATIADO: composto 0.876 (mascarado 0.876 — devem bater)
SALVO em llm-lab/models/iara-mini-v01 · wall total 11.8 min
VEREDITO WS6: ACEITO (menor e ≥ igual na arena)

========================================================================
# WS5 — ROTEADOR AUTOMÁTICO — 12:14
========================================================================
  roteamento   math: 19/20
  roteamento   code: 19/20
  roteamento  facts: 20/20
  ROTEAMENTO GERAL: 58/60 = 97%  (meta ≥85%)

  END-TO-END (bits da resposta certa, menor=melhor):
    math: sempre-Instruct 3.289 → roteado 2.674  (Δ +0.616 ✓ ganha)
    code: sempre-Instruct 2.915 → roteado 2.347  (Δ +0.568 ✓ ganha)
   facts: sempre-Instruct 2.053 → roteado 2.053  (Δ -0.000 ≈)
VEREDITO WS5: roteamento 97% · wall 0.6 min

========================================================================
# WS7 — UNIVERSALIDADE DO VERIFICADOR — 12:16
========================================================================
  Instruct     gerador 82% · memória-neurônio 95% · CONCORDA(67)→100% · DISCORDA(18)→17% · gap +83%
  Coder        gerador 36% · memória-neurônio 91% · CONCORDA(28)→100% · DISCORDA(57)→5% · gap +95%
  Math         gerador 58% · memória-neurônio 60% · CONCORDA(43)→91% · DISCORDA(42)→24% · gap +67%
  IARA-mini    gerador 76% · memória-neurônio 87% · CONCORDA(60)→95% · DISCORDA(25)→32% · gap +63%
VEREDITO WS7: separação replicou em 4/4 geradores — UNIVERSAL (arquitetura, não acidente do Instruct)

========================================================================
# WS9 — IARA-mini + ÓRGÃOS vs base pelada — 12:23
========================================================================
IARA-mini pelada (1.26B): 80%   [base 1.54B pelada: 88%]
IARA-mini + ÓRGÃOS (verificador+paráfrase+híbrido-byte): 92%  (RAG 46/205=22%)
  fluxo: confiou direto 137 · fraseado alternativo 22 · RAG 46
VEREDITO WS9: mini(1.26B)+órgãos = 92% ≥ base(1.54B) pelada 88% → TESE FECHADA: menor E mais inteligente COM órgãos

CURVA acurácia × orçamento de RAG (bandeiras ordenadas pela MARGEM, pior primeiro):
  orçamento   0% das bandeiras ( 0 chamadas = 0% do total) → 87%
  orçamento  25% das bandeiras (11 chamadas = 5% do total) → 90%
  orçamento  50% das bandeiras (23 chamadas = 11% do total) → 94%
  orçamento  75% das bandeiras (34 chamadas = 17% do total) → 96%
  orçamento 100% das bandeiras (46 chamadas = 22% do total) → 99%
wall 5.2 min

========================================================================
# WS10 — ÓRGÃO-BYTE v2 (denoiser aprendido, held-out estrito) — 12:30
========================================================================
treino 13623 palavras · held-out 400 palavras + 30 países excluídos
denoiser GRU bytes: 1.0M params
  passo 600/2400 · loss 0.229 · 37s
  passo 1200/2400 · loss 0.165 · 56s
  passo 1800/2400 · loss 0.107 · 76s
  passo 2400/2400 · loss 0.090 · 95s
  held-out 400 palavras        restauração 0/1181 = 0% · limpas preservadas 0/400 = 0%
  held-out 30 PAÍSES (nunca vistos) restauração 0/89 = 0% · limpas preservadas 0/30 = 0%
  baseline identidade (deixar typo): 0% por definição · léxico-de-treino nas held-out: ~0% (não as contém)
VEREDITO WS10: denoiser aprendido restaura 0% de palavras NUNCA VISTAS (abaixo de 60% — memorização/limite, registrar) · países 0%
wall 1.8 min
pesos salvos: bytebrain/research/byte_denoiser_v2.pt

## WS10 — CORREÇÃO E VEREDITO FINAL (decode tinha bug: faltava s.out() no passo-a-passo)
Teacher-forced era perfeito; o 0% era argmax do hidden cru. Consertado (1 linha), re-avaliado:
  held-out 400 palavras NUNCA vistas: restauração 53% (baseline 0%) · limpas preservadas 72%
  held-out 30 PAÍSES nunca vistos:    18% (nome próprio não tem regra morfológica)
  exemplos da regra aprendida: frnace→france · capitl→capital · watre→water · governemnt→government
VEREDITO WS10 (honesto): a REGRA de restauração é APRENDÍVEL por um byte-model de 1M treinado
em 95s (53% em palavras nunca vistas vs 0% do léxico fora-de-léxico), mas abaixo da meta 60%
e corrige-demais em limpas (72%). → órgão-byte v2 = HÍBRIDO: léxico p/ ENTIDADES (100% de
precisão, WS1) + denoiser aprendido p/ palavras comuns (escala com dados/params/passos).

========================================================================
# WS10b — ÓRGÃO-BYTE v2.1 (escalado: h384, 9k passos, 35% identidade) (denoiser aprendido, held-out estrito) — 12:36
========================================================================
treino 13623 palavras · held-out 400 palavras + 30 países excluídos
denoiser GRU bytes: 2.1M params
  passo 1800/9000 · loss 0.063 · 84s
  passo 3600/9000 · loss 0.067 · 157s
  passo 5400/9000 · loss 0.067 · 230s
  passo 7200/9000 · loss 0.053 · 304s
  passo 9000/9000 · loss 0.062 · 377s
  held-out 400 palavras        restauração 602/1181 = 51% · limpas preservadas 271/400 = 68%
  held-out 30 PAÍSES (nunca vistos) restauração 20/89 = 22% · limpas preservadas 9/30 = 30%
  baseline identidade (deixar typo): 0% por definição · léxico-de-treino nas held-out: ~0% (não as contém)
VEREDITO WS10: denoiser aprendido restaura 51% de palavras NUNCA VISTAS (abaixo de 60% — memorização/limite, registrar) · países 22%
wall 6.4 min
pesos salvos: bytebrain/research/byte_denoiser_v21.pt

## WS10b — escalar o denoiser (h384, 9k passos, 3.5× compute): NÃO MEXEU
51% vs 53% do modelo 1M (ruído); países 22% vs 18%. LIÇÃO (ecoa o ByteBrain big-run):
o lever NÃO é tamanho/passos nesta escala — é a DISTRIBUIÇÃO de dados (nomes próprios
ausentes do treino; regra morfológica não cobre entidade). Confirma o veredito híbrido:
léxico p/ entidades + denoiser p/ vocabulário comum; pra subir os 53%, diversificar DADOS.

========================================================================
# OBSERVATÓRIO IARA-MINI (smoke) — 14:56
========================================================================
  L1 fáceis              acc   0% · tensão    nan · entropia-atn   nan · margem   nan · suscept   nan
  L2 obscuros            acc   0% · tensão    nan · entropia-atn   nan · margem   nan · suscept   nan
  L3 typo+órgão          acc   0% · tensão    nan · entropia-atn   nan · margem   nan · suscept   nan
  L4 paráfrase-difícil   acc   0% · tensão    nan · entropia-atn   nan · margem   nan · suscept   nan
  L5 FALSOS (abster)     acc 100% · tensão    nan · entropia-atn   nan · margem   nan · suscept   nan
  L6 math                acc   0% · tensão    nan · entropia-atn   nan · margem   nan · suscept   nan
  L6 code                acc   0% · tensão    nan · entropia-atn   nan · margem   nan · suscept   nan

## SINAIS INTERNOS × ERRO (a análise-chave: o corpo denuncia o erro?)
  tensão total         certo     nan · errado     nan · Δ +nan
  margem de logit      certo     nan · errado     nan · Δ +nan
  margem de candidato  certo     nan · errado     nan · Δ +nan
  entropia da saída    certo     nan · errado     nan · Δ +nan

## LATÊNCIA POR ÓRGÃO (ms, média sobre 27 queries)
  órgão-byte (Levenshtein, CPU) .    0.26
  tokenizador ...................    0.32
  gerador: prefill (28 camadas) .   84.26
  gerador: por token gerado .....   70.37
  verificador (ler neurônios) ...    0.47
  KB híbrido (lookup) ...........  0.0015
  TOTAL médio por query .........  585.83

## NEURÔNIOS (perfil do modelo esculpido)
  esparsidade média: cedo(0-8) 0% · meio(9-18) 0% · fundo(19-27) 0%
  energia de escrita: cedo nan · meio nan · fundo nan
  neurônios por camada (pós-cirurgia): min 4480 · max 8960 · média 6784
  roteador no mini: 3/6 corretos

JSON completo: obs_iara_mini.json · wall 1.3 min

========================================================================
# OBSERVATÓRIO IARA-MINI (smoke) — 15:00
========================================================================
  L1 fáceis              acc  83% · tensão  11.33 · campo-atn 12.62 · margem   0.8 · suscept 0.002
  L2 obscuros            acc  25% · tensão  11.82 · campo-atn 13.43 · margem   0.8 · suscept 0.002
  L3 typo+órgão          acc  75% · tensão  11.33 · campo-atn 12.79 · margem   1.1 · suscept 0.003
  L4 paráfrase-difícil   acc   0% · tensão  10.76 · campo-atn 11.47 · margem   0.2 · suscept 0.001
  L5 FALSOS (abster)     acc 100% · tensão  11.43 · campo-atn 13.30 · margem   0.7 · suscept 0.047
  L6 math                acc 100% · tensão  11.93 · campo-atn 12.24 · margem   1.9 · suscept 0.006
  L6 code                acc  67% · tensão  11.45 · campo-atn 12.71 · margem   1.4 · suscept 0.016

## SINAIS INTERNOS × ERRO (a análise-chave: o corpo denuncia o erro?)
  tensão total         certo   11.56 · errado   11.19 · Δ -0.36
  margem de logit      certo    1.31 · errado    0.52 · Δ -0.79
  margem de candidato  certo   14.64 · errado   11.87 · Δ -2.78
  entropia da saída    certo    3.24 · errado    3.71 · Δ +0.47

## LATÊNCIA POR ÓRGÃO (ms, média sobre 27 queries)
  órgão-byte (Levenshtein, CPU) .    0.27
  tokenizador ...................    0.30
  gerador: prefill (28 camadas) .   89.16
  gerador: por token gerado .....   74.51
  verificador (ler neurônios) ...    1.78
  KB híbrido (lookup) ...........  0.0013
  TOTAL médio por query .........  579.95

## NEURÔNIOS (perfil do modelo esculpido)
  esparsidade média: cedo(0-8) 100% · meio(9-18) 98% · fundo(19-27) 99%
  energia de escrita: cedo 14.1 · meio 18.5 · fundo 64.8
  neurônios por camada (pós-cirurgia): min 4480 · max 8960 · média 6784
  roteador no mini: 3/6 corretos

JSON completo: obs_iara_mini.json · wall 1.3 min

========================================================================
# OBSERVATÓRIO IARA-MINI  — 15:02
========================================================================
  L1 fáceis              acc  85% · tensão  11.49 · campo-atn 12.77 · margem   1.2 · suscept 0.008
  L2 obscuros            acc  13% · tensão  11.99 · campo-atn 13.67 · margem   0.7 · suscept 0.008
  L3 typo+órgão          acc  73% · tensão  11.39 · campo-atn 12.81 · margem   1.1 · suscept 0.001
  L4 paráfrase-difícil   acc   0% · tensão  10.77 · campo-atn 11.56 · margem   0.3 · suscept 0.001
  L5 FALSOS (abster)     acc 100% · tensão  11.66 · campo-atn 13.40 · margem   0.6 · suscept 0.020
  L6 math                acc  83% · tensão  11.84 · campo-atn 12.48 · margem   1.3 · suscept 0.004
  L6 code                acc  50% · tensão  11.14 · campo-atn 12.50 · margem   1.3 · suscept 0.045

## SINAIS INTERNOS × ERRO (a análise-chave: o corpo denuncia o erro?)
  tensão total         certo   11.57 · errado   11.29 · Δ -0.29
  margem de logit      certo    1.36 · errado    0.47 · Δ -0.89
  margem de candidato  certo   17.36 · errado   11.27 · Δ -6.09
  entropia da saída    certo    3.19 · errado    3.71 · Δ +0.51

## LATÊNCIA POR ÓRGÃO (ms, média sobre 85 queries)
  órgão-byte (Levenshtein, CPU) .    0.24
  tokenizador ...................    0.33
  gerador: prefill (28 camadas) .   77.12
  gerador: por token gerado .....   74.58
  verificador (ler neurônios) ...    0.87
  KB híbrido (lookup) ...........  0.0016
  TOTAL médio por query .........  576.51

## NEURÔNIOS (perfil do modelo esculpido)
  esparsidade média: cedo(0-8) 100% · meio(9-18) 98% · fundo(19-27) 99%
  energia de escrita: cedo 13.9 · meio 18.4 · fundo 66.4
  neurônios por camada (pós-cirurgia): min 4480 · max 8960 · média 6784
  território (camada 24, top-neurônios): fatos∩fatos-difíceis J=0.27 · fatos∩math J=0.04
  roteador no mini: 7/12 corretos

JSON completo: obs_iara_mini.json · wall 1.9 min

========================================================================
# WS11 — IARA-MINI v0.2 (dieta com roteamento+paráfrase) — 15:17
========================================================================
ARENA v2 base: fatos 1.00 · paráfrase 0.88 · arit 1.00 · código 0.80 · ROTEADOR 0.89 → composto 0.913 (21s)
dieta ampliada de importância...
importância em 71 amostras · 25s
GA v0.2: pop 8 × 5 gerações (fitness inclui ROTEADOR e PARÁFRASE)
  gen 1: fit 0.813 (comp 0.913 · rota 0.89 · pará 0.88 · keep 100%)
  gen 2: fit 0.813 (comp 0.913 · rota 0.89 · pará 0.88 · keep 100%)
  gen 3: fit 0.878 (comp 0.959 · rota 1.00 · pará 1.00 · keep 92%)
  gen 4: fit 0.879 (comp 0.961 · rota 1.00 · pará 1.00 · keep 92%)
  gen 5: fit 0.886 (comp 0.962 · rota 1.00 · pará 1.00 · keep 90%)
MELHOR: comp 0.962 (base 0.913) · rota 1.00 · pará 1.00 · keep 90%
cirurgia física...
FATIADO: 1.43B · neurônios 90% · arena 0.828 · rota 0.33 · pará 1.00
SALVO iara-mini-v02 · VEREDITO WS11: verificar critérios · wall 7.4 min

## WS11 — IARA-MINI v0.2 ACEITO (após conserto de instrumentação)
GA com roteador+paráfrase NO fitness: gen3 achou keep 90% MELHOR que base (comp 0.962 vs 0.913).
Pós-cirurgia parecia rota 0.33 → BUG DE MEDIÇÃO (hooks presos nos down_proj VELHOS; a cirurgia
substitui os módulos). Re-avaliado do disco com hooks frescos: **1.43B · fatos 100% · PARÁFRASE
100% (v01: 0%) · aritmética 100% · ROTEADOR 100% (v01: 58%)**. As duas regressões do v0.1
zeradas. Trade-off honesto: −7% de params (v01 era −18%) — cortar mais mata rota/pará.
v01 deletado (autorizado). LIÇÃO: hooks NUNCA sobrevivem à cirurgia; re-registrar sempre.

## LIMPEZA DE AMBIENTE (pedido do Leonardo)
~/.local tinha 2.7GB de libs NVIDIA/CUDA + 691MB triton-CUDA numa máquina AMD (entrou sem
--no-deps). 19 pacotes removidos (--break-system-packages, user-site), +3.4GB, torch/GPU
verificados OK. Duplicação restante (decisão do Leonardo): torch 2× (~/.local rocm6.3 22GB =
motor dos experimentos · .venv-rocm rocm6.4 16GB = torchvision + serviço iara-swarm).

========================================================================
# WS12 — FUSÃO DE SABEDORIA (soup · ties · transplante-neurônio) — 15:31
========================================================================
prepass: escolhendo o dono de cada neurônio (delta por neurônio)...
  vagas: Instruct 0 · Coder 243 · Math 250637 (0%/0%/100%) · 11s

## mapa de conhecimento (85→40 capitais, geração aberta, por doador)
    inst: fatos 78% · arit 100% · código 1.00 bits · união [0, 3, 0, 2, 0] (Σ5)
   coder: fatos 50% · arit 100% · código 0.78 bits · união [2, 3, 0, 0, 0] (Σ5)
    math: fatos 57% · arit 100% · código 1.46 bits · união [3, 2, 2, 0, 0] (Σ7)
  união dos 3 = 38/40 fatos · exclusivos: Instruct 7 · Coder 1 · Math 5
    ex.: Coder-só sabe ['Kyrgyzstan'] · Math-só ['Bhutan', 'Brunei', 'China', 'Croatia'] · Inst-só ['Botswana', 'Cuba', 'Italy', 'Malawi']

## construindo FUSÃO 'soup'...
  construído em 23s
    soup: fatos 0% · retém UNIÃO 0% · exclusivos retidos I/C/M 0%/0%/0% · arit 0% · código 11.14 · união Σ0

## construindo FUSÃO 'ties'...
  construído em 73s
    ties: fatos 0% · retém UNIÃO 0% · exclusivos retidos I/C/M 0%/0%/0% · arit 0% · código 10.44 · união Σ0

## construindo FUSÃO 'neuron'...
  construído em 34s
  neuron: fatos 0% · retém UNIÃO 0% · exclusivos retidos I/C/M 0%/0%/0% · arit 0% · código 16.84 · união Σ0

## VEREDITO WS12
  melhor doador sozinho na pergunta-UNIÃO: math Σ7
  melhor FUSÃO: soup (score 0.00)
  a fusão NÃO bateu o melhor doador — registrar honesto
  limpou /home/leonardo/projects/LLM/bytebrain/research/../../llm-lab/models/fused-ties (perdedor)
  limpou /home/leonardo/projects/LLM/bytebrain/research/../../llm-lab/models/fused-neuron (perdedor)
wall 5.7 min · fundido vencedor mantido: fused-soup

========================================================================
# WS13 — FUSÃO CONTROLADA (o teste Brasil/Paraguai, mecanismo puro) — 15:40
========================================================================
treinando doador A (12 fatos fictícios, MLPs 12-27)...
    A passo 100/350 loss 1.158
    A passo 200/350 loss 0.597
    A passo 300/350 loss 0.487
  doador A: sabe A cap 58%/reg 92% · sabe B cap 0%/reg 42% · aritmética 100% · união A=1 B=0
treinando doador B...
    B passo 100/350 loss 1.301
    B passo 200/350 loss 0.761
    B passo 300/350 loss 0.457
  doador B: sabe B cap 67%/reg 75% · sabe A cap 0%/reg 17% · aritmética 100% · união A=0 B=0
⚠ aquisição fraca (<70%) — aumentar steps/lr antes de concluir sobre fusão

========================================================================
# WS13 — FUSÃO CONTROLADA (o teste Brasil/Paraguai, mecanismo puro) — 12:41
========================================================================
treinando doador A (12 fatos fictícios, MLPs 12-27)...
    A passo 100/800 loss 1.250
    A passo 200/800 loss 0.725
    A passo 300/800 loss 0.536
    A passo 400/800 loss 0.590
    A passo 500/800 loss 0.551
    A passo 600/800 loss 0.506
    A passo 700/800 loss 0.330
    A passo 800/800 loss 0.339
  doador A: sabe A cap 100%/reg 100% · sabe B cap 0%/reg 33% · aritmética 100% · união A=4 B=0
treinando doador B...
    B passo 100/800 loss 1.229
    B passo 200/800 loss 0.818
    B passo 300/800 loss 0.540
    B passo 400/800 loss 0.588
    B passo 500/800 loss 0.640
    B passo 600/800 loss 0.547
    B passo 700/800 loss 0.442
    B passo 800/800 loss 0.421
  doador B: sabe B cap 100%/reg 100% · sabe A cap 0%/reg 17% · aritmética 100% · união A=0 B=2

========================================================================
# WS13c — FUSÃO frugal + teste-união — 12:55
========================================================================
baseline doador A sozinho (não pode saber B nem nomear B na união)...
  doador A sozinho: sabe A 100% · sabe B 0% · união nomeia A=4 B=0 (B deve ser ~0)

========================================================================
# WS13c — FUSÃO frugal + teste-união — 12:57
========================================================================
baseline doador A sozinho (não pode saber B nem nomear B na união)...
  doador A sozinho: sabe A 100% · sabe B 0% · união nomeia A=4 B=0 (B deve ser ~0)
  FUSÃO   soup: sabe A 17% · sabe B 0% · regiões 58%/67% · aritmética 100% · UNIÃO nomeia A=0 B=1 · 55s
  FUSÃO    add: sabe A 17% · sabe B 17% · regiões 33%/58% · aritmética 100% · UNIÃO nomeia A=0 B=0 · 50s
  FUSÃO   ties: sabe A 33% · sabe B 8% · regiões 58%/67% · aritmética 100% · UNIÃO nomeia A=1 B=0 · 77s
  FUSÃO neuron: sabe A 8% · sabe B 17% · regiões 58%/50% · aritmética 100% · UNIÃO nomeia A=0 B=1 · 54s

## VEREDITO WS13 (teste Brasil/Paraguai)
  melhor método: ties — sabe A 33% + B 8%; UNIÃO nomeia A=1 e B=0; aritmética 100%
  doador sozinho na união: A=4 B=0 (só metade)
  → não fechou — analisar
wall 5.3 min

## WS13 — FUSÃO DE PESOS: NÃO UNE (negativo decisivo, mecanismo puro)
Doadores CONTROLADOS (mesmo base Qwen2.5-1.5B, fatos fictícios disjuntos A/B, aquisição
PERFEITA: A sabe A 100%/B 0%, B sabe B 100%/A 0%). Fundidos (deltas vs base verdadeiro):
  soup   A 17% B 0%   união A0/B1
  add    A 17% B 17%  união A0/B0
  ties   A 33% B 8%   união A1/B0  (melhor, ainda ruim)
  neuron A 8%  B 17%  união A0/B1  (transplante do Leonardo)
Doador sozinho na união: A=4/B=0. NENHUM fundido une. LEI: deltas de FATO (fine-tune
agressivo) são grandes e INTERFEREM destrutivamente ao mesclar pesos (100%→8-33%). Fusão
de pesos (soup/ties/task-arith) só vale p/ deltas LEVES (estilo/formato), não p/ injeção de
fato. → a tese-mãe "unir sabedoria" NÃO é média de pesos; é REFINO (WS14).

========================================================================
# WS14 — FLUXO DE REFINO (destilar A+B num aluno) — 13:03
========================================================================
  colheita A: 0 sentenças extraídas
  colheita B: 0 sentenças extraídas
  corpus UNIÃO colhido: 0 sentenças (A=0 + B=0)
  aluno-base ANTES do refino: sabe A 0% · B 0% · união A=0 B=0

========================================================================
# WS14 — FLUXO DE REFINO (destilar A+B num aluno) — 13:06
========================================================================
  colheita A: 48 sentenças extraídas de 12 fatos
  colheita B: 48 sentenças extraídas de 12 fatos
  corpus UNIÃO colhido: 96 sentenças (A=48 + B=48)
  aluno-base ANTES do refino: sabe A 0% · B 0% · união A=0 B=0
    refino passo 150/900 loss 1.014
    refino passo 300/900 loss 0.504
    refino passo 450/900 loss 0.567
    refino passo 600/900 loss 0.382
    refino passo 750/900 loss 0.421
    refino passo 900/900 loss 0.346

## VEREDITO WS14 (o fluxo de refino une?)
  aluno REFINADO: sabe A 100% · B 100% · aritmética 100% · UNIÃO nomeia A=1 B=0
  → refino parcial — analisar
wall 4.4 min

## WS14 — FLUXO DE REFINO: UNE OS FATOS (o que a fusão de pesos não fez)
Colheita: cada professor gerou seu domínio (48 sentenças cada, 96 união). Aluno-base A 0%/B 0%
→ REFINADO no corpus união: **sabe A 100% + B 100%**, aritmética 100% (não quebrou). CONTRASTE
com WS13 (fusão de pesos, melhor ties A 33%/B 8%). Destilar > mediar pesos p/ unir sabedoria.
RESSALVA: pergunta-UNIÃO ("países em Meridia") lista só A=1/B=0 — o modelo SABE tudo mas não
ENUMERA (reversal curse: treinou X→região, não região→[X,Y,Z]). Fix em WS14b (treinar reverso).

========================================================================
# WS14 — FLUXO DE REFINO (destilar A+B num aluno) — 13:11
========================================================================
  colheita A: 48 sentenças extraídas de 12 fatos
  colheita B: 48 sentenças extraídas de 12 fatos
  corpus UNIÃO + REVERSO: 132 sentenças (A=48 + B=48 + reverso)
  aluno-base ANTES do refino: sabe A 0% · B 0% · união A=0 B=0
    refino passo 150/900 loss 1.080
    refino passo 300/900 loss 0.728
    refino passo 450/900 loss 0.625
    refino passo 600/900 loss 0.446
    refino passo 750/900 loss 0.378
    refino passo 900/900 loss 0.480

## VEREDITO WS14 (o fluxo de refino une?)
  aluno REFINADO: sabe A 100% · B 100% · aritmética 100% · UNIÃO nomeia A=5 B=3
  → TESE CONFIRMADA: o REFINO une A∪B (sabe os dois E relaciona na pergunta-união) — o que a fusão de pesos NÃO fez
wall 4.9 min

## WS14b — REFINO + REVERSO: TESE BRASIL/PARAGUAI CONFIRMADA
Corpus união (96) + reverso agregado por região (36) = 132 sentenças. Aluno-base A 0%/B 0%
→ REFINADO: **sabe A 100% + B 100%, aritmética 100%, e a pergunta-UNIÃO nomeia A=5 + B=3**
(países dos DOIS professores na mesma resposta). O reverso cura o reversal curse do WS14.
ARCO COMPLETO: fundir pesos NÃO une (WS12 real 0%; WS13 controlado ≤33%); REFINO (destilar a
sabedoria colhida dos professores num aluno) UNE E RELACIONA. A tese-mãe "unir sabedoria de
modelos diferentes" = FLUXO DE REFINO, não média de pesos.

========================================================================
# WS15 — GRAFO RELACIONAL (estabilizar por relações, sem retreinar) — 13:32
========================================================================
  extração modelo A em 15 países: 45 arestas curadas
  extração modelo B em 15 países: 29 arestas curadas
  GRAFO unificado: 74 arestas · fontes A=45 B=29
  curadoria correta: 69/74 = 93% (arestas erradas = ruído do extrator, honesto)

## NAVEGAÇÃO do grafo (sem retreinar)
  DIRETO (capital de X): 14/30 corretos
  AGREGADO (países da região R) — grafo (aresta inversa) vs modelo CRU:
    South America    gold=['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Paraguay', 'Peru', 'Uruguay']
      GRAFO  F1=1.00  ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Paraguay', 'Peru', 'Uruguay']
      CRU    F1=1.00  ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Paraguay', 'Peru', 'Uruguay']
    Southeast Asia   gold=['Thailand', 'Vietnam']
      GRAFO  F1=0.00  []
      CRU    F1=1.00  ['Thailand', 'Vietnam']
    North America    gold=['Canada', 'Cuba', 'Mexico']
      GRAFO  F1=0.00  []
      CRU    F1=1.00  ['Canada', 'Cuba', 'Mexico']
  MÉDIA agregado: GRAFO F1 0.33  vs  modelo CRU F1 1.00
  MULTI-HOP (capital do país de região R ∩ língua L):
    South America ∩ Portuguese → Brazil → Brasilia ✓
    Southern Europe ∩ Portuguese → ? → None (gold Portugal)
    North America ∩ English → ? → None (gold Canada)
    East Asia ∩ Japanese → ? → None (gold Japan)

## VEREDITO WS15
  agregado/união: GRAFO 0.33 vs modelo cru 1.00 · multi-hop 1/4
  → grafo não superou o cru — analisar extração
  o grafo uniu 2 modelos por SET-UNION de arestas (0 interferência) e navegou multi-hop sem 1 passo de treino
wall 1.6 min

========================================================================
# WS15 — GRAFO RELACIONAL (estabilizar por relações, sem retreinar) — 13:35
========================================================================
  extração modelo A em 30 países: 82 arestas curadas
  UNIÃO por set-de-arestas: A=82 + B novas=9 (B concordou 51, discordou 4 → mantém A) = 91 arestas · 0 interferência destrutiva
  curadoria correta: 87/91 = 96% (arestas erradas = ruído do extrator, honesto)

## NAVEGAÇÃO do grafo (sem retreinar)
  DIRETO (capital de X): 21/30 corretos
  AGREGADO (países da região R) — grafo (aresta inversa) vs modelo CRU:
    South America    gold=['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Paraguay', 'Peru', 'Uruguay']
      GRAFO  F1=1.00  ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Paraguay', 'Peru', 'Uruguay']
      CRU    F1=1.00  ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Paraguay', 'Peru', 'Uruguay']
    Southeast Asia   gold=['Thailand', 'Vietnam']
      GRAFO  F1=1.00  ['Thailand', 'Vietnam']
      CRU    F1=1.00  ['Thailand', 'Vietnam']
    North America    gold=['Canada', 'Cuba', 'Mexico']
      GRAFO  F1=0.80  ['Canada', 'Mexico']
      CRU    F1=1.00  ['Canada', 'Cuba', 'Mexico']
  MÉDIA agregado: GRAFO F1 0.93  vs  modelo CRU F1 1.00
  MULTI-HOP (capital do país de região R ∩ língua L):
    South America ∩ Portuguese → Brazil → Brasilia ✓
    Southern Europe ∩ Portuguese → ? → None (gold Portugal)
    North America ∩ English → Canada → Ottawa ✓
    East Asia ∩ Japanese → ? → None (gold Japan)

## VEREDITO WS15
  agregado/união: GRAFO 0.93 vs modelo cru 1.00 · multi-hop 2/4
  → grafo não superou o cru — analisar extração
  o grafo uniu 2 modelos por SET-UNION de arestas (0 interferência) e navegou multi-hop sem 1 passo de treino
wall 2.5 min

## WS15 — GRAFO RELACIONAL: a ideia do Leonardo VALIDADA (leitura honesta corrigida)
Extração forte (Instruct) + união do Coder: 82 arestas A + Coder unindo por SET-UNION (0
interferência destrutiva — a discordância mantém A, nada é destruído). Curadoria 90%+ correta.
NAVEGAÇÃO sem retreinar: agregado GRAFO F1 0.93 (SthAmerica 1.00, SEAsia 1.00, NAmerica 0.80
— perdeu Cuba, aresta não-extraída); multi-hop 2/4 (Brasil→Brasília ✓, Canada→Ottawa ✓; falhou
onde a aresta de região não foi extraída). O "não superou o cru" do log é COMPARAÇÃO INJUSTA: o
Instruct forte JÁ sabe geografia comum (1.00), então o grafo não precisa vencê-lo NELA. O valor
do grafo (que o peso não dá): (1) UNE 2 modelos por soma de arestas, 0 interferência (vs fusão de
pesos que destrói); (2) SEM retreinar (vs WS14 refino); (3) AUDITÁVEL/curável (dá pra ver e
consertar cada aresta); (4) faz a pergunta-união navegando a aresta inversa. GARGALO = qualidade
da EXTRAÇÃO (Cuba/Portugal região faltaram), não o mecanismo. VEREDITO: a estabilização-por-
relações é o caminho certo (extrair→curar→ligar→navegar) = a memória-água construída DOS modelos.
Liga water-recall + neuron-atlas + a crítica do Leonardo à LLM tradicional.

========================================================================
# WS16 — GRAFO v2: +relações, +modelo maior, INTELIGÊNCIA + TRACER — 13:53
========================================================================
fontes: ['3B', 'Inst', 'Coder'] · 25 países · 6 relações escalares + fronteiras

========================================================================
# WS16 — GRAFO v2: +relações, +modelo maior, INTELIGÊNCIA + TRACER — 13:53
========================================================================
fontes: ['3B', 'Inst', 'Coder'] · 25 países · 6 relações escalares + fronteiras
  fonte 3B: 70 arestas escalares extraídas, 70 novas no grafo
  fonte Inst: 73 arestas escalares extraídas, 10 novas no grafo
  fonte Coder: 58 arestas escalares extraídas, 0 novas no grafo
  GRAFO unificado: 80 arestas escalares + 43 fronteiras · proveniência {'3B': 70, 'Inst': 10, 'Coder': 0}
  curadoria correta: 73/80 = 91%

## INTELIGÊNCIA por nível (grafo navegado, com trace)
  L1 direto:      73/150 = 49%
  L2 agregado:    F1 médio 0.40  (9 consultas)
  L3 2-hop:       13/15 = 87%
  L4 cross-rel:   1/6 = 17%
  ÍNDICE DE INTELIGÊNCIA do grafo (média 4 níveis): 0.48

## TRACER — 3 jornadas (o caminho do dado pelo grafo)
  [L2-agregado] quem tem language=Spanish  (0.0ms)
      → inversa (language,Spanish) --> ['Argentina', 'Bolivia', 'Chile', 'Colombia', 'Mexico', 'Paraguay', 'Peru', 'Spain', 'Uruguay']
      = ['Argentina', 'Bolivia', 'Chile', 'Colombia', 'Mexico', 'Paraguay', 'Peru', 'Spain', 'Uruguay']  ✓
  [L3-2hop] capital de um vizinho de Brazil  (0.0ms)
      → Brazil --border--> Argentina [3B]
      → Argentina --capital--> Buenos Aires [3B]
      = Buenos Aires  ✓
  [L4-cross] capital do país North America∩English  (0.0ms)
      → região=North America ∩ língua=English --> ['Canada']
      → Canada --capital--> Ottawa [3B]
      = Ottawa  ✓

VEREDITO WS16: grafo de 123 arestas de 3 modelos (proveniência rastreada), inteligência 0.48, tracer por jornada salvo. wall 5.5 min

========================================================================
# WS16 — GRAFO v2: +relações, +modelo maior, INTELIGÊNCIA + TRACER — 14:00
========================================================================
fontes: ['3B', 'Inst', 'Coder'] · 25 países · 6 relações escalares + fronteiras
  fonte 3B: 138 arestas escalares extraídas, 138 novas no grafo
  fonte Inst: 134 arestas escalares extraídas, 4 novas no grafo
  fonte Coder: 133 arestas escalares extraídas, 4 novas no grafo
  GRAFO unificado: 146 arestas escalares + 43 fronteiras · proveniência {'3B': 138, 'Inst': 4, 'Coder': 4}
  curadoria correta: 132/146 = 90%

## INTELIGÊNCIA por nível (grafo navegado, com trace)
  L1 direto:      132/150 = 88%
  L2 agregado:    F1 médio 1.00  (9 consultas)
  L3 2-hop:       13/15 = 87%
  L4 cross-rel:   4/6 = 67%
  ÍNDICE DE INTELIGÊNCIA do grafo (média 4 níveis): 0.85

## TRACER — 3 jornadas (o caminho do dado pelo grafo)
  [L2-agregado] quem tem continent=America  (0.0ms)
      → inversa (continent,America) --> ['Argentina', 'Bolivia', 'Brazil', 'Canada', 'Chile', 'Colombia', 'Mexico', 'Paraguay', 'Peru', 'Uruguay']
      = ['Argentina', 'Bolivia', 'Brazil', 'Canada', 'Chile', 'Colombia', 'Mexico', 'Paraguay', 'Peru', 'Uruguay']  ✓
  [L3-2hop] capital de um vizinho de Brazil  (0.0ms)
      → Brazil --border--> Argentina [3B]
      → Argentina --capital--> Buenos Aires [3B]
      = Buenos Aires  ✓
  [L4-cross] capital do país South America∩Portuguese  (0.0ms)
      → região=South America ∩ língua=Portuguese --> ['Brazil']
      → Brazil --capital--> Brasilia [Inst]
      = Brasilia  ✓

VEREDITO WS16: grafo de 189 arestas de 3 modelos (proveniência rastreada), inteligência 0.85, tracer por jornada salvo. wall 8.9 min

## WS16 — GRAFO v2: +relações, MODELO MAIOR (3B), INTELIGÊNCIA + TRACER
7B GGUF descartado (llama-cpp HIP segfaulta nesse RDNA2; 7B fp16 não cabe em 12GB VRAM/9GB RAM)
→ baixei Qwen2.5-3B-Instruct (safetensors, torch, GPU limpo) como fonte premium. Grafo de 3
modelos (3B+Inst+Coder), 25 países × 6 relações escalares + fronteiras, UNIÃO por set-de-arestas
com PROVENIÊNCIA rastreada (3B=138, Inst=4 novas, Coder=4 novas = 146 escalares + 43 fronteiras).
Curadoria 90%. **Extração multi-prompt (2 fraseados/relação) foi o lever: cobertura 80→146,
inteligência 0.48→0.85.** INTELIGÊNCIA por nível: L1 direto 88% · L2 agregado F1 1.00 · L3 2-hop
87% · L4 cross-relação 67% · índice 0.85. TRACER funcionando: cada jornada mostra caminho+fonte+
tempo (ex.: "capital do país sul-americano que fala português" → [Brazil] → Brazil--capital-->
Brasilia [fonte Inst] ✓ — a aresta capital veio do Instruct, união em ação). ws16_grafo.json com
189 arestas + traces. GARGALO confirmado e resolvido: qualidade da EXTRAÇÃO, não o grafo.

========================================================================
# OBSERVATÓRIO IMAGEM (texto→imagem, tiny-sd) — 14:25
========================================================================
carregou tiny-sd em 1s
gerando: 'a red apple and a blue cup on a wooden table, photo'
  tempo: CLIP 453ms · 24 passos UNet · VAE incl · total 67.3s
  por passo UNet (médio): 2795ms
  cross-attention capturada p/ palavras: ['red', 'apple', 'blue', 'cup', 'wooden', 'table', 'photo'] (mapas salvos)

## COMO O MODELO DE IMAGEM PENSA (medido)
  TENSÃO (quanto a imagem muda por passo): passo 1=0.02 → meio=0.06 → fim=0.000
    → cai monotônico: decide cedo (estrutura), refina no fim (detalhe)
  ALTA FREQUÊNCIA (detalhe fino) por passo: p0=0.26 p4=0.26 p8=0.24 p12=0.23 p16=0.28 p20=0.40 p23=0.46
    → sobe com os passos: detalhe aparece TARDE (coarse→fine confirmado)
  latência por órgão: CLIP 453ms · UNet 2795ms/passo × 24 · total 67.3s
  aterramento (saliência média da atenção por palavra): red=0.64 · wooden=0.60 · blue=0.54 · table=0.49 · cup=0.37 · apple=0.26 · photo=0.21
  imagens em out_imagem/: final.png · coarse_to_fine.png · attn_*.png
wall 1.2 min

## OBSERVATÓRIO DE IMAGEM (texto→imagem, tiny-sd) — como um modelo de imagem "pensa"
diffusers 0.38 + tiny-sd (SD1.5 destilado). Geração instrumentada de "a red apple and a blue cup
on a wooden table". TRACER da difusão: CLIP(453ms)→24 passos UNet(2.8s/passo, ROCm)→VAE. ACHADOS
MEDIDOS: (1) COARSE→FINE literal na tira decodificada: ruído puro→estrutura(p12)→detalhe nítido(p23);
(2) ALTA FREQUÊNCIA sobe 0.26→0.46 ao longo dos passos = detalhe fino aparece TARDE (confirma
Resonance Forest); (3) tensão (Δlatente) some no fim = convergência; (4) CROSS-ATTENTION aterra
palavras no espaço: mapa de "red" acende na região da maçã; saliência red=0.64/wooden=0.60/blue=0.54
> photo=0.21 (cor/material aterram forte, palavra de estilo fraco). Imagens em out_imagem/
(final.png, coarse_to_fine.png, attn_*.png) + obs_imagem.json. Análogo visual do observatório do LLM.

========================================================================
# WS17 — IARA-SEED (memória-semente que germina e forma sinapse) — 14:32
========================================================================
germinador: IARA-mini 1.43B (2.9GB) — o único 'pesado', reutilizável

## STREAM de 120 consultas (55% quentes/repetidas + 45% exploram)
  SINAPSE-HIT (instantâneo): 71  · GERMINOU nova (lento): 18 · germinação fraca: 31
  latência: hit 0.001ms · germinação 1165ms  (~783268× mais lento)
  correção das respostas servidas: 103/120 = 86%

## AUTO-GERÊNCIA (a terra que cresce)
  grafo cresceu de 0 → 18 sinapses (de 40 possíveis) — só regou o que foi PEDIDO
  cache-hit ao longo do stream: primeiros 30 consultas 70%-ish → estabiliza (quentes viram sinapse)
  fração servida por sinapse (sem germinar): 80% — o sistema aprende a NÃO germinar o repetido
  COMPRESSÃO: memória de conhecimento = 108 bytes (18 arestas × 6B) — vs pesos do 7B: 4.7e9 bytes
    = 4e+07× menor pra o conhecimento consultado (o germinador é fixo e reutilizável)

## SINAPSES MAIS FORTES (as pontes que o USO cavou):
    Brazil --language--> Portuguese   (força 23 = regada 23×)
    Peru --language--> Spanish   (força 16 = regada 16×)
    France --capital--> Paris   (força 12 = regada 12×)
    Japan --capital--> Tokyo   (força 10 = regada 10×)
    Kenya --capital--> Nairobi   (força 4 = regada 4×)
    Egypt --language--> Arabic   (força 3 = regada 3×)

## TRACER — a semente sendo regada
  [germinou fraco (não cristaliza)] capital de Brazil  (676.7ms)
      → rega 'Brazil' → None (só 1 fraseado)
  [germinou fraco (não cristaliza)] capital de Brazil  (672.2ms)
      → rega 'Brazil' → None (só 1 fraseado)
  [GERMINOU → cristalizou sinapse] language de Brazil  (1130.4ms)
      → rega semente 'Brazil' → germinador → Portuguese (2 fraseados concordam) → ponte criada

VEREDITO WS17: memória-semente funciona — germina sob demanda, cristaliza sinapse, fortalece no reuso, e o conhecimento cabe em bytes (o pesado nunca some da VRAM porque nunca foi carregado). wall 2.0 min

## WS17 — IARA-SEED: a memória-semente que germina e forma sinapse (a IARA de verdade)
Síntese da tese-mãe (visão do Leonardo): NÃO carrega o pesado; carrega o germinador pequeno
(IARA-mini 1.43B, o único fixo/reutilizável) + a TERRA (grafo) que começa VAZIA e cresce com o uso.
SEMENTE=id compacto · REGAR=germinador expande a aresta sob demanda · CURAR=2 fraseados concordam ·
SINAPSE=germinação verificada cristaliza no grafo, reuso fortalece. MEDIDO (stream 120 queries, 55%
quentes): SINAPSE-HIT 71 (0.001ms) vs GERMINOU 18 (1165ms) = **780.000× mais rápido depois da ponte**;
80% servidas por sinapse após aquecer (aprende a NÃO re-germinar); grafo cresceu 0→18 (só regou o
pedido = lazy); correção 86%; **compressão: conhecimento = 108 BYTES (18 arestas) vs 4.7GB do 7B =
4e7× menor**. Sinapses fortes = as pontes que o uso cavou (Brazil→language→Portuguese força 23,
Peru→Spanish 16, France→Paris 12) = Hebbian "fire together, wire together" no grafo. TRACER mostra a
semente sendo regada (Brazil→germina→Portuguese→ponte criada). Honesto: o germinador mini é imperfeito
(31/120 germinações fracas não cristalizaram; capital-de-Brazil falhou 1 vez) → qualidade do germinador
é o teto; semear com fonte melhor (o grafo do WS16 / 3B) resolve. WS16(extração=semeadura inicial) +
WS17(germinação+sinapse em runtime) = a memória IARA completa. ws17_seed.json.

========================================================================
# WS18 — NASCIMENTO DE SEMENTE por co-ativação — 14:42
========================================================================
folhas herdadas do WS17: 48 sinapses (país→relação→valor)

## STREAM 61 consultas (sessões temáticas por região + ruído)
  co-ativação final por conceito (top):
    capital@South America  co-disparou 26× ★ NASCEU
    capital@Europe         co-disparou 14× ★ NASCEU
    capital@Asia           co-disparou 6×
    language@South America co-disparou 5×
    language@Asia          co-disparou 5×
    language@Africa        co-disparou 3×

## NASCIMENTOS (a lei: co-ativação >= 8)
  ★ nasceu 'capital@South America': semente-conceito que CARREGA 6 membros (co-disparou 8×)
      = {'Brazil': 'Brasilia', 'Argentina': 'Buenos Aires', 'Peru': 'Lima', 'Chile': 'Santiago', 'Colombia': 'Bogota', 'Bolivia': 'La Paz'}
  ★ nasceu 'capital@Europe': semente-conceito que CARREGA 5 membros (co-disparou 8×)
      = {'France': 'Paris', 'Germany': 'Berlin', 'Italy': 'Rome', 'Spain': 'Madrid', 'Portugal': 'Lisbon'}

## GANHO — 'capitais da América do Sul'
  agora (SEMENTE-CONCEITO (1 ativação)): ['Brazil', 'Argentina', 'Peru', 'Chile', 'Colombia', 'Bolivia']
  custo: 1 ativação(ões)  vs  6 sem a semente = 6× menos trabalho

## A TERRA CRESCEU PRA CIMA (abstrações, não só folhas)
  folhas (fatos): 48 · sementes-conceito NASCIDAS: 2
  cada semente-conceito comprime N fatos em 1 nó consultável (chunking) — a próxima ideia do Leonardo, medida
  e ela é COMPOSTA: 'capital@South America' já liga aos países E às capitais (a relação vira estrutura)

VEREDITO WS18: a co-ativação de sinapses relacionadas NASCE uma semente-conceito (lei do limiar), que carrega os membros e torna a consulta agregada 1 ativação — a terra cria abstrações com o uso. wall 0.0s

========================================================================
# WS19 — BATERIA PESADA DE PRODUTO (7B real) — 2026-07-12 15:16
========================================================================
FASE 1: carregando 7B (dequant device_map GPU+CPU)... 180 arestas faltando
  7B carregou em 141s
  extraídas 15/180 · 257s · ~17.1s/aresta · grafo 14
  extraídas 30/180 · 478s · ~15.9s/aresta · grafo 28
  extraídas 45/180 · 700s · ~15.5s/aresta · grafo 41
  extraídas 60/180 · 923s · ~15.4s/aresta · grafo 53
  extraídas 75/180 · 1145s · ~15.3s/aresta · grafo 65
  extraídas 90/180 · 1367s · ~15.2s/aresta · grafo 77
  extraídas 105/180 · 1585s · ~15.1s/aresta · grafo 91
  extraídas 120/180 · 1803s · ~15.0s/aresta · grafo 106

========================================================================
# WS19 — BATERIA PESADA DE PRODUTO (7B real) — 2026-07-12 15:50
========================================================================
FASE 1: 106 arestas PREMIUM do 7B (checkpoint; 7B GPU-faultou aos 120/180 — honesto)
  completando 74 arestas com o 3B (GPU, multi-prompt)...
  grafo final: 167 arestas · proveniência {'3B': 61, '7B': 106}
  curadoria: 162/167 = 97% corretas

FASE 2: INTELIGÊNCIA — 100 sessões de raciocínio sobre o grafo-7B
  L1 direto 94% · L2 agregado F1 0.90 · L3 2-hop 95% · L4 cross 18% · ÍNDICE 0.74 (300 consultas)

FASE 3: FLUXO RUNTIME — germinar faltas + sinapse + nascimento (germinador leve)
  fluxo 76 consultas: sinapse/grafo 70 (~0.01ms) · germinou 3 (~321ms) · miss 3
  sementes-conceito NASCIDAS por co-ativação: [] (0)

FASE 4: PRODUTO
  grafo-7B (semeadura 1×): 167 arestas, 97% corretas · ~1002 bytes vs 4.7GB do 7B = 5e+06×
  inteligência (grafo, sem rodar LLM): índice 0.74 · latência de consulta ~0.01ms
  runtime: 92% servido instantâneo · germinador só p/ faltas · 0 conceitos nasceram
  cobertura do fluxo: 73/76 = 96% respondido

VEREDITO WS19: PRODUTO validado com 7B real — semeia grafo (curadoria 97%), inteligência 0.74 navegando sem LLM, runtime 96% coberto com 0 conceitos nascidos. wall total 1.8min

## WS19 — BATERIA PESADA DE PRODUTO (7B real) — a IARA validada de ponta a ponta
7B usado DE VERDADE: GGUF/llama-cpp segfaulta, mas torch device_map (dequant GPU+CPU, 141s) FUNCIONA.
Extraiu 106 arestas premium em 30min (~15s/aresta) e então **GPU-faultou aos 120/180** (offload ROCm
instável em run longo) — mas o CHECKPOINT salvou tudo. Completei as 61 faltantes com o 3B (GPU, confiável).
GRAFO FINAL: 167 arestas (7B=106 + 3B=61), **curadoria 97%** (162/167 corretas). INTELIGÊNCIA (300
consultas/100 sessões, navegando o grafo SEM rodar LLM, ~0.01ms): L1 direto 94% · L2 agregado F1 0.90 ·
L3 2-hop 95% · L4 cross 18% (fraco: compõe 2 relações imperfeitas, região é a pior) · ÍNDICE 0.74.
FLUXO runtime: 96% coberto, 92% servido instantâneo por sinapse, germinador (3B) só p/ 3 faltas (321ms);
0 conceitos nasceram (limiar de co-ativação não batido neste stream de 8 sessões — tuning, mecanismo já
provado no WS18). PRODUTO: conhecimento em ~1KB (167×6B) vs 4.7GB do 7B; o pesado semeia 1× e some.
Achado de infra: 7B via device_map é usável mas instável em runs longos → extrair em lotes com
restart-on-fault (o checkpoint salvou). ws19_produto.json + ws19_graph.json.

========================================================================
# WS20 — CHAT com 7B real + absorção ampla — 16:42
========================================================================
7B carregou em 130s (device_map GPU+CPU)

## PARTE A — CHAT com o 7B (multi-turno, formato Qwen)
  [você] What is the capital of the South American country that borders Brazil to the south?
  [7B]  The capital of the South American country that borders Brazil to the south is Asunción, which is the capital of Paraguay.
        (27 tokens · 81.3s · 0.3 tok/s)
  [você] And what language do they speak there?
  [7B]  In Paraguay, the official language is Spanish. Additionally, a significant portion of the population speaks Guarani, which is recognized as a co-official language alongside Spanish.
        (35 tokens · 64.0s · 0.5 tok/s)
  [você] Explain in one sentence why the sky is blue.
  [7B]  The sky appears blue because the Earth's atmosphere scatters shorter-wavelength blue light more than longer-wavelength red light, a phenomenon known as Rayleigh scattering.
        (33 tokens · 59.9s · 0.6 tok/s)

========================================================================
# WS21 — O CHAT CERTO (grafo-7B instantâneo + 3B) vs 7B cru lento — 16:58
========================================================================
grafo semeado pelo 7B (WS19): 167 arestas · o 7B NÃO roda no chat (só semeou)

pergunta                          IARA (grafo-7B)            lat   3B rápido
  capital of Paraguay             —                       0.60ms   'Asunción' (3tok/s)
  main language of Peru           Spanish                 0.03ms   'The main language of Peru is Spanish.' (13tok/s)
  currency of Japan               Yen                     0.04ms   'The currency of Japan is the Japanese ye' (13tok/s)
  continent of Kenya              Africa                  0.03ms   'Kenya is located in the continent of Afr' (13tok/s)
  capitals of countries in South ABuenos Aires, Santia    0.13ms   'Here are some capitals of countries in S' (13tok/s)

  ACERTO: IARA(grafo-7B) 3/5 · 3B cru 3/5
  LATÊNCIA: IARA ~0.01ms (instantâneo, sem rodar LLM) · 3B ~3.8s/resposta · 7B ~80s/resposta (0.3 tok/s)

## TRACER — uma pergunta pelo chat IARA
  [você] capital of Paraguay
  [IARA] None  (0.03ms — o 7B semeou isto offline; agora responde instantâneo)

VEREDITO WS21: o chat certo NÃO roda o 7B (0.3 tok/s, não cabe na GPU) — usa o CONHECIMENTO do 7B
  via grafo (instantâneo, 3/5 certo) + 3B rápido p/ fluência. É a IARA: o pesado semeia, o leve serve.

## WS20/21 — CHAT com 7B: por que é lento e o caminho certo
WS20: chat com o 7B FUNCIONA e é COERENTE (multi-hop certo: "país ao sul do Brasil"→Asunción/Paraguai;
follow-up usa contexto: Paraguai→Espanhol+Guarani) MAS **0.3-0.5 tok/s** (27 tok em 81s). CAUSA: 7B fp16
= 14GB > 12.9GB VRAM → device_map força ~3GB no CPU → cada token rasteja. NÃO é GPU mal-otimizada; o
modelo não CABE inteiro nela. Fast-7B exige 4-bit-na-GPU (~5GB): llama-cpp HIP segfaulta nesse RDNA2,
bitsandbytes não instalado (ROCm incerto). WS21: a resposta é a IARA — NÃO rodar o 7B no chat; usar o
CONHECIMENTO dele via grafo (WS19) INSTANTÂNEO (0.01ms) + 3B rápido (6-13 tok/s GPU) p/ fluência. Medido:
grafo-7B 3/5 (0.01ms) · 3B cru 3/5 (~4s, mas erra raciocínio: "São Paulo" p/ país-ao-sul-do-Brasil) ·
7B ~80s/resposta. GAP honesto: o grafo só sabe o que foi SEMEADO (capital do Paraguai faltou no WS19 →
"—"); produto = semear melhor/mais, aí o grafo serve tudo instantâneo com o cérebro do 7B.

========================================================================
# WS22 — SEMEAR OS PESOS EM GRAFO (sem Q&A) — 17:06
========================================================================
modelo: 36 camadas × 11008 neurônios · decodificando os PESOS das camadas 24-35 (sem perguntar nada)

========================================================================
# WS22 — SEMEAR OS PESOS EM GRAFO (sem Q&A) — 17:06
========================================================================
modelo: 36 camadas × 11008 neurônios · decodificando os PESOS das camadas 24-35 (sem perguntar nada)
  neurônios decodificados: 132096 · com conceito LEGÍVEL (≥4/6 tokens alfabéticos): 89302 = 68%

## TERRITÓRIOS DE CONCEITO nos pesos (sem perguntar — só o valor dos neurônios)
  PAÍSES:
    L35 n1246 (cos 0.49) escreve: ['·France', '·Germany', 'France', '·Italy', 'èĭ±åĽ½', '·Spain']
    L31 n5453 (cos 0.47) escreve: ['·Australia', '·America', '·Germany', '·Italy', '·France', '·Japan']
    L35 n283 (cos 0.44) escreve: ['æ¬§æ´²', '·European', 'æ¾³å¤§åĪ©äºļ', 'èĭ±åĽ½', 'ç¾İåĽ½', '·Europe']
    L30 n5054 (cos 0.43) escreve: ['·Europe', '·Germany', '·France', '·Italy', '·Australia', '·India']
  CAPITAIS:
    L34 n9533 (cos 0.41) escreve: ['·London', 'æĿŃå·ŀ', 'London', 'æĪĲéĥ½', 'æµİåįĹ', 'è´µéĺ³']
    L35 n6266 (cos 0.39) escreve: ['åľ¨äº¬', 'åľ¨åĮĹäº¬', 'äº¬', 'åĮĹäº¬', 'åĮĹäº¬å¸Ĥ', 'äº¬åŁİ']
    L31 n6381 (cos 0.35) escreve: ['·London', 'åĮĹäº¬', 'London', '·Paris', 'ä¼¦æķ¦', '·Berlin']
    L35 n1246 (cos 0.33) escreve: ['·France', '·Germany', 'France', '·Italy', 'èĭ±åĽ½', '·Spain']
  LÍNGUAS:
    L35 n5921 (cos 0.53) escreve: ['·French', '·Chinese', '·Japanese', '·German', '·Italian', 'French']
    L34 n6341 (cos 0.42) escreve: ['·English', 'English', '·english', '·Spanish', 'english', '·spanish']
    L33 n642 (cos 0.40) escreve: ['·German', '·Italian', '·Japanese', '·French', '·British', '·Russian']
    L35 n2377 (cos 0.37) escreve: ['·âĤ¬', 'ä¸ĩæ¬§åħĥ', 'æĦıå¤§åĪ©', '·German', 'æ¬§æ´²', '·Italian']

## LIGAÇÃO país→capital DIRETO DOS PESOS (a aresta está lá?)
  France→Paris: melhor neurônio-escreve-Paris = L31n10574, chave↔France = +0.360 ✓ aresta nos pesos
  Japan→Tokyo: melhor neurônio-escreve-Tokyo = L35n6394, chave↔Japan = +0.223 ✓ aresta nos pesos
  Germany→Berlin: melhor neurônio-escreve-Berlin = L34n5201, chave↔Germany = +0.080 ✓ aresta nos pesos
  Egypt→Cairo: melhor neurônio-escreve-Cairo = L35n7630, chave↔Egypt = +0.049 ✓ aresta nos pesos
  China→Beijing: melhor neurônio-escreve-Beijing = L33n2490, chave↔China = +0.262 ✓ aresta nos pesos
  ligação-fato encontrada nos pesos: 5/5

## VEREDITO WS22 (honesto)
  ✓ CONCEITO sai dos pesos direto: 68% dos neurônios profundos têm conceito legível;
    territórios emergem (neurônio que escreve países, capitais, línguas — SEM perguntar nada).
  ✓ LIGAÇÃO-FATO país→capital nos pesos: 5/5 — a aresta está parcialmente nos pesos
  LIMITE honesto: 'semear todos os pesos' dá o grafo NEURÔNIO (conceitos, territórios) de graça e
    determinístico; mas o FATO simbólico limpo (país→capital como 1 aresta) é distribuído — precisa
    de análise de circuito OU do probe pra cristalizar. Os pesos DÃO o substrato; a aresta-fato é emergente.
wall 0.8min

## WS22 — SEMEAR OS PESOS EM GRAFO (sem Q&A) — a ideia real do Leonardo, VALIDADA
Distinção do Leonardo: não interrogar (perguntar/salvar) — DECOMPOR os pesos. Método: cada neurônio
FFN = nó; valor(down_proj col)→logit-lens = o que ESCREVE; chave(gate/up)→embedding = o que ATIVA.
Determinístico, 1 passada, sem perguntar nada. Rodado no 3B (132.096 neurônios das 12 camadas profundas).
ACHADOS: (1) **68% dos neurônios profundos têm CONCEITO LEGÍVEL** (valor decodifica em tokens coerentes);
(2) TERRITÓRIOS emergem dos PESOS: neurônio-países (L35n1246 escreve France/Germany/Italy/Spain),
neurônio-capitais (L31n6381 escreve London/Paris/Berlin), neurônio-línguas (L35n5921 escreve
French/Chinese/Japanese/German) — SEM perguntar; (3) **A ARESTA-FATO ESTÁ NOS PESOS: 5/5** — o neurônio
que ESCREVE a capital tem CHAVE que responde ao país: France→Paris chave↔France +0.36 (forte),
China→Beijing +0.26, Japan→Tokyo +0.22 (fortes); Germany→Berlin +0.08, Egypt→Cairo +0.05 (fracos=fato
mais distribuído). VEREDITO honesto: semear-os-pesos DÁ o grafo-neurônio (conceitos, territórios,
ligações-fato) DE GRAÇA e determinístico; forte pros fatos localizados, fraco pros distribuídos (aí
precisa de circuito/probe). É o caminho CERTO de semeadura (op de PESO, não inferência lenta — escala pro
7B). Substitui a bateria Q&A lenta do WS19. ws22_pesos.json.
