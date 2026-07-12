# Byte-Neuron — Resultados experimentais

Validação empírica da tese do neurônio-byte (ver [`../BYTE_NEURON_PLAN.md`](../BYTE_NEURON_PLAN.md)).
Tudo roda em PyTorch CPU no host (sem GPU). Scripts: `research/*.py`. Honesto: acertos E fracassos.

## Fase 2.1 — Espectro de Walsh do embedding (`analyze_byte_spectrum.py`)
Os bytes são organizados pela estrutura de bits? Rank das componentes de bit único (de 255):

| Modelo | estado | bit5 (caso) | bit6 (região) | bit7 (UTF8) |
|---|---|---|---|---|
| 8M | convergido | 8 | 20 | 2 |
| 40M | convergido (val 1.288) | **6** | **4** | **1** |
| 86M | cru (step 500, val 3.43) | 235 | 104 | 149 |

→ **Modelos treinados põem os bits semânticos no TOPO do espectro.** O "branco" do 86M era sub-treino (≈ init aleatório), não capacidade — hipótese de capacidade **corrigida pelos dados**. (Single-bits somam ~3.2% da energia total ≈ uniforme; o que muda é o RANK/proeminência.)

## Fase 2.3 — Probe de caso (`probe_case.py`)
| Modelo | cos(eixo-caso, bit5) | acc só-bit5 |
|---|---|---|
| 40M | 0.53 | 100% |
| 8M | 0.53 | 96% |
| 86M (cru) | 0.41 | 96% |

→ Caso vive no eixo espectral bit5 (separável a 96-100%); direção de caso do modelo ~metade alinhada (cos 0.53) com o bit5 puro. Caso aparece **cedo** (96% já no modelo cru) — padrão fácil; a proeminência espectral é que cresce com o treino. (Ignorado: acc "eixo-caso" deu 100% em todos = overfit 768-dim/52-pontos.)

## Fase 1.2 — Quantos bits o neurônio precisa? (`bitdepth_sweep.py`, 40M)
Quantiza o estado dos neurônios (residual stream) em k bits, Δ vs full (bpb 1.118):

| bits | per-tensor | per-canal |
|---|---|---|
| 8 | +2.3% | **+0.0%** |
| **4** | +355% 💥 | **+2.9%** ✅ |
| 3 | +395% | +30% |
| 2 | +382% | +224% 💥 |
| 1 | +383% | +264% 💥 |

→ **Neurônio precisa de ~4 bits (nibble = 16 níveis), e isso é quase de graça.** 1-2 bit colapsa ⇒ neurônio **não é liga/desliga**, carrega valor graduado (≈ frequência de disparo quantizada). O colapso per-tensor em 4 bit era **outlier de ativação** (problema conhecido), resolvido per-canal — o muro era o método, não a ideia.

## Fase 4 — treinar o neurônio JÁ em nibble (quant-aware A/B) ✅
Modelo nasce com ativações em 4 bits (per-canal, STE). A/B controlado (mesma init/seed/dados, 8M, 8000 steps, corpus pt_big). `train.py --quant-bits N --seed 0`.

| Run | neurônio | best val_bpb | gera PT? |
|---|---|---|---|
| baseline | float | **1.802** | sim |
| nibble | **4-bit nativo** | **1.936** (+7.4%) | **sim** (rodando em 4-bit) |

Amostras (ambos prompt "O Brasil é um país", 8M @ 7k steps):
- float: *"…comercial em dez persas faixas e primeiros externos, presentes por clubes de grupos…"*
- 4-bit: *"…na Campanha do Terceiro Norte… ele foi apenas o tilhar foi realizado pela Fazer de Paulo…"*

→ **Um modelo que NASCE com neurônio-nibble gera PT coerente rodando nativamente em 4 bits**, a ~7% de bpb do float nesse orçamento curto.

**Sweep quant-aware completo (15k steps, mesmo config/seed):**

| neurônio | best val_bpb | Δ vs float |
|---|---|---|
| float | **1.652** | — |
| **4-bit** | **1.795** | **+8.7%** |
| 3-bit | 2.265 | +37% |
| 2-bit | (pulado — colapsa) | — |

→ **4 bits é o piso prático** (Leonardo's call, confirmado pelo número). O gap do 4-bit **não fechou** com treino longo (8.7% vs 7.4% no run curto — até abriu um pouco): o nibble tem **custo real**, não é "de graça". ≤3 bits não vinga. Veredito: neurônio = **nibble (4 bits)** é a fronteira leve viável; abaixo disso, colapsa.

## Fase 3 — grafo esparso top-16 (a "sinapse") — SINAL POSITIVO preliminar
top-16 sparse attention (cada neurônio só liga aos 16 mais relevantes; O(N·16), mais leve) vs dense, mesma seed/config, val_bpb por step:

| step | top-16 | dense | Δ |
|---|---|---|---|
| 1000 | 2.668 | 2.757 | **−0.089** |
| 2000 | 2.150 | 2.177 | −0.027 |
| 3000 | 1.983 | 1.991 | −0.008 |
| 4000 | 1.888 | 1.891 | −0.003 |

→ top-16 **≤ dense em TODO step** — **e é mais leve** (O(N·16) vs O(N²)). Cedo treina mais rápido (grafo esparso = regularizador/viés indutivo), gap estreita pra ~empate ~4k. Honesto: pode virar empate por ruído ao fim; 1 seed, budget curto, L=256 (esparsidade leve). Mas a ideia "16 conexões = sinapse" **passou no 1º teste** — não machuca a qualidade, treina ~igual/melhor, mais barato.

**Final @5k:** top-16 = **1.841** vs dense **1.838** → **empate** (vantagem cedo lavou). Win = mesma qualidade + mais leve (O(N·16)) + treino mais rápido cedo.

**mtp-4 (folga da representação / multi-byte): NÃO vingou** — 1.852 @5k (pior que dense 1.838). Prever t+2..t+5 custou a tarefa principal em 8M/5k (MTP costuma precisar de escala maior; peso 0.3 talvez alto). Descartado por ora.

**k-sweep + combo (5k, seed 0):**

| config | bpb@5k | Δ | peso |
|---|---|---|---|
| dense | 1.838 | — | densa + float |
| **top-8** | **1.838** | 0% | O(N·8) + float |
| top-16 | 1.841 | +0.2% | O(N·16) + float |
| top-32 | 1.841 | +0.2% | O(N·32) + float |
| mtp-4 | 1.852 | +0.8% | (neg) |
| **top-16+4bit** | **1.965** | +6.9% | O(N·16) + nibble |

→ **Esparsidade é de graça em L=256** — até top-8 empata com dense (k 8/16/32 não importa: língua redundante, 8 conexões pegam o sinal). Ganho real de compute só em **contexto longo** (L≥1024, onde O(N²) explode). **Combo grafo+nibble compõe limpo** (+6.9% = custo do nibble; grafo é free) = modelo mais leve nos 2 eixos. Geração GPU: top-8 e combo **escrevem PT gramatical** (acentos ok). mtp não vingou.

**Conclusão do arco:** o "byte-LM leve" tem suporte empírico — grafo esparso (free) + neurônio nibble (~+7%) escreve PT. Próximo p/ provar o valor do grafo: **escalar o contexto (L=1024+)**, onde a esparsidade vira velocidade real.

## Veredito (calibrado, honesto)
- ✅ "1 byte = 1 neurônio" (8 bit) → de graça; e **nibble (4 bit) também** → mais leve que o esperado.
- ❌ "liga/desliga (1 bit)" p/ ativações → colapsa; a versão **multi-bit (nibble)** é a que vale.
- ✅ "letra = frequência/espectro" → confirmado nos modelos treinados (bits semânticos = topo do espectro).
- ⏳ "revolucionário" só se decide na **Fase 4** (treinar um modelo byte-neurônio nibble + comparar bpb×tamanho×velocidade). Estes resultados são preliminares (n=3 modelos, 1 corpus PT, quant pós-treino — quant-aware training provavelmente baixa mais).
