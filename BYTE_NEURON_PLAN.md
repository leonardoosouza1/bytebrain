# Byte-Neuron — Plano de Ponta a Ponta

**Tese (Leonardo):** o átomo da rede é um **neurônio-byte** — uma unidade cujo *estado* cabe em poucos bits (1/2/4/8). Bytes vivem num **espectro binário** (Walsh-Hadamard) onde componentes carregam significado (bit5 = caso, validado KL≈0.003). Neurônios se **juntam** em níveis superiores (junção de bytes = conhecimento), formando uma **pirâmide multi-resolução**: base = forma (bytes), topo = conhecimento (conceitos). Objetivo: muitos neurônios, leves, esparsos — como o cérebro.

**Disciplina:** cada fase tem hipótese, método, métrica e **porta de decisão**. Seguimos o número, não a empolgação. Onde já há resultado contrário, está marcado 🧱.

**Ferramenta-microscópio:** o iara-engine (llm-lab) já mede "disparo" por camada (`hot_neurons`, `hot_neuron_values`, `mlp_activation_rate`, `spikes`). Unir ByteBrain ao engine = poder VER e MEDIR cada neurônio por byte.

---

## Fase 0 — A ponte (ByteBrain ↔ iara-engine): construir o microscópio

| Passo | O quê | Estado |
|---|---|---|
| 0.1 | `export_to_safetensors.py`: `.pt → safetensors + config.json` (arch ByteGPT) | ✅ escrito |
| 0.2 | Arch `ByteGPT` no engine: LayerNorm + pos aprendida + GELU + MHA, vocab 256 (sem RoPE/GQA). Reaproveita `layer_norm`/`embed_lookup`/`gemm`/`softmax`; falta só GELU plano | ⏳ |
| 0.3 | Tokenizer modo-byte (`texto → bytes 0-255`, bypassa BPE) | ⏳ |
| 0.4 | **Parity gate**: engine reproduz a bpb do ByteBrain (PyTorch) no mesmo input, dentro de tolerância | ⏳ |
| 0.5 | Telemetria estendida: dump do vetor de ativação por byte (não só top-10) p/ análise offline | ⏳ |

**Porta 0:** se a engine NÃO reproduzir a bpb do PyTorch → o microscópio mente; consertar antes de qualquer conclusão. *(Sem isso, todos os testes abaixo são lixo.)*

---

## Fase 1 — O neurônio já é uma "chave"? (esparsidade + profundidade de bits)

**Hipótese:** as ativações já são quase binárias/esparsas; 2-4 bits por neurônio quase não pioram a bpb.

- **1.1** `mlp_activation_rate` por byte: que % dos neurônios dispara? (aposta: >90% ficam off).
- **1.2** **Sweep de bits** (responde "2 ou 4 bits?"): quantiza ATIVAÇÕES em 1/2/3/4/8 bits → bpb em cada. Gráfico bpb × bits/neurônio.
- **1.3** Mesmo sweep nos PESOS (liga com DNA-Q4).
- **1.4** Histograma das magnitudes: distribuição é bimodal (chave) ou gaussiana (contínuo)?

**Porta 1:** se 2-4 bits seguram a bpb (Δbpb < ~5%) → "**neurônio-byte leve**" validado. Se desaba abaixo de 8 bits → a tese cai aqui (e a gente reporta honesto).

---

## Fase 2 — O espectro do byte (semântica de Walsh-Hadamard)

**Hipótese:** a estrutura de bits do byte carrega a maior parte da informação do embedding; a tabela 256×dim é parcialmente redundante.

- **2.1** Decomposição de Walsh do embedding aprendido (256×dim) → quais componentes são grandes?
- **2.2** Mapear cada componente forte → feature (bit5=caso ✅; bit4=vogal/grupo? bit7=UTF-8? pares de bits?).
- **2.3** Probe linear sobre o vetor de 8 bits prevê caso/vogal/etc.? (estende o bit5 KL=0.003 pro espectro inteiro).
- **2.4** **Arch "bit como entrada"**: troca embedding por 8 bits (±1) crus OU o Hadamard deles. Iguala/bate o embedding com custo mínimo?

**Porta 2:** se a estrutura de bits/Walsh explica a maior parte do embedding → bytes SÃO um código de frequência; tabela de embedding é redundante (modelo mais leve e interpretável).

---

## Fase 3 — Neurônio = junção de bytes + a pirâmide (multi-resolução) 🧱

**Hipótese:** uma representação multi-resolução (base=bytes/forma → topo=conceitos/conhecimento) bate a rede flat ao mesmo nº de params.

🧱 **Prior honesto:** já testamos hierarquia (estilo MEGABYTE) na escala doméstica e **PERDEU** (val 3.1 vs 2.2 flat). Esta fase **re-testa** com a lente NOVA (neurônios low-bit + espectro), não repete o mesmo experimento. Se perder de novo, a gente aceita.

- **3.1** Definir níveis: L0 = bytes (forma), L1 = pares/n-gramas (junções), … Lk = conceitos. "Neurônio = junção de bytes" = pooling de bytes num nível superior.
- **3.2** **Pirâmide de frequência**: decompor a sequência em bandas multi-escala (reaproveitar a intuição da Resonance Forest / pirâmide latente multi-freq que bateu JPEG).
- **3.3** Pirâmide vs flat, mesmos params: a lente nova vira o resultado do MEGABYTE?
- **3.4** "Conhecimento no topo, forma na base": níveis altos preveem melhor a próxima PALAVRA (semântica) e os baixos o próximo BYTE (forma)?

**Porta 3:** hierarquia AJUDA agora? Se sim, segue pra Fase 4. Se não → reporta honesto, fica com o flat + Fases 1-2.

---

## Fase 4 — Síntese: o modelo byte-neurônio IARA

Se 1-3 derem verde: treinar um modelo que é (a) neurônios-byte 2-4 bit, (b) entrada bit/Walsh-estruturada, (c) pirâmide multi-resolução. Comparar **bpb × tamanho × velocidade** vs o ByteBrain float. **Aqui o "revolucionário" se prova ou cai — com gráfico.**

---

## Regra anti-muro (pedido do Leonardo)
Em cada passo, se travar: trocar o ângulo, não parar. Ex.: parity falhou → comparar camada-a-camada; sweep de bits ruim → testar por-camada (talvez só algumas camadas precisem de 8 bits); hierarquia perde → testar só o pooling de entrada, não a arquitetura inteira. **Reportar todo número — acerto E fracasso** (vai pro [FINDINGS.md] e overnight_journal).
