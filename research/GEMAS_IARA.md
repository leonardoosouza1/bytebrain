# GEMAS IARA — mapa único do que JÁ VENCEU → onde cada uma entra

> Regra: parar de reinventar. Só juntar o que foi validado e conectar na carcaça/IARA. Números honestos.

| # | Gema (validada) | O que é / o número | Onde entra |
|---|---|---|---|
| 1 | **Água÷grau (recall)** | PPR/difusão ÷ grau: gota entra, flui até os nós de ambos os termos. **Multi-hop 100% vs lookup 3%**. Falha=hub → conserta ÷grau. "graduada acha, spike despeja". | **Recuperação da FLORESTA** (hoje é travessia simples). Vira o `_gather` da IARA e o "sentir o resíduo" do átomo. |
| 2 | **amp em forma fechada** | `amp = ⟨r,ψ⟩/‖ψ‖²` — o ser BEBE o coeficiente ótimo; sobrevivência só esculpe geometria. | **Solver-água (imagem/áudio/vídeo).** Sem isso EXPLODE (foi meu bug). |
| 3 | **Relaxação Gauss-Seidel** | cada átomo refita o resíduo DEPOIS dos outros (não todos contra o mesmo) = harmonização. | **Solver-água** — evita duplo-cômputo/explosão. |
| 4 | **Lei do Metabolismo (AIC)** | `met_c = σ²(1+2·dof)`. Dial taxa-distorção: **a população de equilíbrio EMERGE** (met 0.005→1.0 = 3000→129 seres). | **Sobrevivência** — quem explica < met_c morre. Controla população sem hiperparâmetro. |
| 5 | **Lei do Garimpo (RIC)** | `pen = 2σ²·ln(M)`. Penalidade de seleção. Generaliza: Deflated Sharpe, seleção de variável, **poda de rede (bate magnitude 6/6)**. | **Seleção/poda** — na floresta (podar sinapse fraca) e na população. |
| 6 | **Reflexo raiz-digital** | atratores de `n→s(n²)` = {1},{9},{13↔16} = 4 classes de raiz digital (quadrados mod 9). Kaprekar 6174. **0 parâmetro.** | **Reflexo (System 1)** — órgão calculável barato. |
| 7 | **Memória = grafo relacional** | união por SET-DE-ARESTAS = **0 interferência** (fundir peso destrói). 1KB vs 4.7GB. 2-hop 95%. | **A FLORESTA é a memória** (não peso). Modelos = fontes de aresta. |
| 8 | **Neurônios calculáveis** | Hopfield = a própria água-recall; ELM XOR 100% sem backprop; **readout = verificador** (sei/blefo 100% LOO). | **Verificador + reflexo** — órgãos exatos e leves. |
| 9 | **Seletor não gerador** | tese-mãe: ganha em rotear/escolher/emergir; perde em gerar/comprimir. No linear, guloso≥sobrevivência (não p-hackear). Lever = **átomo não-linear + prior importado→geração**. | **Bússola** — não medir no eixo errado. |

## Ordem de implementação (juntar o que venceu)
1. **Solver-água CERTO** (gemas 2+3+4): amp forma-fechada + Gauss-Seidel + lei do metabolismo → a imagem EMERGE da sobrevivência (conserta a explosão). Valida: PSNR sobe, população emerge, pirâmide.
2. **Água÷grau na floresta** (gema 1): recall por difusão÷grau → multi-hop real. Valida: multi-hop >> lookup.
3. **Reflexo raiz-digital** (gema 6) + **poda por garimpo** (gema 5) na carcaça. Valida: 0-param exato + poda que preserva.
4. Amarrar: a carcaça embodied usa água÷grau pra sentir, forma-fechada pra beber, metabolismo pra viver, verificador pra saber.

> A `iara-agua` (Rust) já É o solver-água F0 (31 dB, áudio/vídeo). Aqui valido as gemas juntas em Python rápido; depois pluga na carcaça/Universe.

## RESULTADOS — junção validada (2026-07-21)

| passo | gemas | script | resultado | veredito |
|---|---|---|---|---|
| 1 | 2+3+4 | `iara_agua_carcaca.py` | PSNR **8.3→24.1 dB** monotônico, 1.9s; população **6→93** (lei do metabolismo, não hiperparâmetro); escalas σ{3,6,10}=pirâmide; cada ser tila região+freq distintas | ✅ a imagem EMERGE da sobrevivência; conserta a explosão da v1 |
| 2 | 1 | `water_recall.py` | multi-hop: **ppr÷grau 100% vs ppr cru 0%**; 1-hop: lookup trivial já 100% | ✅ ÷grau é a alavanca do multi-salto (conserta hub) |
| 3a | 5 | `poda_metabolica.json` | met(garimpo)=OMP-oráculo, bate magnitude e Wanda na reconstrução por bloco (0.176/0.711/0.752) | ✅ na reconstrução; ⚠️ em perplexidade global Wanda ganha (honesto) |
| 3b | 6 | `iara_reflexo_raiz.py` | n→dr(n²): **3 atratores {1},{4,7},{9}=quadrados mod 9, 100%/1000, 0-param**; Kaprekar **8991/8991→6174 ≤7 passos** | ✅ reflexo exato System-1, sem peso |

**O que a junção mostra:** as gemas não são truques isolados — são o MESMO princípio em domínios diferentes.
A forma-fechada (beber o coeficiente ótimo) + Gauss-Seidel (harmonizar com os outros) + metabolismo (morrer se não
explica) faz a imagem emergir da sobrevivência dos átomos; a MESMA água que difunde ÷grau no grafo faz o recall
multi-salto; a MESMA lei (AIC/RIC) que controla a população dos átomos poda a rede. Reflexo raiz-digital = o System-1
calculável embaixo de tudo. Falta amarrar na carcaça embodied (passo 4): sentir com água÷grau, beber forma-fechada,
viver por metabolismo, saber por verificador.
