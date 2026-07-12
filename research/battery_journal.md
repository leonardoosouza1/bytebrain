
## Lote 1 — compressão/modularidade (Qwen3-4B, 150s)
- núcleo 280854 · spec 57177 · morto 12177 · chat baseline 3.202
- T1 só-núcleo (80.2% ativos): chat Δ+5.635
- T2 podar mortos+símbolo: chat Δ+0.024
- T3 camadas mais podáveis: [(21, -0.036), (15, -0.023), (17, -0.021), (19, -0.02), (8, -0.01)]
- T4 compressão: {'100%': 3.202, '75%': 12.755, '50%': 14.174, '30%': 14.628, '15%': 15.852}
- T7 condicional: {'chat_pt': 4.008, 'código': 3.461, 'matemática': 2.651, 'inglês': 6.798, 'outro idioma': 5.103}
- T8 fundir cód+mat: {'cod_loss': 3.121, 'mat_loss': 2.907, 'pt_loss': 10.673, 'ativos_pct': 84.0}
- T9 amplificar código: {'x0.0': 6.701, 'x0.5': 3.856, 'x1.0': 2.921, 'x2.0': 3.128, 'x3.0': 4.58}

## Lote 2 — compressão evoluída (chat, Qwen3-4B, 225s)
- baseline chat 3.193
- poda uniforme por importância: {'75%': 3.251, '50%': 3.367, '30%': 3.914}
- EVOLUÍDA: chat 3.121 com 61.7% dos neurônios (vs uniforme colapsando)

## Lote 3 — multimodal (SmolVLM-256M, 25s)
- interpretou 4 imagens (ex: In this image there is a table with two rows and two columns)
- território VISUAL no LLM: 7934 neurônios (17.2%) vs linguagem 1967

## Eletroestimulação Da Vinci (Qwen3-4B, 90s)
- eletrodo chat_pt: evoca ['names', '缝', 'those', 'doing', 'wich'] (KL 11.906)
- eletrodo código: evoca ['є', '和个人', 'opsis', 'sing', 'matplotlib'] (KL 1.233)
- eletrodo matemática: evoca ['-$', '$x', '_$', '$I', '.$'] (KL 0.991)
- eletrodo inglês: evoca ['/false', '这一点', '这点', '给她', 'falsehood'] (KL 0.675)
- caos: damp médios {'L0_s2.0': 1.012, 'L9_s2.0': 0.853, 'L18_s2.0': 0.703, 'L27_s2.0': 0.788}

## T16 — modelo ESCULPIDO fisicamente (96s)
- 4.02B → 2.99B (-26%) | chat 3.193 → 3.119 | 9.0 → 11.4 tok/s | VRAM 8.1 → 6.0GB

## Lote 4 — T17 K emergente + T19 edição (158s)
- K emergente por limiar: {'0.9': 20, '0.95': 35, '0.98': 53}
- módulos τ=0.95: [(271045, 'wiki'), (3884, 'matemática'), (3101, 'outro'), (2675, 'código'), (2643, 'código'), (2627, 'wiki'), (2484, 'inglês'), (2388, 'código')]
- edição: França 'onde está o Pal'→'' | Japão 'Tokyo, mas por'→'Japão J' (3000 neurônios)

## Lote 5 — T12/T15 bytes universais + T10 cross-model (28s)
- ByteBrain 40M territórios de modalidade: {'aleatório': 0, 'código': 563, 'imagem': 5, 'texto': 961, 'áudio': 3} | roteador 63%
- Qwen1.5B territórios: {'português': 9053, 'código': 3635, 'matemática': 4194, 'inglês': 2453} | roteador 96% (4B: 90%)

## ARENA — ranking local (79s)
- qwen3-4b (original): composto 0.85 | fatos 0.7 arit 1.0 mmlu 0.7 cód 1.0 ppl 14.95
- qwen2.5-1.5b: composto 0.825 | fatos 0.8 arit 1.0 mmlu 0.5 cód 1.0 ppl 11.98
- smollm2-1.7b: composto 0.537 | fatos 0.2 arit 0.75 mmlu 0.2 cód 1.0 ppl 13.04
- iara-3b (esculpido): composto 0.256 | fatos 0.2 arit 0.12 mmlu 0.2 cód 0.5 ppl 47.55
- bytebrain-40m: composto 0.212 | fatos 0.6 arit 0.0 mmlu 0.0 cód 0.25 ppl None

## Programa de eletricidade (117s)
- dose: {"matemática": [{"beta": 0.5, "kl": 0.048, "evoca": ["|$", "$l", "$", "$x"]}, {"beta": 1.0, "kl": 0.124, "evoca": ["|$", ".$", "*$", "-$"]}, {"beta": 2.0, "kl": 0.463, "evoca": ["-$", ".$", "*$", "_$"]}, {"beta": 4.0, "kl": 2.563, "evoca": ["sum", "=sum", "-sum", "-$"]}, {"beta": 8.0, "kl": 11.347, 
- posição: [{'faixa': 'cedo 0-11', 'kl': 0.061, 'evoca': ['的基础上', '常见的', 'ないこと', '的是']}, {'faixa': 'meio 12-23', 'kl': 0.085, 'evoca': ['OOK', '+</', '另一位', '我对']}, {'faixa': 'fundo 24-35', 'kl': 0.428, 'evoca': ['.$', '$x', '$s', '*$']}]
- steer base: ' quente, e a gente pode aproveitar para sair e passear. Mas,'
- ruído global: [{'sigma': 0.02, 'kl': 0.0007}, {'sigma': 0.05, 'kl': 0.0109}, {'sigma': 0.1, 'kl': 0.0092}, {'sigma': 0.2, 'kl': 0.052}, {'sigma': 0.4, 'kl': 0.0743}]
- roteador barato: {'4_camadas': '24/24', '8_camadas': '24/24', '12_camadas': '22/24', '36_camadas': '24/24'}

## Carve V2 (dieta ampla) — {'fatos': 0.5, 'aritmética': 1.0, 'mini-MMLU': 0.8, 'código': 1.0, 'wiki_ppl': 17.57, 'composto': 0.825, 'params_B': 3.28}

## Evolução guiada pela ARENA (1245s)
- melhor: composto 0.900 com keep 73% ({'fatos': 0.7, 'arit': 1.0, 'mmlu': 0.9, 'cod': 1.0})
- hist: [{'gen': 0, 'fit': 0.712, 'composto': 0.9, 'keep': 0.754}, {'gen': 1, 'fit': 0.712, 'composto': 0.9, 'keep': 0.754}, {'gen': 2, 'fit': 0.712, 'composto': 0.9, 'keep': 0.754}, {'gen': 3, 'fit': 0.714, 'composto': 0.9, 'keep': 0.746}, {'gen': 4, 'fit': 0.717, 'composto': 0.9, 'keep': 0.734}, {'gen': 5, 'fit': 0.717, 'composto': 0.9, 'keep': 0.731}, {'gen': 6, 'fit': 0.717, 'composto': 0.9, 'keep': 0.731}, {'gen': 7, 'fit': 0.717, 'composto': 0.9, 'keep': 0.731}]

## IARA-1 v0.1 salvo em /home/leonardo/projects/LLM/llm-lab/models/iara-3b-v01 — {'fatos': 0.7, 'aritmética': 1.0, 'mini-MMLU': 0.9, 'código': 1.0, 'composto': 0.9, 'params_B': 3.3, 'demo': ' \nokay, the user is asking why the sky is blue in one sentence. i need to explain the science behind the blue color of t'}

## SFT chat no ByteBrain (946s, 400 passos CPU)
- USER: Qual é a capital da Itália? | ANTES ' Resolva passo a passo: 23 * 47. O resultado é:\nA)' | DEPOIS ' A capital da Itália é Roma.\n\n'
- USER: Oi, tudo bem? | ANTES ' MPLA, Tecnologia\nEm programação, banco de dados é' | DEPOIS ' Oi! Tudo bem sim, e com você?\n\n'
- USER: O que é o DNA? | ANTES ' DNA é a molécula que armazena a informação genéti' | DEPOIS ' O DNA é a molécula que guarda as informações gené'
- USER: Quem é você? | ANTES ' Retorno para adicionar um item à pilha\n       con' | DEPOIS ' Eu sou o IARA, um assistente pequeno que roda no '
- USER: Qual é o maior planeta? | ANTES ' FIFA COLAPAPA é um conjunto de regras e protocolo' | DEPOIS ' O maior planeta do sistema solar é Júpiter.\n\n'

## Anatomia completa — phi4mini (76s)
- arena {'fatos': 0.5, 'aritmética': 1.0, 'mini-MMLU': 0.8, 'código': 1.0, 'composto': 0.825}
- territórios {'português': 8336, 'código': 2327, 'matemática': 3710, 'inglês': 2220} | roteador {'full': '24/24', '4layers': '24/24'}
- mortos 0.0% spec 6.3% | ondas {'L0': {'damp': 2.33, 'kl': 0.485}, 'L16': {'damp': 0.77, 'kl': 0.02}, 'L26': {'damp': 3.55, 'kl': 0.02}}
- eletrodo mat: ['188', '182', '190', '185', '186', '170']

## Anatomia de imagem (tiny-sd, 92s)
- VAE territórios: {'alta': 519, 'baixa': 0, 'borda': 19, 'foto': 6, 'média': 152, 'plana': 17}
- UNet t×freq: {800: {'canais_pró_alta': 8255, 'canais_pró_baixa': 10735, 'razão_média': 1.15}, 400: {'canais_pró_alta': 1598, 'canais_pró_baixa': 1226, 'razão_média': 1.045}, 50: {'canais_pró_alta': 16, 'canais_pró_baixa': 10, 'razão_média': 1.007}}
- eletrodo visual: 16 canais alta, imgs em research/img_stim/
