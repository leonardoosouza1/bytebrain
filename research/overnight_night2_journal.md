# BATERIA NOTURNA 2 — início Fri Jul  3 04:13:31 PM -03 2026


========== RUN bytebrain/research/marco112_fact_forest.py  (16:13:31) ==========
Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]Loading weights:   0%|          | 1/338 [00:01<07:23,  1.32s/it]Loading weights:  40%|███▉      | 135/338 [00:01<00:01, 113.73it/s]Loading weights:  49%|████▉     | 166/338 [00:02<00:01, 98.10it/s] Loading weights:  58%|█████▊    | 195/338 [00:02<00:01, 115.64it/s]Loading weights:  68%|██████▊   | 230/338 [00:02<00:00, 144.31it/s]Loading weights:  76%|███████▌  | 257/338 [00:02<00:00, 158.41it/s]Loading weights:  87%|████████▋ | 294/338 [00:02<00:00, 194.67it/s]Loading weights: 100%|██████████| 338/338 [00:02<00:00, 132.27it/s]
/home/leonardo/projects/LLM/.venv-rocm/lib/python3.12/site-packages/torch/nn/modules/module.py:1357: UserWarning: expandable_segments not supported on this platform (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:36.)
  return t.to(
[    19s] M112v2 START | Math-1.5B congelado hidden 1536
[    20s] pool 1-token: 80 | banco 600 fatos ARBITRÁRIOS
[    20s] === A) FLORESTA 10×(K=1,20 fatos) vs SEMENTE K=10 — mesmos bytes, 200 fatos ===
[   151s]   A.floresta semente  0: fp16 20 | i4pt 3 | i4g 12 | i8g 20 (/20)
[   282s]   A.floresta semente  1: fp16 20 | i4pt 0 | i4g 0 | i8g 20 (/20)
[   412s]   A.floresta semente  2: fp16 20 | i4pt 10 | i4g 6 | i8g 20 (/20)
[   542s]   A.floresta semente  3: fp16 20 | i4pt 2 | i4g 10 | i8g 20 (/20)
[   673s]   A.floresta semente  4: fp16 20 | i4pt 0 | i4g 5 | i8g 20 (/20)
[   860s]   A.floresta semente  5: fp16 20 | i4pt 0 | i4g 5 | i8g 20 (/20)
[  1046s]   A.floresta semente  6: fp16 20 | i4pt 3 | i4g 7 | i8g 20 (/20)
[  1233s]   A.floresta semente  7: fp16 20 | i4pt 2 | i4g 9 | i8g 20 (/20)
[  1419s]   A.floresta semente  8: fp16 20 | i4pt 3 | i4g 1 | i8g 20 (/20)
[  1606s]   A.floresta semente  9: fp16 20 | i4pt 2 | i4g 9 | i8g 20 (/20)
[  3357s]   A.FLORESTA(10×K1) fp16 200/200 i4pt 25 i4g 64 i8g 200
[  3357s]   A.SEMENTE K=10   fp16 200/200 i4pt 115 i4g 159 i8g 200
[  3357s]   A.VEREDITO i4-grupo (mesmos 10 vetores): UNICA | 120.0 B/fato
[  3357s] === R) REGA: 60 fatos, curva recall × K (regar = +K) ===
[  3736s]   R.K= 1 (0.75KB): fp16 32 | i4pt 5 | i4g 4 | i8g 32 (/60) | 192.0 B/fato i4g
[  4115s]   R.K= 2 (1.50KB): fp16 60 | i4pt 18 | i4g 31 | i8g 60 (/60) | 49.5 B/fato i4g
[  4523s]   R.K= 4 (3.00KB): fp16 60 | i4pt 25 | i4g 46 | i8g 60 (/60) | 66.8 B/fato i4g
[  5017s]   R.K= 8 (6.00KB): fp16 60 | i4pt 38 | i4g 58 | i8g 60 (/60) | 105.9 B/fato i4g
[  5679s]   R.K=16 (12.00KB): fp16 60 | i4pt 59 | i4g 60 | i8g 60 (/60) | 204.8 B/fato i4g
[  5679s] === B1) o fato aguenta ruído nos pesos do decoder? ===
[  5923s]   B1.sigma    0.0: recall 40/40 (base 40)
[  5925s]   B1.sigma  0.001: recall 40/40 (base 40)
[  5925s]   B1.sigma  0.003: recall 40/40 (base 40)
[  5926s]   B1.sigma   0.01: recall 40/40 (base 40)
[  5926s]   B1.sigma   0.03: recall 15/40 (base 40)
[  5926s]   B1.sigma    0.1: recall 0/40 (base 40)
[  5926s] === B2) modelo RANDOM vs TREINADO — a sabedoria barateia o armazenamento? ===
[  6701s]   B2.treinado: K1 fp16 39 i4g 1 | K4 fp16 40 i4g 31 | K8 fp16 40 i4g 40
[  7430s]   B2.random  : K1 fp16 31 i4g 22 | K4 fp16 40 i4g 30 | K8 fp16 40 i4g 34
[  7430s] DONE M112v2 (7430s)
[W703 18:17:27.939846465 AllocatorConfig.cpp:29] Warning: PYTORCH_HIP_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())
>>> bytebrain/research/marco112_fact_forest.py OK (18:17:30)


========== RUN make-shorts-video/iara_pinboard/marco113_night_image.py  (18:17:30) ==========
[     0s] M113 START | base 5000 adapt 1200 | 180/fonte
/home/leonardo/projects/LLM/make-shorts-video/iara_pinboard/marco113_night_image.py:33: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
  out.append(torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0)
[    23s] treino 451 imgs | held-out 48
[    24s] 1) DOMÍNIOS EMERGENTES: {'cluster0': {'n': 184, 'dominante': 'aereo', 'pureza': 0.75}, 'cluster1': {'n': 117, 'dominante': 'indoor', 'pureza': 0.73}, 'cluster2': {'n': 52, 'dominante': 'natural', 'pureza': 0.98}, 'cluster3': {'n': 98, 'dominante': 'natural', 'pureza': 0.66}}
/home/leonardo/projects/LLM/.venv-rocm/lib/python3.12/site-packages/torch/nn/modules/module.py:1357: UserWarning: expandable_segments not supported on this platform (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:36.)
  return t.to(
[    27s]   núcleo passo 0: loss 0.0641
[    41s]   núcleo passo 500: loss 0.0094
[    55s]   núcleo passo 1000: loss 0.0067
[    69s]   núcleo passo 1500: loss 0.0063
[    83s]   núcleo passo 2000: loss 0.0052
[    97s]   núcleo passo 2500: loss 0.0043
[   111s]   núcleo passo 3000: loss 0.0052
[   125s]   núcleo passo 3500: loss 0.0062
[   140s]   núcleo passo 4000: loss 0.0050
[   154s]   núcleo passo 4500: loss 0.0049
[   202s]   adapter cluster0 (184 imgs, 11339 params → fp16 22.1KB / int4 5.5KB)
[   235s]   adapter cluster1 (117 imgs, 11339 params → fp16 22.1KB / int4 5.5KB)
[   268s]   adapter cluster2 (52 imgs, 11339 params → fp16 22.1KB / int4 5.5KB)
[   301s]   adapter cluster3 (98 imgs, 11339 params → fp16 22.1KB / int4 5.5KB)
[   301s] [aereo   ] núcleo 26.59 | certo_fp16 26.8 | certo_int4 25.24 | errado 25.92 | jpeg 25.24
[   301s] [pov     ] núcleo 33.27 | certo_fp16 32.51 | certo_int4 29.3 | errado 33.2 | jpeg 40.34
[   302s] [indoor  ] núcleo 32.43 | certo_fp16 32.58 | certo_int4 30.45 | errado 32.1 | jpeg 36.96
[   302s] [natural ] núcleo 29.35 | certo_fp16 29.41 | certo_int4 28.14 | errado 29.14 | jpeg 33.91
[   302s] 5) PODA (galho paga se ganho>0.15dB): {'aereo': {'ganho_fp16': 0.21, 'ganho_int4': -1.35, 'paga': True}, 'pov': {'ganho_fp16': -0.76, 'ganho_int4': -3.97, 'paga': False}, 'indoor': {'ganho_fp16': 0.15, 'ganho_int4': -1.98, 'paga': False}, 'natural': {'ganho_fp16': 0.06, 'ganho_int4': -1.21, 'paga': False}}
[   302s] roteador 0.78 | DONE M113 (302s)
[W703 18:22:49.418779980 AllocatorConfig.cpp:29] Warning: PYTORCH_HIP_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())
>>> make-shorts-video/iara_pinboard/marco113_night_image.py OK (18:22:50)


========== RUN bytebrain/research/marco114_wisdom_distill.py  (18:22:50) ==========
[     0s] FASE 1: professor Phi-4-mini responde 40 perguntas
[transformers] This model config has set a `rope_parameters['original_max_position_embeddings']` field, to be used together with `max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_parameters`with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, as it is compatible with most model architectures.
[     1s] Phi falhou (cannot import name 'LossKwargs' from 'transformers.utils' (/home/leonardo/projects/LLM/.venv-rocm/lib/python3.12/site-packages/transformers/utils/__init__.py)); fallback SmolLM2-1.7B
Loading weights:   0%|          | 0/218 [00:00<?, ?it/s]Loading weights:   0%|          | 1/218 [00:00<00:48,  4.48it/s]Loading weights:  14%|█▍        | 30/218 [00:00<00:01, 110.93it/s]Loading weights:  22%|██▏       | 49/218 [00:00<00:01, 121.58it/s]Loading weights:  29%|██▉       | 64/218 [00:00<00:01, 94.06it/s] Loading weights:  35%|███▍      | 76/218 [00:00<00:02, 68.26it/s]Loading weights:  39%|███▉      | 85/218 [00:01<00:02, 64.21it/s]Loading weights:  43%|████▎     | 93/218 [00:01<00:01, 66.36it/s]Loading weights:  47%|████▋     | 103/218 [00:01<00:01, 70.98it/s]Loading weights:  51%|█████     | 111/218 [00:01<00:01, 70.51it/s]Loading weights:  58%|█████▊    | 127/218 [00:01<00:01, 90.79it/s]Loading weights:  63%|██████▎   | 137/218 [00:01<00:01, 76.58it/s]Loading weights:  67%|██████▋   | 146/218 [00:01<00:00, 75.00it/s]Loading weights:  71%|███████   | 155/218 [00:02<00:00, 63.14it/s]Loading weights:  74%|███████▍  | 162/218 [00:02<00:00, 56.33it/s]Loading weights:  78%|███████▊  | 169/218 [00:02<00:00, 51.11it/s]Loading weights:  80%|████████  | 175/218 [00:02<00:00, 51.06it/s]Loading weights:  84%|████████▍ | 183/218 [00:02<00:00, 52.04it/s]Loading weights:  88%|████████▊ | 192/218 [00:02<00:00, 59.47it/s]Loading weights:  93%|█████████▎| 203/218 [00:02<00:00, 64.21it/s]Loading weights: 100%|██████████| 218/218 [00:03<00:00, 71.72it/s]
/home/leonardo/projects/LLM/.venv-rocm/lib/python3.12/site-packages/torch/nn/modules/module.py:1357: UserWarning: expandable_segments not supported on this platform (Triggered internally at /pytorch/c10/hip/HIPAllocatorConfig.h:36.)
  return t.to(
[transformers] The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[    21s]   P: Qual é a capital da Austrália?           → Canberra
[    22s]   P: Qual é o maior planeta do sistema solar? → O maior planeta do sistema solar é o pla
[    23s]   P: Quem pintou a Mona Lisa?                 → Leonardo da Vinci
[    24s]   P: Qual é o metal mais leve?                → O metal mais leve é o óxido de carbono
[    25s]   P: Em que ano o homem pisou na Lua?         → 1990
[    26s]   P: Qual é o maior oceano da Terra?          → O Oceano Pacifico
[    27s]   P: Qual é o rio mais longo do mundo?        → Rio de Janeiro
[    28s]   P: Qual é o osso mais longo do corpo humano → O osso mais longo do corpo humano é o os
[    29s]   P: Qual gás as plantas absorvem?            → O gás que as plantas absorvem é o oxigên
[    29s]   P: Qual é a moeda do Japão?                 → Yen
[    30s]   P: Quantos lados tem um hexágono?           → 6 lados
[    31s]   P: Qual é o símbolo químico do ouro?        → Au
[    32s]   P: Qual é a montanha mais alta do mundo?    → Montaña de Kilimanjaro, África
[    33s]   P: Quem escreveu Dom Casmurro?              → Machado de Assis
[    34s]   P: Qual é o menor país do mundo?            → Apenas um país tem um nome, o menor país
[    35s]   P: Qual planeta é conhecido como planeta ve → Mars
[    36s]   P: Qual é a capital do Canadá?              → Ottawa
[    37s]   P: Quantos continentes existem?             → 7
[    38s]   P: Qual é o animal terrestre mais rápido?   → O tatuá
[    38s]   P: Qual é a velocidade da luz aproximada em → 300.000 km/s
[    39s]   P: Qual é o maior mamífero do mundo?        → O mamífero mais grande do mundo é o ele
[    40s]   P: Quem descobriu a gravidade?              → A gravidade foi descoberta pelo Newton
[    41s]   P: Qual é o elemento mais abundante no univ → O elemento mais abundante no universo é 
[    42s]   P: Qual é a capital da Coreia do Sul?       → Seoul
[    43s]   P: Qual é o maior deserto quente do mundo?  → O deserto de Kalahari, que se localiza n
[    44s]   P: Quantos ossos tem o corpo humano adulto? → 206
[    45s]   P: Qual é a língua mais falada do mundo?    → A língua mais falada do mundo é a lí
[    46s]   P: Quem foi o primeiro presidente dos EUA?  → George Washington
[    47s]   P: Qual é o ponto de ebulição da água em Ce → 100°C
[    47s]   P: Qual é a capital da Argentina?           → Buenos Aires
[    48s]   P: Qual é o maior órgão do corpo humano?    → O maior órgão do corpo humano é o cé
[    49s]   P: Quantos planetas há no sistema solar?    → 8
[    50s]   P: Qual é a fórmula química do sal de cozin → Na fórmula de sal de cozinha, o sal é co
[    51s]   P: Qual é a capital do Egito?               → 100000000000000
[    52s]   P: Quem escreveu Romeu e Julieta?           → Shakespeare
[    53s]   P: Qual é o metal líquido à temperatura amb → O metal líquido à temperatura ambiente é
[    54s]   P: Qual é a maior floresta tropical do mund → Amazônia
[    55s]   P: Qual é a capital da Alemanha?            → Berlin
[    56s]   P: Quantas cordas tem um violino?           → 4
[    56s]   P: Qual é o planeta mais próximo do Sol?    → Apenas o Sol
[    56s] professor (SmolLM2-1.7B) respondeu 40/40
[    57s] FASE 2: aluno Math-1.5B congelado — baseline + destilação
Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]Loading weights:   0%|          | 1/338 [00:00<03:22,  1.66it/s]Loading weights:  40%|███▉      | 135/338 [00:00<00:00, 223.51it/s]Loading weights:  53%|█████▎    | 179/338 [00:01<00:01, 89.73it/s] Loading weights:  61%|██████    | 205/338 [00:02<00:01, 94.14it/s]Loading weights:  67%|██████▋   | 225/338 [00:02<00:01, 85.85it/s]Loading weights:  72%|███████▏  | 243/338 [00:02<00:01, 81.87it/s]Loading weights:  83%|████████▎ | 279/338 [00:02<00:00, 110.99it/s]Loading weights:  90%|█████████ | 305/338 [00:02<00:00, 128.96it/s]Loading weights: 100%|██████████| 338/338 [00:03<00:00, 112.24it/s]
[    87s]   destila [Qual é a capital da Austrália?    =Canberra      ] fp16 ok int4 ok
[   114s]   destila [Qual é o maior planeta do sistema =O maior planet] fp16 X int4 X
[   140s]   destila [Quem pintou a Mona Lisa?          =Leonardo da Vi] fp16 ok int4 ok
[   167s]   destila [Em que ano o homem pisou na Lua?  =1990          ] fp16 ok int4 ok
[   193s]   destila [Qual é o maior oceano da Terra?   =O Oceano Pacif] fp16 ok int4 ok
[   219s]   destila [Qual é o rio mais longo do mundo? =Rio de Janeiro] fp16 ok int4 ok
[   247s]   destila [Qual gás as plantas absorvem?     =O gás que as p] fp16 ok int4 ok
[   273s]   destila [Qual é a moeda do Japão?          =Yen           ] fp16 ok int4 ok
[   299s]   destila [Quantos lados tem um hexágono?    =6 lados       ] fp16 ok int4 ok
[   324s]   destila [Qual é o símbolo químico do ouro? =Au            ] fp16 ok int4 ok
[   351s]   destila [Qual é a montanha mais alta do mun=Montaña de Kil] fp16 ok int4 ok
[   377s]   destila [Quem escreveu Dom Casmurro?       =Machado de Ass] fp16 ok int4 ok
[   404s]   destila [Qual é o menor país do mundo?     =Apenas um país] fp16 ok int4 X
[   430s]   destila [Qual planeta é conhecido como plan=Mars          ] fp16 ok int4 ok
[   455s]   destila [Qual é a capital do Canadá?       =Ottawa        ] fp16 ok int4 ok
[   482s]   destila [Qual é o animal terrestre mais ráp=O tatuá       ] fp16 ok int4 X
[   508s]   destila [Qual é a velocidade da luz aproxim=300.000 km/s  ] fp16 ok int4 ok
[   535s]   destila [Qual é o maior mamífero do mundo? =O mamífero mai] fp16 ok int4 ok
[   561s]   destila [Quem descobriu a gravidade?       =A gravidade fo] fp16 ok int4 ok
[   588s]   destila [Qual é o elemento mais abundante n=O elemento mai] fp16 ok int4 ok
[   613s]   destila [Qual é a capital da Coreia do Sul?=Seoul         ] fp16 ok int4 ok
[   640s]   destila [Quantos ossos tem o corpo humano a=206           ] fp16 ok int4 ok
[   667s]   destila [Quem foi o primeiro presidente dos=George Washing] fp16 ok int4 ok
[   693s]   destila [Qual é o ponto de ebulição da água=100°C         ] fp16 ok int4 ok
[   719s]   destila [Qual é a capital da Argentina?    =Buenos Aires  ] fp16 ok int4 ok
[   748s]   destila [Qual é a fórmula química do sal de=Na fórmula de ] fp16 ok int4 ok
[   775s]   destila [Qual é a capital do Egito?        =10000000000000] fp16 ok int4 ok
[   801s]   destila [Quem escreveu Romeu e Julieta?    =Shakespeare   ] fp16 ok int4 ok
[   827s]   destila [Qual é a maior floresta tropical d=Amazônia      ] fp16 ok int4 ok
[   853s]   destila [Qual é a capital da Alemanha?     =Berlin        ] fp16 ok int4 ok
[   878s]   destila [Quantas cordas tem um violino?    =4             ] fp16 ok int4 ok
[   904s]   destila [Qual é o planeta mais próximo do S=Apenas o Sol  ] fp16 ok int4 ok
[   904s] RESUMO: aluno já sabia 8/40 | destilados int4 29/32 (0.91) | 26.5 B/fato
[   904s] DONE M114 (904s)
[W703 18:37:59.624990367 AllocatorConfig.cpp:29] Warning: PYTORCH_HIP_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead (function operator())
>>> bytebrain/research/marco114_wisdom_distill.py OK (18:38:01)

# FIM DA BATERIA — Fri Jul  3 06:38:01 PM -03 2026
