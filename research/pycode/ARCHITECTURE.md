# ARCHITECTURE — do mundo sintético ao especialista em Python (ledger #27–#30)

Escrito 2026-10-26, só papel (nada rodado). Responde: "como treina? ajusta os pesos? cadê o banco de τ infinito?
como vira especialista em Python, macro e micro, pedido em pt-BR, Qwen/Sonnet como juiz, quanto da função está
correta?". Leia junto: `USER-REQUESTS.md`, `GUESS-AND-DERIVE.md`, `README.md` §7.

---

## 1. Como é hoje (verificado no código, não no resumo)

Três peças: **planner** (`synth/blockaddr.py`, 149 220 params), **writer** (`synth/bytewriter.py` /
`multifn.Writer`, 13k–45k params), **regras** (Python fixo). Juiz = `bwrap` (execução isolada).

### 1.1 TREINO — sim, os pesos são ajustados (backprop + AdamW)

```
 gerador sintético (regra)                       ┌──────────── PLANNER (149k) ──────────────┐
 task = (units, final)  ou spec de ops           │ bytes do NOME  "add_3_then_sort_then_sum" │
   │                                             │ emb(37 bytes) → conv k5 → conv k5 (local) │
   ├─ name_of(task) ──── entrada ───────────────▶│ gate g_i = σ(w·h_i)   "grupo termina aqui"│
   │                                             │ a_i = Σ_{j≤i} g_j     ← τ = ∞ (endereço)  │
   ├─ blocks_of(task) ── alvo (símbolos) ───────▶│ decoder: contador c_t, lê softmax(-(a-c)²/s)│
   │   [op:add d:3 op:sort fin:sum <eos>]        │ GRU(2d=128) → logits do símbolo + avanço p │
   │                                             │ c_{t+1} = c_t + p_t   ← τ = ∞ (contador)  │
   │                                             └──── loss = CE(símbolo), teacher forcing ────┘
   │                                                 AdamW lr 3e-3 wd 0.01, warmup 50, cosseno,
   │                                                 clip 1.0, batch 48, 1500–2000 passos, CPU
   │
   └─ block_text(bloco) ── alvo (chars) ────────▶ WRITER: cond = emb(op)+emb(dígito)  [bytewriter]
                                                          ou emb(símbolo)          [multifn]
                                                  GRU(d) char a char, h zerado a cada bloco
                                                  loss = CE(char), teacher forcing, AdamW, batch 64
```

- Planner e writer são treinados **separados**, cada um com sua CE; não há gradiente passando de um ao outro,
  nem da execução para nenhum dos dois. O alvo vem da **regra** que gerou a tarefa (supervisão perfeita).
- Wall medido: planner + writer d64 ≈ 95 s CPU (`bytewriter18/bytewriter_d64_s1.json`); multifn ≈ 107 s.
- No `synth/blockaddr` **não há self-training**. O laço que atualiza pesos com saídas verificadas existe só na
  linha anterior (byte-LM 3M, `locate.py`, loc_ce: 25.7 → 99.8% test_b; iteração on-policy = +42 pp).

### 1.2 INFERÊNCIA — pesos congelados, nada aprende

```
 nome ─▶ planner.greedy (argmax, MAX_OUT passos, corta no <eos>) ─▶ parse(símbolos)
        │ inválido → corpo "return None"   (regra)
        ▼
 writer.write(op, dígito) por bloco, greedy, máx 160 chars ─▶ texto de cada bloco
        ▼
 MONTADOR (regra): head(task) copiado da tarefa · indentação "    " · dedup de defs repetidos ·
                   main/dispatch + "return None" (multifn) · linha `return` do main renderizada
                   da árvore call/leaf (compose.render_expr)
        ▼
 bwrap: executa em 40 entradas (synth2) / 20 casos (a,b) (multifn/compose) → pass1 estrito, exec, plan_exact
```

### 1.3 O que é aprendido × o que é regra

| aprendido (pesos) | regra (hardcode, ok por agora) |
|---|---|
| onde termina cada grupo de bytes (gate) | vocabulário de blocos: 47 units + 13 finals (synth2), 15 ops (multifn) |
| qual símbolo vem em cada endereço | linha `def` copiada da tarefa (planner não escreve o cabeçalho) |
| quantos passos avançar (p_t) | indentação do corpo, dedup de defs, `main`/dispatch, `return None` |
| dependências (hyp → sqrt, add, square → mul) | `return` do main em compose (árvore → expressão) |
| texto de cada bloco, char a char (writer) | o cabeçalho; os valores de teste; o juiz |

### 1.4 Onde mora o banco de τ

```
 τ = ∞   a_i = cumsum(g)        endereço sobre a entrada: só muda num evento de fronteira ("stress")
 τ = ∞   c_t = c_{t-1} + p_t    contador do decoder: só anda quando ele decide avançar
 τ finito, aprendido   GRU do planner (gate z da GRU = vazamento por dimensão, 128 dims)
 τ curto, zerado/bloco GRU do writer (esquece tudo entre blocos: memória só local)
```

Ou seja: o "banco" hoje tem **dois extremos** (∞ e curto) e nada fixo no meio; não é um banco log-espaçado.
O achado `project_banco_de_tau` diz que **ter** τ espalhados paga e **modular** τ não paga — o desenho da §3.2
segue isso.

### 1.5 gradsearch: a derivada é na INFERÊNCIA, sobre logits, não sobre pesos

`synth/gradsearch.py`: não tem rede treinada. Cada slot = vetor de logits (47 units + identidade) iniciado
aleatório; executor diferenciável mistura as saídas de todos os units; loss vs 8 exemplos; **Adam (lr 0.3) move só
os logits** daquele problema; argmax → checa nos 8 → bwrap nos 40. Nenhum peso muda, nada fica aprendido para o
próximo problema. Resultado: perde para busca aleatória (len3 hidden 3% vs 33%). É o laço rápido do #26, ainda
sem o "chute" do planner.

---

## 2. Por que ainda não é um especialista em Python (lacunas honestas)

1. **Vocabulário fechado**: 47 units / 15 ops. Bloco = símbolo de uma tabela; não existe "escrever uma
   expressão nova".
2. **O nome soletra o programa**: `add_3_then_sort_then_sum` já é o plano em prefixo. O planner aprende a
   segmentar e copiar, não a entender intenção. compose: a árvore está no nome.
3. **Sem entrada em linguagem natural**: nem pt-BR, nem descrição, nem exemplos (exceto gradsearch).
4. **Sem variáveis, sem `if` livre, sem laço livre**: o estado é sempre `xs` (ou `a, b`); `if`/`for` só existem
   dentro de blocos copiados. Nenhum escopo, nenhum nome inventado.
5. **Writer não escreve op nunca visto**: condicionado em `emb(op)`; op novo = embedding não treinado.
   Só compõe op visto + dígito novo (test_num 0.72–0.86 conforme d).
6. **Sem crédito parcial**: a métrica é pass1 estrito 0/1; não se sabe "quanto" de uma função errada está certo.
7. **Sem laço lento no sistema planner+writer**: nada do que o bwrap aprova volta para os pesos.
8. **Comparação Qwen é dentro do nosso mundo** (nós 100 vs Qwen-0.5B 8–53): prova especialização, não
   competência em Python.

---

## 3. Arquitetura proposta

### 3.0 Visão do fluxo completo

```
 "faz uma função que recebe uma lista de notas e devolve a média das aprovadas (>= 6)"   (pt-BR)
        │
        ▼
 ┌─ TRADUTOR (Qwen2.5-Coder 3B q8, congelado, few-shot + gramática GBNF) ─────────────────────────┐
 │ saída = SPEC na DSL interna (§3.1): assinatura + intenção em passos + TESTES (entrada → saída) │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
        │ spec (texto curto, gramática fechada → parse sempre válido)
        ▼
 ┌─ PLANNER hierárquico (GLOBAL) ─ endereços módulo→função→bloco→linha, banco de τ (§3.2) ────────┐
 │ emite: grafo de chamadas (macro) e, por função, cadeia de blocos com intenção (micro)          │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
        │ por bloco: (intenção, endereço, tabela de nomes em escopo)
        ▼
 ┌─ WRITER (LOCAL) ─ bytes de UMA linha/bloco; pointer-copy de nomes/constantes da spec (§3.4) ──┐
        │
        ▼
 MONTADOR (regra: indentação, ordem de defs, dedup) ─▶ código Python
        │
        ▼
 JUÍZES (§3.3): J0 sintaxe (GPU) → J1 bwrap nos testes (verdade) → J2 LLM (Qwen/Sonnet, calibrado)
        │ score vetor: parse, exec, io_frac, first_wrong_block, j2
        ├── falhou e orçamento sobra → LAÇO RÁPIDO (§3.5): chuta+deriva / reamostra o bloco
        └── passou → resposta  +  vai para o pool do LAÇO LENTO (§3.5): pesos são re-treinados
```

### 3.1 DSL interna (o que o tradutor escreve e o planner lê)

Linhas, uma declaração por linha, gramática fechada (GBNF no llama-server → nunca sai inválida).

```
MODULE notas                                  ← MACRO: módulo
FN media_aprovadas(notas: list[float]) -> float   ← MACRO: função + assinatura
  CALLS media                                 ← MACRO: aresta do grafo de chamadas
  STEP filter notas  x >= 6  -> aprovadas     ← MICRO: bloco (verbo da tabela, args, nome de saída)
  STEP guard_empty aprovadas  0.0             ← MICRO
  STEP call media aprovadas -> m              ← MICRO
  STEP return m                               ← MICRO
  TEST ([7, 4, 9],) -> 8.0                    ← verdade: vira juiz J1
  TEST ([],) -> 0.0
FN media(xs: list[float]) -> float
  STEP reduce_sum xs -> s
  STEP return s / len(xs)
  TEST ([2, 4],) -> 3.0
```

- **Macro** = `MODULE`, `FN` (nome, parâmetros, tipo), `CALLS` (grafo). **Micro** = `STEP` (verbo de uma tabela
  aberta de ~30 verbos: map, filter, reduce_*, guard_*, if, for_acc, call, assign, return…; args são
  **expressões Python curtas**, não ids) e `TEST`.
- Fase 1: o tradutor escreve só macro + TEST + STEP com verbos; o planner refaz/ordena os STEP. Fase 3+: o
  tradutor só escreve macro + TEST e o planner gera os STEP (senão o Qwen faz todo o trabalho — §5 Q1).
- STEP args contêm expressões (`x >= 6`, `s / len(xs)`): é aqui que o vocabulário abre.

### 3.2 Macro/micro, local/global: endereço hierárquico + banco de τ

```
 nível     endereço (inteiro, τ=∞)     estado de conteúdo (τ finito, fixo, log-espaçado)   quem lê
 ───────   ─────────────────────────   ──────────────────────────────────────────────    ────────
 L0 módulo  m = Σ g_mod                 s0  τ≈∞ (média do módulo inteiro: "do que se trata")  planner
 L1 função  f = Σ g_fn  (reset em L0)   s1  τ≈ tamanho da função (≈200 bytes)                planner
 L2 bloco   b = Σ g_blk (reset em L1)   s2  τ≈ tamanho do bloco (≈40 bytes)                  planner+writer
 L3 linha   l = Σ g_line(reset em L2)   s3  τ≈ 8 bytes                                       writer
 L4 byte    i (posição na linha)        conv k5 (janela local)                               writer
```

- Cada nível tem seu **gate de fronteira aprendido** (como o `gate` atual): o contador do nível sobe 1 quando o
  gate dispara e o contador do nível de baixo zera. Endereço completo = tupla (m, f, b, l, i). É o `cumsum`
  de hoje repetido em 4 escalas — o τ=∞ que você pediu, mas um por nível.
- Estados de conteúdo: `s_k ← s_k + (1/τ_k)·e_k·(x − s_k)`, onde `e_k` = evento do nível (só atualiza no
  "stress" daquele nível). τ **fixos e espalhados** (achado: espalhar paga, modular não paga).
- **Lento = macro/global, rápido = micro/local**: o writer recebe [s2, s3, conv local, tabela de nomes]; o planner
  recebe [s0, s1, s2] + leitura por endereço como hoje (`softmax(-(a-c)²/s)`), agora por nível.
- Teste falsificável (S2): mesmo modelo faz multifn **e** compose **e** synth2 (3 mundos, 1 rede).
  Previsão: len8/calls8 ≥ 90% (hoje 100/93–100 com 3 redes separadas). **Kill**: perda > 10 pp em qualquer
  split vs a rede dedicada, ou ablação "sem s0–s2" igual a "com" (± 2 pp) → os estados lentos não fazem nada,
  volta-se ao endereço puro. Custo: CPU minutos (~150k–400k params), 3 sementes ≈ 15 min.

### 3.3 Juízes — quanto da função está correta

| juiz | o que mede | treino | inferência |
|---|---|---|---|
| J0 sintaxe: `ast.parse` + autômato L1 em GPU (já existe) | 0/1 e posição do 1º byte ilegal | filtro | máscara durante a geração |
| J1 bwrap nos TESTs | `exec`, `io_frac` = testes iguais / total, `first_wrong_block` (roda prefixos de blocos e checa o contrato de cada STEP quando há TEST de bloco) | **única verdade** que admite amostra no pool | decide aceitar / reparar |
| J2 LLM (Qwen-3B local; Sonnet via `claude -p` com `CLAUDE_CONFIG_DIR` limpo) | nota 0–10 + "linha suspeita" + "os TESTs cobrem o pedido?" | só peso/desempate, nunca admite sozinho | aponta onde a spec/teste não cobre o pedido |

- Resposta direta a "o quanto dela está correta?": um **vetor**, não um número: `parse, exec, io_frac,
  blocks_ok/blocks, first_wrong_block, j2`. Em sintético, `blocks_ok` é exato (temos o estado de referência por
  linha, como loc_ce); em pt-BR real, só onde a spec trouxer TEST por bloco.
- **Calibração de J2**: 300 funções com J1 conhecido (metade certas, metade com 1 bloco trocado — mutações já
  existem no synth). Medir AUC(J2 vs J1) e acerto de `first_wrong_block`. J2 entra no laço só se AUC ≥ 0.85
  (critic.py atual: 0.82). Custo: Qwen-3B ~5 s/função GPU → 25 min; Sonnet 300 chamadas sequenciais.
- Armadilha conhecida: o tradutor escreve os TESTs **e** o juiz confia neles. Teste errado = verdade errada.
  Mitigação: no marco (§4) os TESTs de avaliação são escritos à mão por nós, não pelo Qwen; os do Qwen servem
  só para o laço interno (seleção/reparo), e medimos à parte quantos TESTs do Qwen estão errados.

### 3.4 Writer de vocabulário aberto

```
 entrada: verbo do STEP + endereço (b, l) + s2,s3 + TABELA DE NOMES (params, saídas de STEPs anteriores, consts)
 saída por byte:  mistura  p = λ·gerar_byte  +  (1−λ)·copiar_nome_da_tabela   (pointer/copy)
```

- Nomes e constantes **não** precisam estar no vocabulário: são copiados da tabela (resolve "op nunca visto"
  quando o op é uma expressão de peças vistas: `x >= 6` = nome + operador + constante).
- O writer escreve **uma linha** por chamada; o `if`/`for` vira STEP com corpo (endereço L3 dentro de L2).
- Teste falsificável (S3, mundo synth v3: argumentos, `if`, `for`+acumulador, variáveis nomeadas): treina em
  verbos×expressões de 2 operandos, testa expressões de 3 operandos e nomes de variáveis nunca vistos.
  Previsão: pass1 ≥ 80% em nomes novos, ≥ 50% em expressões de 3 operandos. **Kill**: nomes novos < 50% →
  pointer-copy não funciona nesse tamanho; tentar d maior (d256 ter já deu 100% em synth2) antes de abandonar.
  Custo: writer 50k–600k params, CPU 2–7 min por semente (bytewriter d256 ter = 384 s).

### 3.5 Os dois laços

```
 LAÇO RÁPIDO (inferência, por pedido, ms–s, PESOS CONGELADOS)          LAÇO LENTO (treino, horas, PESOS MUDAM)
 ┌──────────────────────────────────────────────┐                     ┌─────────────────────────────────────┐
 │ GUESS: planner dá logits por slot             │                     │ 1. Qwen parafraseia/gera N pedidos  │
 │ DERIVE: executor diferenciável (gradsearch)   │                     │    pt-BR + spec (com TEST)          │
 │   Adam nos LOGITS até |Δloss| < ε (#26)       │   verificadas ───▶  │ 2. pipeline inteiro, k amostras     │
 │   só para STEPs que são experts conhecidos    │                     │ 3. J1 por bloco; mantém só 100%     │
 │ SNAP argmax → J1 bwrap                        │                     │ 4. CE nos planner+writer (AdamW)    │
 │ falhou: reamostra a partir do first_wrong     │                     │    metade próprio / metade replay   │
 │ passou: devolve + manda pro pool              │   ◀── pesos novos   │ 5. nova rodada com o modelo novo    │
 └──────────────────────────────────────────────┘                     └─────────────────────────────────────┘
```

- **Rápido** ajusta *logits* de um problema e esquece. **Lento** ajusta *pesos* e fica. A saída boa do rápido
  vira alvo do lento: é assim que a derivada "ensina" a intuição (amortização).
- Programas verificados viram **experts novos** na biblioteca (novo verbo/STEP chamável): os módulos crescem.
- Evidência já paga: no lento, iteração on-policy + verdade por linha deu +42 pp e +11 pp (loc_ce). No rápido,
  reparar a partir do erro = reamostrar com mesmo orçamento (README §7, lição 7) — então o reparo só se
  justifica se o DERIVE vencer reamostragem; isso é o teste S4.
- Teste S4 (guess+derive, synth2 len3–4 com 16 exemplos, `strict=False`): 3 braços com orçamento igual de
  execuções — guess só / derive só / guess+derive. Previsão: guess+derive ≥ guess + 10 pp em hidden40 nos
  problemas onde guess erra. **Kill**: guess+derive ≤ guess+reamostragem (k igual) → derivada não paga, fica só
  seleção por execução. Custo: GPU em lote (todos problemas × slots em um tensor) — minutos; CPU proibido p/ bateria.

### 3.6 Escada até Python real

| estágio | mundo | o que abre | teste/kill |
|---|---|---|---|
| S0 (hoje) | synth2, multifn, compose | — | feito: 100% dentro do mundo |
| S1 | os mesmos mundos | **entrada pt-BR** via tradutor | §4 (marco) |
| S2 | os 3 mundos numa rede | endereço hierárquico + banco τ | §3.2 |
| S3 | synth v3: args, `if`, `for`+acc, variáveis | writer aberto (pointer-copy) | §3.4 |
| S4 | S3 + exemplos | laço rápido guess+derive | §3.5 |
| S5 | S3 + rodadas | laço lento com juiz J1 por bloco (+J2 se calibrado) | ≥ +20 pp após 6 rodadas vs controle de passos iguais; kill se ≤ controle |
| S6 | **Python real limpo** | verbos = tipos de nó AST (Assign, If, For, Return, Call…), não units | abaixo |

S6 (pedido #1: só funções, sem comentários): corpus de funções reais **autocontidas** (só builtins), docstring e
comentários removidos, ≤ 30 linhas, dedup; TESTs gerados por Qwen-3B e **aceitos só se a função original os
passa** em bwrap (a função real é a referência). Spec DSL extraída do AST por regra (STEP = statement de topo).
Teste: pedidos pt-BR escritos à mão para 100 funções held-out; métrica io_frac/pass1 por J1 com nossos testes.
Kill: pass1 < Qwen-0.5B sozinho no mesmo conjunto → a arquitetura não escala para fora do sintético; reportar.
Lembrete do 26M real (README §7): aprendeu FORMA, zero SIGNIFICADO — S6 só depois de S3/S5 mostrarem
significado com vocabulário aberto.

---

## 4. Primeiro marco: "pede em pt-BR, sai função Python correta"

**Menor experimento possível**: S1 sobre o mundo multifn+compose (calculadora), sem mudar planner nem writer.

```
 pedido pt-BR ──▶ Qwen-3B (GBNF: só aceita "calc_" + ops do vocabulário, ou árvore prefixa)
             ──▶ nome/spec ──▶ planner multifn/compose (já treinado, 100%) ──▶ writer ──▶ bwrap
```

- Conjunto: **150 pedidos pt-BR escritos à mão** (nós, não o Qwen), 3 estilos: direto ("soma e
  subtração"), indireto ("a hipotenusa de um triângulo com catetos a e b"), composto ("a média entre a e a
  diferença de a e b"). Splits: 50 / 50 / 50. Referência = módulo de referência do multifn/compose.
- Braços (mesmo juiz bwrap, mesmos 20 casos, greedy):
  - **A** nosso: Qwen-3B tradutor (GBNF) → planner+writer.
  - **B** Qwen-3B sozinho escreve o módulo inteiro em Python a partir do pedido (chat template, 8 shots).
  - **C** Qwen-0.5B como tradutor (A com tradutor menor) — quanto do ganho é do tradutor.
  - **D** B + seleção por execução k=8 contra os TESTs que o próprio Qwen escreveu (o baseline forte, igual
    ao hybrid codegen).
  - Diagnóstico: % de traduções corretas (spec == spec de referência) separa erro do tradutor de erro nosso.
- Previsão: A ≥ 90% direto, ≥ 70% composto; A ≥ B + 15 pp no composto; A ≈ D no direto.
- **Kill**: A ≤ B em todos os estilos → o tradutor está fazendo o trabalho e o nosso sistema só copia; nesse
  caso a peça útil é o juiz/seleção, não o gerador (consistente com `insight_selector_not_generator`).
- Custo: VRAM Qwen-3B q8 ≈ 3.6 GB pesos + ~1 GB KV/ctx 4k ≈ 4.6 GB de 12 GB (38%, folga ok). RAM < 3 GB.
  150 pedidos × 4 braços, ~3–6 s cada em GPU ≈ 30–40 min. Planner/writer: inferência ms. Escrever os 150 pedidos:
  1–2 h humanas (ou Sonnet escreve, nós revisamos — mas então declarar).
- Honestidade embutida: neste marco o "entendimento" do pt-BR é do Qwen; o que se testa é se a divisão
  tradutor→planner→writer+juiz bate o Qwen sozinho. O entendimento próprio vem em S3/S6.

---

## 5. Perguntas para debater (com minha recomendação)

1. **Quanto a spec do tradutor pode conter?** Se o Qwen escrever os STEPs, ele já resolveu o problema.
   *Recomendo*: marco com STEPs do Qwen (aceitar a muleta); a partir de S3 o tradutor só escreve macro + TEST,
   e o planner gera os STEPs — o ganho nosso tem que aparecer aí.
2. **Quem escreve os TESTs da avaliação?** *Recomendo*: avaliação = testes humanos (ou Sonnet + revisão sua);
   TESTs do Qwen só no laço interno, com taxa de erro medida.
3. **J2 (LLM juiz) pode admitir amostra para o treino?** *Recomendo*: não. Só J1 admite; J2 vira peso/desempate
   e só depois de AUC ≥ 0.85 contra J1. Sonnet só em calibração/auditoria (custo e latência), Qwen-3B no laço.
4. **Treinar planner e writer juntos (fim a fim) ou separados?** *Recomendo*: separados até S3 (cada um tem
   alvo exato e é o que deu 100%); fim a fim só no laço lento, com CE nos dois sobre o mesmo programa verificado.
5. **Derivada vale o esforço?** Hoje perde para busca aleatória. *Recomendo*: um teste S4 bem feito (guess +
   derive vs guess + reamostragem, orçamento igual, GPU em lote); se perder, manter a derivada só para STEPs
   contínuos (constantes numéricas, limiares) e usar seleção por execução para os discretos.
6. **Quando pular para Python real (S6)?** *Recomendo*: só quando S3 mostrar ≥ 80% com nomes/expressões nunca
   vistos; antes disso o modelo pequeno repete o 26M (forma sem significado).
