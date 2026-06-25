# Diário de evolução — ByteBrain overnight

Cada entrada (~30min) = o estado do modelo naquele momento: métricas de coesão + uma amostra gerada.
Lê de cima pra baixo pra ver a evolução da noite. Métricas: `wtrans` (menor=mais coeso, PT real ~6),
`distinct4` (alto=menos repetição), `rep-bigramas` (menor=menos loop), `span coerente` (maior=melhor).

---

### 24/06 23:45 · step 23000 · val_bpb 1.354
`wtrans 5.48` · `distinct4 0.73` · `rep-bigramas 15%` · `span coerente 67`

> O Brasil é um dos principais centros de desenvolvimento de água.

O Brasil foi criado pelo presidente Roberto Guimarães, que foi um dos maiores centros de pesquisa do Brasil. O país, no entanto, foi declarado como um dos capitães-generais que deveriam ser objeto de pesquisa. De acordo com o país, a maioria dos países-m

### 25/06 00:21 · step 23000 · val_bpb 1.358
`wtrans 5.68` · `distinct4 0.81` · `rep-bigramas 8%` · `span coerente 30`

> O Brasil em pouco tempo, que não foi atribuído.

Ao longo dos anos, o primeiro-ministro David Buckley afirmou: "Não há nada que tenha sido morto. Ele foi um dos principais movimentos de que as pessoas de John P. Buckley, que tiveram um bom número de pessoas que foram colocadas."

Em 2014, John P. Stillight publicou uma

↳ **ciclo 00:21:** as 2 entradas são do mesmo checkpoint (ckpt_best congelado em val 1.354) — diferenças = ruído de amostragem (span 67 vs 30). val em **plateau ~1.354-1.358** (foi 1.47→1.354 em 7h). FIX aplicado: avaliador agora usa `ckpt.pt` (modelo vivo) p/ rastrear evolução real. Treino segue (LR ainda decaindo, refino marginal restante).

### 25/06 00:45 · step 24528 · val_bpb 1.360
`wtrans 5.94` · `distinct4 0.73` · `rep-bigramas 10%` · `span coerente 48`

> O Brasil é um dos grandes prêmios, com o primeiro trabalho de Curitiba.

A música foi a principal fonte de música, que é o que o grupo de rock influente. A história, mas não é uma fonte de apoio ao que não é. Não há uma forma de criar um grupo, mas não há nada.

:Olá! Não é bem-vindo. Agradeço, porque a mesma fonte é u

↳ **ciclo 00:45:** avaliador CONSERTADO (agora ckstep 24528 = modelo vivo, não o congelado). val **plateau 1.354-1.360** (3 ciclos sem novo best). 40M convergindo perto do teto; LR ainda ~1.7e-4 (refino pode vir). Critério: se +2 ciclos sem novo best → acelero decay LR (50k→30k) pra forçar fase de refino. Treino VIVO seguindo.

### 25/06 09:30 · step 38264 · val_bpb 1.304
`wtrans 5.39` · `distinct4 0.71` · `rep-bigramas 14%` · `span coerente 49`

> O Brasil e o país, como um dos maiores jogadores de futebol.

O Brasil é um dos principais eventos de futebol, como o Brasil. O Brasil foi a principal categoria de futebol, com exceção de todos os países. Em 2016, o Brasil teve um grande sucesso comercial.

O Brasil é composto por quatro jogadores, um dos quais o Brasi
