
========================================================================
# BATERIA DE PRODUTO  — cérebro acoplado — 2026-07-11 00:16
========================================================================
205 fatos (85 capitais + 73 continentes + 47 moedas), gerador Qwen2.5-1.5B em cuda
forward em 205 fatos: 97s

## LOTE A — geração ABERTA multi-token (a métrica honesta)
  capital   : 82%  (70/85)
  continent : 99%  (72/73)
  currency  : 83%  (39/47)
  GERAL: 88%

## LOTE A2 — paráfrase: responde consistente em vários fraseados?
  acurácia por fraseado: ['88%', '58%', '90%'] · consistência (mesmo veredito nos 3): 60%

## LOTE B — verificador (concorda=confia) + calibração
  território CONCORDA (147) → acurácia 100%
  território DISCORDA (58) → acurácia 59%
  bandeira: precisão 41% · recall dos erros 100%
  ECE (2-bin concorda/discorda): 0.166  (0=perfeitamente calibrado)

## LOTE B2 — calibração: a acurácia sobe com a confiança do cérebro?
  conf ALTA  (tipo-válido + território concorda): 100%  (147/147)
  conf MÉDIA (válido, território discorda): 100%  (4/4)
  conf BAIXA (gerador desistiu): 56%  (30/54)
  (monotônico = calibrado: mais confiança → mais acerto)

## LOTE C — corretor: memória-neurônio recupera o erro?
  memória aponta a certa em 20/24 erros do gerador (83%)

## LOTE D — retrieval externo REAL (grafo-água, não gabarito)
  acurácia do retrieval (frase recuperada contém o fato certo): 111/205 = 54%

## LOTE E — pipeline completo MEDIDO + ablações
  só gerador ......................... 88%
  + correção NAIVE (override sempre) .. 93%   (Δ +5%)  ← se PIORA, prova que precisa de gate
  + PRODUTO (gate + RAG-água) ........ 98%   (Δ +9%)   RAG disparou em 58/205=28%

## LOTE F — robustez: typo no nome do país
  gerador limpo 82% → com typo 45% → typo+correção-neurônio 55%

## LOTE J — abstenção: entidade FALSA → o verificador levanta bandeira (abstém em vez de alucinar)?
  entidades FALSAS sinalizadas (abstém): 8/8 = 100%
  (vs falso-flag em reais: 58/205=28% — a diferença é o valor do sinal)

## LOTE H — custo do roteamento adaptativo
  gerador: 100% · verificador (grátis, mesmo forward): 100% · correção-interna: 28% · RAG-externo: 28%
  = o caro (RAG externo) dispara só em 28% das queries (só nas bandeiras)

## LOTE G — paleta: rotear pro especialista (Instruct/Math/Coder) paga?
  Instruct  matemática 0.487 bits · código 1.157 bits
  Math      matemática 0.345 bits · código 0.940 bits
  Coder     matemática 0.485 bits · código 0.753 bits
  → melhor matemática: Math · melhor código: Coder  (paleta paga)

# BATERIA CONCLUÍDA em 7.5 min · 00:23
Honesto: geração aberta, RAG-água real (não gabarito), ablações mostram o Δ de cada órgão, robustez e custo medidos.
