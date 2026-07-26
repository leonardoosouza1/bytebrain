# TaskbarHero — Relatório de vulnerabilidade (disclosure responsável)

**Tipo:** Estado de jogo client-authoritative + rate-limiting apenas no cliente → amplificação de farm por manipulação de velocidade (Time.timeScale).
**Severidade:** Média-Alta (integridade de progressão/economia; a economia é compartilhada via TheBackend + Steam market).
**Descoberto por:** jogador no próprio cliente, sem interação com contas de terceiros. Reportado para correção.

## Resumo
O ritmo de combate (e portanto a taxa de kills que geram baús de chefe) é resolvido **no cliente**. Acelerando o
tempo do jogo (`UnityEngine.Time.timeScale`), o cliente reporta mortes de mob ~N× mais rápido; o servidor aplica a
chance de baú **por kill reportado** sem validar a taxa contra o **relógio real**. Resultado observado: com timeScale ~6×,
o farm de baús de chefe ficou ~6× o de um jogador normal (≈6 min de farm em 1 min real).

## Comportamento observado (PoC — descrição, sem exploit armado)
- `Time.timeScale` elevado → fases completam e dropam baú muito mais rápido; kills reportados ao backend na mesma cadência acelerada.
- Efeitos client-side persistem além do esperado (ex.: buff de +90% ATK de um personagem continuou aplicado após removê-lo até a troca de fase; clear de fase em 1 hit) → confirma que parte do estado de combate é resolvida/mantida no cliente.

## Causa-raiz (evidência de código — reconstruída dos binários IL2CPP)
- **`TaskbarHero.StageBoxRequestAbuseGuard`** — a *única* proteção de taxa é **client-side** (roda em Assembly-CSharp) e baseada em **intervalo de tempo** (campos `m_detectionIntervalSeconds`, `m_requestThreshold`, `m_blockDurationMinutes`). Sendo client-side, é desativável; sendo por intervalo, é sensível à escala de tempo.
- **Fluxo de kill→baú:** `TaskbarHero.Monster.DestroyUnitMySelf` / `Manager.MonsterSpawnManager.RemoveDeadMonster` (morte do mob) → chance de baú → `UI.StageBox` / `BoxCreateItem` → **`InventoryProcessBoxRequest.ToJson`** (envia ao backend; usa `CodeStage Obscured*`, ou seja, ofuscação **client-side**, não autoridade de servidor).
- **Autoridade:** a chance/quantidade de baú é aplicada por evento de kill vindo do cliente; não há verificação server-side de "kills por segundo de relógio real" nem de "tempo de fase plausível".

## Por que a mitigação atual não basta
- **Rate-limit no cliente** (`StageBoxRequestAbuseGuard`) pode ser desligado por qualquer mod loader (ex.: MelonLoader/HarmonyX). Defesa client-side é sempre contornável.
- **ObscuredTypes (CodeStage)** protegem valores na memória contra edição direta, mas **não** tornam o servidor autoritativo — o servidor ainda confia no *evento* reportado.

## Impacto
- Farm de baús/itens/progressão acima do pretendido → distorce economia (itens fluem pro **Steam trading/market** = afeta outros jogadores) e ranking/eventos.

## Correção recomendada (server-side)
1. **Validar taxa contra relógio real no servidor:** kills/baús por janela de **wall-clock** (não tempo de jogo). Rejeitar/limitar quando a taxa exceder o máximo fisicamente possível da fase.
2. **Sanidade de tempo de fase:** o servidor conhece HP do mob e DPS máximo teórico do build → tempo mínimo de clear; clears abaixo disso = suspeitos.
3. **Tornar a concessão de baú server-authoritative:** o servidor deriva os baús do progresso validado, não de eventos de kill crus do cliente.
4. **Não depender de guardas client-side** (`StageBoxRequestAbuseGuard`) como única barreira — mover a lógica de rate-limit para o backend.
5. (defesa em profundidade) telemetria server-side de `timeScale`/taxa anômala + rate-limit por conta no endpoint de baú.

## Nota de disclosure
Teste feito apenas na própria conta/cliente, sem manipular contas ou itens de terceiros. Objetivo: permitir a correção
da arquitetura de validação. Recomenda-se mover a autoridade de combate/kill/baú para o servidor.
