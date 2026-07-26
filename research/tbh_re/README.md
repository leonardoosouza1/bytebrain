# TBH-RE — kit de engenharia-reversa/entendimento do TaskbarHero (Unity IL2CPP)

Objetivo: **entender a arquitetura e o fluxo de código** do jogo (que você tem na Steam), e dar
**ferramentas ao Claude** pra explorar e testar. Escopo: entendimento / mod-dev / pesquisa no
próprio jogo. NÃO é pra burlar anti-cheat nem a economia online contra terceiros (o jogo tem
TheBackend BaaS + Steam trading + CodeStage AntiCheat com validação no servidor).

## Duas fases
**FASE 1 — ESTÁTICO (roda em QUALQUER máquina, CPU-only, só precisa de Python 3):**
1. `python3 tbh_dump.py [caminho_do_jogo]` → parser puro do `global-metadata.dat` (IL2CPP v31) →
   reconstrói 24.081 classes com campos+métodos → `tbh_map.json`.
2. `python3 tbh_tool.py <cmd>` → explora o mapa:
   - `stats` · `integrations` · `systems` · `tree`
   - `find <regex>` · `class <Nome>` · `methods <Nome>` · `refs <palavra>`
   - `flow <Nome>` (métodos = candidatos a hook) · `hookgen <sistema>`

**FASE 2 — RUNTIME (na sua máquina Linux sem GPU, via MelonLoader):** ver `melonloader/README.md`.
Observa o fluxo real: quais métodos disparam, quando. O `gen_hooks.py` escolhe os métodos legíveis
de um sistema; o mod `TbhTrace` loga cada chamada.

## O que já sabemos (Fase 1)
- 1850 classes de jogo (Assembly-CSharp.dll). Namespace `TaskbarHero.*`.
- Integrações: Steam (Heathen Steamworks), Easy Save 3 (save AES), CodeStage AntiCheat (ObscuredTypes),
  UniTask (async), Newtonsoft JSON, **TheBackend (BaaS online)**, DOTween, Addressables, InputSystem.
- Sistemas: Save(83) Boss(34) Skill(43) Item(136, validado no servidor) Combate(91) Player(569)
  UI(215) Backend(35) Economia Steam-trading Áudio Anti-cheat.
- Save: `AccountSaveData` guarda `lastExitSteamUnixSecEncrypted`+`Key` (anti-relógio) amarrado ao `ownerSteamId`.
- Parte do código é **ofuscada** (métodos viram `dwk/eju…`) — o hookgen filtra só nomes legíveis.

## Portabilidade
Fase 1 = só Python stdlib (copie a pasta + o jogo). Fase 2 = MelonLoader + .NET SDK (ver o outro README).
