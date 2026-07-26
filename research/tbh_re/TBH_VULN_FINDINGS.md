# TaskbarHero — achados de segurança (varredura Claude sobre a RE)
> 156 riscos candidatos em 323 classes sensíveis. Disclosure responsável — análise, sem exploit.

## [ALTA] Client-Authority em Timing de Skill + Efeitos Não Validados
- **classe:** `SkillSO`
- **problema:** SkillSO define timings (preTime, actionTime, postTime) e parâmetros de efeito que podem estar sendo confiados pelo servidor sem validação. Em um RPG com trading+BaaS, se rate-limiting de skills ou cálculo de dano/área estiver no cliente, há risco de spam de ataques, dupe, ou manipulação de posições (traps).
- **verificar:** 1) Servidor valida `preTime/actionTime` antes de processar próximo comando de skill, ou só bloqueia no cliente? 2) Dano/healing/statusEffect são calculados no cliente ou validados no servidor? 3) Posições de trap (`WillSpawnTrap*Offset*`) vêm do cliente sem sanidade check? 4) Alguém com Cheat Engine consegue edit stats/cooldowns na memória e o servidor aceita?
- **correção:** Servidor DEVE: (a) manter cópia canônica de SkillSO e validar cada ação contra ela; (b) rejeitar skills se cooldown/casttime não transcorreu no servidor; (c) calcular dano/heal/area no servidor, apenas aceitar `skill_id + target` do cliente; (d) validar spawn positions (turrets/traps) em zona segura server-side.

## [ALTA] Client-Authority na Spawn de Monstros — Dupe de Itens e Economy Break
- **classe:** `MonsterRandomSpawnData`
- **problema:** SpawnCount, RandomSpawnMonsterKey e offsets são editáveis no cliente sem validação server-side. Attacker edita SpawnCount (1→999), escolhe monstro premium, bypassa rate-limit cliente → economia quebra, dupe de loot ilimitado.
- **verificar:** Fiddler/Wireshark: capture pacote de spawn, edite JSON/binary SpawnCount: 1→999, envie. Se 999 monstros com loot aparecem, vulnerável. Repita 100× em 1s → rate-limit só cliente.
- **correção:** Servidor gera spawns aleatoriamente (seed time/player-ID, nunca confie em client). Client recebe apenas spawn-event-ID. Valide: SpawnCount ∈ [1, MAX_SAFE], RandomSpawnMonsterKey ∈ whitelist, rate-limit server (1 spawn/s por player). Signature request com HMAC.

## [ALTA] Campo 'bbug' suspeito + client-authority de ouro
- **classe:** `GoldTexture`
- **problema:** Campo 'bbug' parece ser flag de debug em produção. Se 'text_Gold' ou 'feedback_Create' forem usados para validar/transmitir quantidades de ouro sem validação server-side, permite dupe de item ou inflation.
- **verificar:** 1) Procurar por 'bbug' no código — se for true/false, é debug que não deveria estar vivo; 2) Rastrear quem chama GoldTexture.text_Gold e se envia para TheBackend sem verificação; 3) Verificar se servidor confia em valores de ouro do cliente antes de efetuar trade no Steam
- **correção:** 1) Remover 'bbug' completamente; 2) NUNCA transmitir quantidades de ouro do cliente — servidor valida inventário local; 3) Steam trading requer server-side signature; 4) AntiCheat deve monitorar eventos de ganho de ouro suspeito

## [ALTA] Public EXP Fields Editable via Memory Tampering
- **classe:** `OfflineHeroSlot`
- **problema:** Os campos `baseExp` e `addedExp` são públicos int sem encapsulamento. Em modo offline, são editáveis via CheatEngine/Frida; ao sincronizar com servidor, valores fraudulentos são enviados como 'verdade' se o servidor não valida contra histórico ou regras de progressão imutáveis.
- **verificar:** 1) Editar baseExp/addedExp via memory editor enquanto jogo offline; 2) Sincronizar com servidor; 3) Verificar se XP aumentado persiste ou é rejeitado. 4) Examinar logs servidor: valor recebido é validado contra progression rules ou apenas trust no client?
- **correção:** Server-side: Recomputar XP total a partir de imutável log de ações (kills, quests) ao sync, nunca aceitar valores diretos. Client: encapsular em property com setter privado, usar playprefs encrypted ou local secure storage. Adicionar rate-limit e anomaly detection (XP +1000000 em 1s = suspeito).

## [ALTA] Client-Authority em SpawnSomethingData — Spawn Ilegal de Prefabs e Itens
- **classe:** `SpawnSomethingData`
- **problema:** A classe permite que o cliente escolha qual prefab invocar (SpawnPrefab) e envie isso ao servidor sem validação. Padrão idêntico ao bypass de StageBoxRequestAbuseGuard: cliente reporta evento, servidor confia. Permite duplication de loot, spawn de items normalmente não obtíveis e farm acelerado via timeScale.
- **verificar:** 1) Hookar BoxCreateItem ou InventoryProcessBoxRequest para logar SpawnPrefab values reais. 2) Editar SpawnPrefab em memória pra invocar item_lendario/boss_drop em fase normal. 3) Enviar múltiplas SpawnSomethingData por segundo (timeScale 6× ou ASM edit). 4) Verificar se servidor aplica o drop sem validar 'é fisicamente possível spawnar isso agora?'
- **correção:** 1) Servidor gera e valida drops, cliente SÓ reporta 'completei fase' + checksum. 2) Validar por drop policy: boss.GetDropTable() → filtra items lícitos. 3) Rate-limit server-side: drops/min de relógio real, nunca cliente. 4) Seed drops no servidor com ServerTime, não ClientTime. 5) Ofuscação (CodeStage) é cosmetica—move autoridade pro backend.

## [ALTA] Client Authority em Cálculo de Dano
- **classe:** `TaskbarHero.DamageInfo`
- **problema:** A classe DamageInfo encapsula OriginDamage e IsCritical como dados enviáveis do cliente. Se o servidor aceita essas valores sem recalcular baseado em stats do servidor, permite que o cliente injecte danos arbitrários, forçe críticos, e quebre economia/combate.
- **verificar:** Capture tráfego (Burp Suite/Charles). Envie um evento de ataque com OriginDamage=999999 e IsCritical=true contra um PNJ/inimigo. Se o HP dele cai pelo valor injetado, confirma vulnerability.
- **correção:** Servidor recalcula COMPLETAMENTE o dano usando: stats do attacker (vindos do BD servidor), fórmula de dano (servidor), RNG de crítico (servidor). Cliente recebe SÓ o resultado final (dano aplicado, se crítico). DamageInfo é struct READ-ONLY do servidor em direção ao cliente — nunca confia em valores reversos.

## [ALTA] Client-Authority ServerState — Simulação de Offline/Maintenance no Cliente
- **classe:** `TaskbarHero.ServerState`
- **problema:** A classe ServerState definida no cliente permite que a lógica do jogo tome decisões baseadas em um estado que o próprio cliente controla. Padrão clássico: bypass de validação quando 'Offline' ou 'Maintenance', ou memory-editing para forçar estados que desabilitam checks server-side.
- **verificar:** 1) Grep: procurar `if (ServerState == Offline)`, `if (ServerState == Maintenance)` — se houver lógica condicional que relaxa validação, é crítico. 2) Reflection: testar se enum pode ser modificado via CheatEngine/memory editor. 3) Offline mode: pode o cliente jogar, acumular recursos, e depois sync pro servidor sem validação?
- **correção:** 1) Servidor valida TUDO independente do estado (moeda, itens, progresso). 2) Offline é puramente visual/cache — servidor recalcula e valida na reconexão. 3) Rate-limits, gacha, economia: servidor-side only. 4) Client envia ações, servidor aprova e confirma estado real.

## [ALTA] Client-Authority em CurrencyKey (Inventário de Moedas Não Validado)
- **classe:** `TaskbarHero.CurrencyData`
- **problema:** Se CurrencyKey é serializado/enviado ao servidor para processar transações (gasto, troca, trading) sem validação contra lista branca server-side, o cliente pode inventar moedas fictícias, duplicar, ou bypassar regras de economia. A classe sugere que cada moeda é identificada apenas por um string/enum — sem servidor validar se esse ID existe no jogo, qualquer payload customizado é aceito.
- **verificar:** 1) Interceptar POST de transação (ex: /api/spend-currency, /api/trade) → enviar currencyKey='FAKE_CURRENCY_X' com valor alto. 2) Se o servidor responde 200 OK ou credita o saldo, é vulnerável. 3) Verificar se há rota /api/validate-currency que o cliente chama ANTES de transação — se existe, servidor pode estar ciente; se não existe, é negligência. 4) Checar logs: servidor registra reject de IDs inválidos?
- **correção:** Server-side: (A) Manter HashMap<CurrencyKey, CurrencyDef> de moedas válidas. (B) Em toda transação, validar key contra essa lista ANTES de creditar. (C) Nunca confiar em quantidades/balances enviadas pelo cliente — recomputar do histórico server-side. (D) Rate-limit SERVIDOR, não cliente.

## [ALTA] Credenciais de Backend Exposto no Cliente
- **classe:** `TS.BackendServerSettingSO`
- **problema:** BackendServerSettingSO armazena signatureKey e functionAuthKey no cliente (ScriptableObject). Qualquer pessoa pode extrair essas chaves via descompilação IL2CPP ou edição de memory, depois forjar requisições autenticadas ao backend (duplicação de itens, bypass de rate-limits, modificação de inventário).
- **verificar:** 1) Descompilar APK/build com Il2CppDumper → achar BackendServerSettingSO.cs → confirmar se m_signatureKey/m_functionAuthKey estão hardcoded ou serializados. 2) Tentar ler o ScriptableObject em runtime via MelonLoader/Frida. 3) Verificar se servidor valida assinatura ou apenas confia na key do cliente.
- **correção:** 1) NÃO armazenar secrets no cliente. 2) Gerar signatureKey server-side, entregar ao cliente em login (session token temporário). 3) Servidor SEMPRE valida/re-assina requisições antes de aplicar efeito no game state. 4) Rate-limit no servidor, não no cliente.

## [ALTA] Client-authority damage calculation sem server-side re-validation
- **classe:** `TaskbarHero.Monster`
- **problema:** Classe Monster resolve CÁLCULO de dano e stats (bcwq-bcww = AttackDamage/Speed/MaxLife) no cliente. Métodos GetMonsterStats/CalculateDamage são client-side com Anti-Cheat apenas em itens, não em combate. Sem evidência de server re-calc ou checksum validation do fluxo de dano.
- **verificar:** 1) Decompile/hook TaskbarHero.Combat.*.CalculateDamage + ApplyDamage → confirmar cliente computa dano sozinho; 2) Network sniff POST para /api/damage ou /api/combat → ver se servidor aceita valores de dano brutos do cliente sem re-validação; 3) Memory (Cheat Engine) modificar MaxLife de monstro vivo → se HP renderiza como modificado client-side, vuln confirmada; 4) Enviar HTTP PATCH fake damage payload → servidor aceita sem sanity check.
- **correção:** 1) Server recalcula dano completo (não confie em números do cliente); 2) Servidor valida stats do monstro antes de aplicar dano (GetMonsterStats deve vir de servidor, não cache local); 3) Checksum/HMAC de combate state; 4) Rate limit (max N ataques/s); 5) Mover DamageInfo.Apply + CalculateDamage para autoridade do servidor, cliente só recebe resultado final autorizado.

## [ALTA] Client-Authority em Recompensas Offline — Manipulação de Timestamps e Valores de Gold
- **classe:** `TaskbarHero.OfflineRewardResult`
- **problema:** A classe armazena valores de recompensa (Gold, HeroResults) com timestamps criados pelo CLIENTE (StartSteamUnixSec, EndSteamUnixSec). Um atacante edita o arquivo de save local (desencriptado em Unity por padrão) aumentando arbitrariamente os timestamps ou valores de ouro. O servidor, sem validação criptográfica, confia cegamente no OfflineRewardResult recebido.
- **verificar:** 1) Use um save editor (dnSpy/UnityEx) pra modificar Gold ou StartSteamUnixSec (ex: fingir que ficou 30 dias offline). 2) Envie pra servidor — se aceita e credita o ouro, está vulnerável. 3) Verifique se há HMAC/assinatura dos valores no cliente ou validação strict do servidor (ex: max Gold/sec = 100, fail se exceder).
- **correção:** 1) Servidor calcula período offline via LastLoginTime (do BD, imutável) — NUNCA confia em timestamps do cliente. 2) Rate-limit hard: max_gold_per_sec = config, rejeita se (Gold / tempo_real) > limite. 3) Assina OfflineRewardResult no cliente com chave privada servidor; valida signature ao receber. 4) Log anomalias (Gold > esperado, timestamps backdatados) e flag conta pra review.

## [ALTA] Rate-limit apenas client-side + Time.timeScale exploitável
- **classe:** `TaskbarHero.StageBoxRequestAbuseGuard`
- **problema:** StageBoxRequestAbuseGuard roda 100% no cliente e valida apenas contra Time.timeScale (não relógio real). Atacante eleva Time.timeScale (ex: 6×) → kills/baús reportados ~6× mais rápido ao servidor que **confia** no evento de kill do cliente sem validar taxa de relógio real → farm massivo de itens que vaza para Steam market.
- **verificar:** 1) Abrir console/mod loader (MelonLoader/HarmonyX); 2) desabilitar StageBoxRequestAbuseGuard ou elevar Time.timeScale para 6+; 3) observar baús dropando em taxa amplificada; 4) confirmar backend aceitou todos os baús sem limitar por wall-clock real.
- **correção:** Servidor-side: validar kills/baús contra relógio real, não contra tempo de jogo. Rejeitar clears fisicamente impossíveis. Tornar baú server-authoritative (derivar do progresso validado, não de eventos brutos do cliente). Remover confiança em StageBoxRequestAbuseGuard como única barreira.

## [ALTA] Client-Authority + Validação Fraca em Recompensas Offline
- **classe:** `TaskbarHero.OfflineRewardHeroResult`
- **problema:** Se OfflineRewardHeroResult é criada/serializada no cliente e aceita como-está pelo servidor, permite abuse: falsificar EndLevel > StartLevel+10, AddedExp > máximo legítimo, ou repetir mesma recompensa. Sem token de nonce ou hash, dupe é trivial.
- **verificar:** (1) O cliente cria OfflineRewardHeroResult? (2) Servidor aceita AddedExp sem validar contra regras offline legítimas (tempo offline × exp/min)? (3) Há um nonce/token único por recompensa, ou token pode ser replicado? (4) Servidor valida que EndLevel ≤ StartLevel + f(tempoOfflineEmMinutos)?
- **correção:** Servidor: (a) Nunca confie em AddedExp bruto — recalcule: exp_esperado = min(tempo_offline_segundos/TAXA_EXP_ESPERADA_SEGUNDO, TETO_OFFLINE). (b) Assine HeroKey+StartLevel+StartExp com HMAC(secret_servidor); rejeite se assinatura falha. (c) Rate-limit: mín 1h entre recompensas offline por herói. (d) Validar EndLevel ∈ [StartLevel, StartLevel+2] (limite de UPS num período). (e) One-time token: gere GUID no servidor ao registrar login offline, valide presença no clawback.

## [ALTA] Client-Authority em Items e Timestamps — Duping + Expiração Bypassável
- **classe:** `TaskbarHero.BackendPostSaveData`
- **problema:** Se BackendPostSaveData.items é aceito pelo servidor como verdade sem validação criptográfica, o cliente pode editar quantidade/tipo de items localmente (arquivo de save, memória). Igualmente, expirationTicks validado apenas no cliente permite contornar quest timeouts.
- **verificar:** 1) Capture tráfego (HTTP/WS) entre cliente e servidor ao completar quest → verifique se items vão em claro no JSON/buffer; 2) Edite save local (Unity PlayerPrefs / StreamingAssets) aumentando item.quantity; 3) Recarregue jogo e confirme se quantidade persiste no servidor; 4) Teste enviar expirationTicks=9999999999 → servidor aceita quest expirada? 5) Se CodeStage AntiCheat só valida código (não dados), o save serializado não é protegido.
- **correção:** Servidor: (a) Nunca confie em BackendPostSaveData.items direto — recalcule itemRewards baseado em QuestID + PlayerID + timestamp assinado; (b) Valide expirationTicks com horário servidor (não cliente); (c) Assine/encripte payload com HMAC (chave servidor-only); (d) Rate-limit POST de save por player por minuto; (e) Log audição: {playerId, questId, deltaItems, timestamp, source} — flag anomalias (50+ items 1 segundo).

## [ALTA] Múltiplas vulnerabilidades de client-authority em AccountSaveData
- **classe:** `TaskbarHero.AccountSaveData`
- **problema:** Campos como 'playTime', 'sessionCounter' e 'forceApplyServerData' são editáveis no cliente. Se o servidor confia neles pra gating de rewards/progressão/dupes, é crítico. 'lastExitSteamUnixSecKey' no cliente = falsa segurança (chave de criptografia nunca deve estar no cliente).
- **verificar:** (1) Buscar no servidor: valida playTime/sessionCounter/forceApplyServerData contra steam achievements ou rate-limit? (2) Procurar por 'AccountSaveData' sendo deserializado sem validação. (3) Ver se há dupe-protection baseado em sessionCounter no servidor.
- **correção:** Todos os valores-chave (playTime, rewards, progressão, sessionCounter) devem ser calculados/validados NO SERVIDOR apenas. Cliente pode ter cópia local pra UI, mas servidor é source-of-truth. Remover 'forceApplyServerData' ou implementar rate-limit/timeout no servidor. Nunca usar 'lastExitSteamUnixSecKey' — pedir Steam API diretamente.

## [ALTA] Client-Authority em Sincronização de Items — Dupe e Edição de Counts
- **classe:** `TaskbarHero.BackendPostItemSaveData`
- **problema:** A classe envia `itemID`, `itemName` e `count` do cliente pro servidor sem validação de integridade. Um atacante pode editar o arquivo de save localmente, usar Cheat Engine pra manipular memória, ou interceptar a requisição pra aumentar `count`, duplicar items com IDs inválidos, ou injetar nomes malformados — o servidor confia cegamente nesses dados.
- **verificar:** 1) Edite o arquivo de save local ou use memory editor (Cheat Engine) pra aumentar `count` de um item (ex.: 1 → 999); 2) Intercepte a requisição com Burp/Fiddler e altere o `count` antes de enviar; 3) Envie 2 requisições simultâneas com o mesmo itemID e `count` baixo — se ambas processarem, há race condition + dupe. 4) Teste com `itemID` inválido/inexistente — servidor aceita?
- **correção:** **Servidor** deve: a) Validar que o jogador realmente possui o item com `count` declarado (cross-check com BD); b) Se `count` é offline-acumulável (idle mechanic), calcular incremento máximo permitido via (wall_time - last_sync) × rate_cap, rejeitar qualquer `count` > expectativa; c) Usar HMAC/assinatura dos dados do cliente (timestamp + itemID + count) pra detecção de tampering; d) Implementar rate-limit no servidor (não no cliente); e) Tratar `itemName` como não-confiável (validar contra whitelist de items válidos, nunca usar em queries diretas).

## [ALTA] Client-Authority em Progression e Playtime — Exploração Trivial
- **classe:** `TaskbarHero.CommonSaveData`
- **problema:** Campos críticos (playTime, maxCompletedStage, currentStageWave, ArrangedPetKey, arrangedHeroKey) armazenados no cliente sem validação server-side. Jogador edita save local ou memória → server aceita como verdadeiro → progressão/rewards fake e trading de itens ilícitos.
- **verificar:** 1) Edite CommonSaveData.playTime +1000h em um save exportado; 2) Defina maxCompletedStage para stage final sem jogo; 3) Adicione um ArrangedPetKey nunca desbloqueado; 4) Reloade no jogo → servidor aceita tudo?
- **correção:** 

## [ALTA] Client-Authority na Reivindicação de Items (Item Duplication)
- **classe:** `TaskbarHero.MailSaveData`
- **problema:** A classe persiste o estado de reivindicação (`isClaimedItem`) localmente no cliente. Um atacante pode editar o save, duplicar o email, ou ressinalar como não-reivindicado, e o servidor não consegue detectar múltiplas reivindicações do mesmo `itemInstanceId` se confiar apenas no estado do cliente.
- **verificar:** 1) Obtenha um email com item valioso. 2) Edite o save (JSON/XML) ou memória alterando `isClaimedItem=false`. 3) Envie múltiplos requests de 'claim' do mesmo `itemInstanceId`. Se receber o item mais de uma vez, confirmada a dupe.
- **correção:** Servidor DEVE manter registro independente: para cada `itemInstanceId`, gravar QUANDO foi reivindicado (timestamp, playerID). Antes de processar claim, verificar se `itemInstanceId` já está no log de reivindicações. Rejeitar reivindicações duplicadas com erro. Nunca confiar em `isClaimedItem` do cliente.

## [ALTA] Client Authority Over Complete Game State
- **classe:** `TaskbarHero.PlayerSaveData`
- **problema:** PlayerSaveData persiste economia, inventário, heróis, trading, runas e progressão no cliente. Se o servidor sincroniza esses dados sem validação rigorosa server-side, um jogador pode editar o save para duplicar itens, inflacionar moeda, ou falsificar progressão.
- **verificar:** 1) Intercepte requisição de save/sync ao servidor; 2) Modifique currenySaveDatas ou inventorySaveDatas; 3) Verifique se o servidor aceita sem re-calcular estado derivado (ex: levels, stats); 4) Teste remakeTradingStashSaveDatas trocando ID de item por outro falsificado; 5) Veja se há rate-limit ou validação de origem do item no servidor.
- **correção:** 

## [ALTA] Client-Authority em Automação de Gameplay (RepeatStage, ForceSynthesis)
- **classe:** `TaskbarHero.SettingSaveData`
- **problema:** Campos como StageRepeatStageFailed, RepeatActBossDuringConsumeAllStone e ForceSynthesisOnCubeOpen são persistidos no cliente e podem ser usados para automatizar ações de gameplay. Se o servidor confia nesses flags para autorizar repetições/sínteses sem re-validar estado (recursos consumidos, tentativas disponíveis, cooldowns), permite dupe de tentativas ou farming sem consumo de recursos.
- **verificar:** 1) Salvar game, editar SettingSaveData (JSON/binary) para forçar true em RepeatActBossDuringConsumeAllStone; 2) Abrir cubo de boss; 3) Verificar se o servidor autoriza repetição automática sem validar se stones foram realmente consumidos; 4) Monitorar requisições POST de 'repeat action' — se levam esses flags como autoridade.
- **correção:** Servidor deve: (a) IGNORAR esses flags ao processar ações — lógica de repetição deve ser server-authoritative baseada em estado validado (recursos, cooldowns, tentativas); (b) Se cliente envia 'repeat request', revalidar: recursos suficientes? Cooldown expirado? Tentativas restantes? Nunca confiar em flag do cliente.

## [ALTA] Client-Authority em Encantamento — Forjamento de Ingredientes
- **classe:** `EnchantIngredients`
- **problema:** Classe é data-carrier puro (0 métodos). Campos como IsValid, MaterialItemSource, IDs de item e índices são totalmente controláveis pelo cliente. Permite duping de itens, bypassar checagens de invetário e forjar encantamentos sem consumir materiais reais.
- **verificar:** 1. Usar Melon Loader pra editar IsValid=true sem ter material válido + triggerar encantamento. 2. Enviar EnchantIngredients com MaterialSlotIndex=-1 ou índice fora do range. 3. Falsificar EnchantTargetUniqueId de item que não existe. Se encantamento processa, é vulnerabilidade confirmada.
- **correção:** No servidor: (1) Re-validar TODOS os campos contra estado real do jogador (inventário, cubo, gear). (2) Verificar que Material existe no inventário do jogador. (3) Consumir o material ANTES de enviar confirmação ao cliente. (4) Validar MaterialItemSource contra log/tipo real do item. (5) Rate-limit encantamentos por jogador (risco de auto-click bots).

## [ALTA] Client-side box authority + BoxType tampering
- **classe:** `BoxData`
- **problema:** Se servidor confia em BoxData do cliente (tipo, propriedade, estado), atacante pode editar savefile/memória para alterar BoxType (comum→raro), reutilizar UniqueId, ou abrir caixas de outros players.
- **verificar:** Verificar se BoxData é deserializado do cliente sem re-validação; tentar editar savefile com Cheat Engine para mudar BoxType; tentareenviar mesma BoxData múltiplas vezes; verificar se CodeStage bloqueia.
- **correção:** Implementar server-authoritative boxes: (1) servidor gera BoxType baseado em player progression, (2) armazena em BD, (3) valida posse antes de abrir, (4) consome UMA VEZ apenas, (5) envia resultado ao cliente.

## [ALTA] Mail/Correio: Client-side Claim sem re-validação server-side
- **classe:** `TaskbarHero.SerializedMailData`
- **problema:** ClaimItem() e AddMailEntry resolvem-se no cliente; um jogador pode reivindicar itens múltiplas vezes ou fabricar entradas de correio (premium/raros) editando MailSaveData.isClaimedItem=false e sincronizando com backend. Sem checksum/assinatura do servidor, a integridade é contornável.
- **verificar:** Enviar requisição de claimItem 2× para mesmo email; interceptar InventoryReceivePostItemAllRequest e reinjec tar itemInstanceId falsificado; editar save local adicionando ItemKey+wasMarketItem=true (Steam dupe). Se aceitar, vulnerável.
- **correção:** Server-side: Manter registro imutável de claims já processados. Ao receber ClaimItem, validar contra log. Adicionar transaction ID + timestamp + hash do mail. Rejeitar claims duplicados mesmo que o cliente envie isClaimedItem=false. Não confiar em MailSaveData.isClaimedItem; usar backend como autoridade.

## [ALTA] Client-Authority em Adição de Item — Confiança Não-Validada
- **classe:** `TaskbarHero.AddItemResult`
- **problema:** Uma classe que encapsula ItemType + status de operação, usada em contexto onde o cliente resolve muita coisa, é red flag clássica: se o servidor confia no AddResult sem re-validar, permite dupe, injeção de ItemType arbitrário e manipulação de savefiles editáveis.
- **verificar:** 1) Procure onde AddItemResult é serializado/enviado ao servidor (RPC, HTTP, JSON). 2) Confirme se o servidor re-valida ItemType contra regras de game antes de persistir. 3) Teste: edite o savefile localmente ou intercepte a rede, mude ItemType para algo valioso (ex: Legendary Weapon id=999), veja se servidor aceita.
- **correção:** Servidor NUNCA confia em AddItemResult do cliente. Em vez disso: (a) Cliente solicita 'AddItem(slot, targetType)' SEM enviar resultado. (b) Servidor valida: regras de game, posse, inventory space, cooldowns. (c) Se válido, servidor cria o item e notifica cliente. (d) Salva no backend (não no cliente). (e) Código AntiCheat: monitore para ItemType_ids fora do catálogo.

## [ALTA] Client-Authority na Geração de Ouro (Enum como Claim de Origem)
- **classe:** `TaskbarHero.EGoldCurrencySource`
- **problema:** O enum EGoldCurrencySource existe porque o cliente está comunicando AO SERVIDOR a 'fonte' de cada ganho de ouro. Isso permite que um attacker modifique o cliente (ou intercepção de rede) para fingir que ouro foi ganhado por OfflineReward (calculado 100% cliente-side) ou CubeAlchemy, sem que o servidor valide se a origem é legítima. Resultado: duplication ilimitado de ouro → economia quebrada → RMT lucro.
- **verificar:** 1. Capturar request de ouro ao server (ex: após kill de mob) — observe se contém {gold: X, source: 'MonsterKill'}. 2. Modificar localmente client para enviar {gold: 999999, source: 'OfflineReward'}. 3. Se server aceita sem verificação de offline-time real ou limite razoável, vuln confirmada. 4. Revisar se servidor realmente valida que offline rewards foram GANHADOS (checando login history + tempo).
- **correção:** 

## [ALTA] Client-Authority Vulnerability: Temporal Manipulation & Item Tampering
- **classe:** `TaskbarHero.BackendPostData`
- **problema:** Campos expirationTime/reservationTime/inDate armazenados no cliente sem re-validação server-side permitem bypass de cooldowns e limitações temporais. Items associados sem assinatura/hash são editáveis e duplicáveis via Cheat Engine. O padrão 'BackendPostData' sugere dados do servidor processados localmente como autoridade.
- **verificar:** 1) Editar expirationTime com Cheat Engine (buscar valor de tempo epoch); verificar se evento permanece ativo após expiração. 2) Duplicar item no cache local; sincronizar com servidor — servidor aceita o dobro? 3) Enviar POST fabricado com items count alterado; servidor atualiza sem validar.
- **correção:** 1) Server valida expirationTime a cada request (nunca confia no cliente). 2) Items assinados com HMAC(serverKey + itemData) — cliente não pode modificar sem chave. 3) Authoritative source: servidor recalcula estado completo, não sincroniza bidirecional com cache local.

## [ALTA] Client-side item authority — count tamperável
- **classe:** `TaskbarHero.BackendPostItem`
- **problema:** BackendPostItemSaveData envia count diretamente do cliente ao servidor sem validação server-side. Atacante pode editar a requisição POST (aumentar count) e duplicar itens indefinidamente.
- **verificar:** Modificar count em um BackendPostItem via devtools/interceptador de rede antes de enviar ao servidor. Se servidor aceita sem validar contra estado interno, há dupe de item.
- **correção:** Servidor NUNCA deve aceitar valores absolutos (count=999) do cliente. Aceitar apenas deltas (delta=+10) e validar contra state-of-truth server. Revalidar itemID contra inventário server; rejeitar se discrepância.

## [ALTA] Client-Authority em InventoryAddItemRequest — Adição não validada de itens
- **classe:** `TaskbarHero.InventoryAddItemRequest`
- **problema:** O campo itemId é enviado sem ofuscação ao servidor (apenas plain-text). Validação ocorre só no cliente via Validate(), que é trivialmente contornável (memory hack, packet edit). Servidor aparentemente confia no itemId recebido, permitindo duplicação de items e adição de items arbitrários.
- **verificar:** 1) Interceptar requisição InventoryAddItemRequest via proxy (Fiddler/Burp): itemId vem como plain int? 2) Enviá-la múltiplas vezes: item duplica no inventário? 3) Editar itemId para um ID inválido/premium: servidor aceita? 4) Memory hack (Cheat Engine): modificar itemId antes do ToJson e enviá-lo. 5) Verificar se server faz CopyFromObscured ou ValidateChecksum (comparar com BoxCreateItem — se não há, é client-authority).
- **correção:** Server-side: (1) Nunca confiar em itemId do cliente — validar contra tabela de items autorizados + verificar se jogador tem privilégio. (2) Implementar rate-limit por jugador/item. (3) Usar HMAC/checksum server-side: gerar hash(itemId + timestamp + playerId + secret) no servidor, exigir que cliente reenvie. (4) Desduplicar: idempotency key ou checar se requisição já foi processada (dedupe window 5-10s). (5) Logging de todas as adições pra auditoria posterior. (6) Usar obfuscação bidirecional como BoxCreateItem faz (CopyToObscured/CopyFromObscured + ValidateChecksum).

## [ALTA] Client Authority em Recebimento de Item — Potencial Dupe/Tampering
- **classe:** `TaskbarHero.InventoryReceivePostItemRequest`
- **problema:** A classe envia `postType` e `postIndate` ao servidor, sugerindo que o cliente controla atributos do item recebido. O padrão o_postType/postType indica tentativa de rastrear divergência, mas se o servidor aceita valores do cliente sem re-validar contra BD (ID único do post + propriedade + idempotência), atacante pode duplicar itens ou reclamar valores falsos via Cheat Engine/interceptação.
- **verificar:** 1) Burp Suite: capturar POST legítimo de claim item 2) Modificar postType para item raro/valioso 3) Repetir mesma requisição 2× com mesmo ID 4) Verificar se servidor aceitou e duplicou no inventário
- **correção:** Servidor NUNCA deve confiar em postType/postIndate do cliente. Implementar: (A) carregar post via ID único da BD, (B) re-validar tipo/valor contra item catalog, (C) validar propriedade + status 'unclaimed' antes de transferir, (D) usar transação atômica com duplicate check (idempotency key), (E) rate-limit server-side, (F) log audit de todas as transações

## [ALTA] Servidor confia em evento de recebimento de post do cliente sem validação de autoridade
- **classe:** `TaskbarHero.InventoryReceivePostItem`
- **problema:** A classe roda no cliente (Assembly-CSharp) e envia requisições com postType e postIndate para o servidor. Não há evidência de que o servidor valida se o post realmente existe, pertence ao jogador, ou está elegível para reclamação — confia cegamente no evento do cliente para conceder itens.
- **verificar:** Monitorar POST /inventory/receive-post-item na rede. Enviar postIndate futuro/passado arbitrário, postType falso, ou mesmo ID múltiplas vezes. Se servidor retorna sucesso/aprova sem correlacionar com registro real no banco, a vuln é confirmada.
- **correção:** Servidor DEVE: (1) validar que postId existe na tabela de correio/posts; (2) verificar que está no status não-reclamado ('unclaimed'); (3) verificar que postIndate cai dentro da janela de validade real do servidor; (4) marcar como reclamado ATOMICAMENTE (previne dupe); (5) rejeitar se cliente já reclamou esse post no histórico.

## [ALTA] Client Authority DLC Verification — Bypass de Paywall
- **classe:** `TaskbarHero.VerifyDlc`
- **problema:** Classe VerifyDlc roda no cliente ('sistema: save') e retorna bool de desbloqueio sem revalidação server-side a cada acesso. Atacante pode modificar Assembly-CSharp.dll (ILSpy/dnSpy) ou alterar save file para fazer GetResultData() retornar isSuccess=true, ganhando acesso a conteúdo DLC-locked sem pagamento.
- **verificar:** 1) Interceptar chamada GetResultData() ao acessar conteúdo premium (usar Frida ou debugger). 2) Verificar se servidor revalida propriedade de DLC contra Steam API a cada ação sensível. 3) Tentar editar AccountSaveData.ownedDlc em disco e recarregar — se funciona sem erro server-side, vulnerável.
- **correção:** 

## [ALTA] Falta de autenticação por player + sincronização de inventário sem idempotência
- **classe:** `TaskbarHero.InventoryReceivePostItemAllRequest`
- **problema:** Requisição sincroniza TODOS os itens do inventário usando apenas o_postType/postType sem mencionar validação de qual player está solicitando (token/session). Sem autenticação server-side, atacante pode pedir inventário alheio. Sincronização de snapshot completo sem idempotência = risco de duplication em retry.
- **verificar:** 1) Não há token/playerID validado na requisição? 2) Servidor aplica itens sem checar autenticação? 3) Enviar requisição duplicada = itens duplicam? 4) Pode-se mudar postType pra 'load' arbitrariamente?
- **correção:** Server: (1) Validar sessão/token do player antes de sincronizar. (2) Usar request ID + deduplicação server-side (idempotência). (3) Validar integridade de o_postType vs postType (não apenas existência). (4) Rate-limit por player_id, não só IP.

## [ALTA] Client-Authority em Operações de Crafting/Exchange
- **classe:** `TaskbarHero.InventoryExchangeResult`
- **problema:** A presença de campos 'removed' e 'added' numa classe C# (client-side) de resultado sugere que o cliente está gerando/reportando modificações de inventário. Se o servidor confiar nestes valores sem revalidar a receita, quantidades e ownership no BD, permite dupe e item manipulation.
- **verificar:** 1) Verificar se InventoryExchangeResult é serializado/enviado DO CLIENTE pro servidor (RPC, SendData, JSON POST); 2) Confirmar se servidor revalida receita + itens no BD ou apenas acredita em 'removed'/'added'; 3) Buscar onde InventoryExchangeResult.added/removed é processado server-side.
- **correção:** Servidor DEVE: (a) receber SÓ recipeId + opcionalmente slot IDs; (b) buscar receita do BD; (c) validar que player tem items.removed no inventário com quantidade >= receita.cost; (d) processar changes atomicamente no BD; (e) enviar resposta com SÓ {success, newInventoryState} gerado pelo servidor, NUNCA ecoar valores do cliente.

## [ALTA] Client-Authority Item Consumption — Arbitrary Item Removal + Duplication
- **classe:** `TaskbarHero.InventoryConsumeResult`
- **problema:** A classe `InventoryConsumeResult` permite ao cliente reportar qual item foi consumido (via campos `removed`, `invalidUniqueId`). Se o servidor confia nesta lista para atualizar inventário sem re-validação server-side, atacante pode editar memory/traffic para remover items arbitrários ou falsificar sucesso sem consumir nada.
- **verificar:** 1) Interceptar requisição de consumo (Fiddler/Burp); 2) Editar `removed[]` para incluir items valiosos não originais; 3) Ou enviar `isSuccess=true` sem realmente consumir; 4) Se servidor aceita e remove items do DB, é vulnerável. Compare inventário antes/depois.
- **correção:** Servidor RECLALCULA consumo inteiro server-side: validar recipe existe, validar client TEM items requeridos (SELECT... FOR UPDATE), remover items no DB, retornar novo inventário ao cliente (server-authoritative). Nunca confiar em `removed[]` do cliente — usar apenas como confirmação visual de UI.

## [ALTA] Client-Authority + Falta de Server-Side Validation em InventoryExchange
- **classe:** `TaskbarHero.InventoryExchange`
- **problema:** Método Execute() aparentemente roda no cliente sem validação servidor. Contraste com InventoryConsumeRequest (que tem o_itemKey ofuscado) sugere InventoryExchange é menos protegido. Permite duplication via race condition (dois Execute → um Remove) ou bypass direto na API sem re-validação.
- **verificar:** 1) Ligar proxy (Fiddler/mitmproxy), interceptar POST /exchange. 2) Modificar itemID ou quantidades. Servidor aceita? 3) Chamar Execute 2× rapidamente (mesmo itemID). Items duplicados? 4) Reverter inventário localmente (edit JSON save), sincronizar. Servidor aceita estado inválido?
- **correção:** Server-side: (1) RE-VALIDAR recepção+requisitos no backend ANTES de qualquer remove(). (2) Transação atômica: remove DENTRO da mesma transação que add (não sequencial). (3) Nonce/deduplication token (tipo o_itemKey em Consume). (4) Rate limit por player por segundo. (5) Assinatura HMAC da requisição (cliente assina, servidor verifica com chave privada). (6) Audit log de todas as trocas.

## [ALTA] Client-Authority: Servidor confia em resultado de processamento de caixa do cliente
- **classe:** `TaskbarHero.InventoryProcessBoxResult`
- **problema:** A classe InventoryProcessBoxResult encapsula um resultado calculado e validado PELO CLIENTE. Se o servidor aplicar items/boxes sem re-validar (verificar se a caixa existe, se ainda não foi processada, se player tem direito), atacante consegue dupar items ou processar caixas múltiplas vezes via memory editor.
- **verificar:** Interceptar tráfego: chamar API com InventoryProcessBoxResult falsificado (isSuccess=true, added=[items_valiosos], boxes=[ids_caixas]). Se servidor aplica sem validar, vulnerável. Teste 2: enviar mesmo boxes ID 2x — dupa garantida se não há validação server-side.
- **correção:** Servidor deve processar caixa (gerar items, decrementar boxes) e NUNCA confiar no resultado do cliente. Fluxo correto: (1) cliente envia 'abrir caixa ID X', (2) servidor valida (existe? já aberta? inventory space?), (3) servidor gera resultado, (4) servidor envia resultado pro cliente (não o contrário).

## [ALTA] Client-Authority em Item Drops + Ofuscação Inefetiva
- **classe:** `TaskbarHero.InventoryProcessBoxRequest`
- **problema:** Cliente determina QUANDO/QUAIS itens aparecem no baú (o_getitemkey, createItemList) e servidor confia cegamente. Ofuscação CodeStage não é validação — ambos campos públicos e ofuscados viajam na rede sem assinatura criptográfica. Flag isReset permite cliente resetar estado de recompensa já coletada.
- **verificar:** 1. Interceptar requisição InventoryProcessBoxRequest com Fiddler/mitmproxy; 2. Editar getitemkey para item raro (ex: lendário que não dropou) ou quantidade (createItemList); 3. Server aceita → vulnerável. 4. Testar enviar isReset=true pra caixa já aberta → confirmará reset client-side.
- **correção:** 1. **Servidor gera itens**: Drop determinado server-side (RNG seed + nível do mob + tabela de loot) enviado APENAS na resposta, nunca confia campos do cliente. 2. **Assinatura HMAC**: Cada InventoryProcessBoxRequest precisa de timestamp + HMAC(timestamp+userId+requestId, chave_server_secreta). 3. **Nonce + expiry**: Incluir nonce único por requisição + TTL <5s pra evitar replay. 4. **isReset bloqueado**: Servidor controla se baú foi coletado (banco de dados, chave: userId+boxId), ignora flag client. 5. **Rate limit**: Max 1 req/s por usuário em endpoints de loot.

## [ALTA] Client-Authority na Serialização de Items para Steam
- **classe:** `TaskbarHero.UploadToSteamData`
- **problema:** A classe serializa dados de item (ID, quantidade) controlados pelo cliente e os envia ao servidor para sincronização Steam. Se o servidor não valida posse real do item, quantidade em estoque, e ownership antes de aceitar, permite dupe de items, inflação de economia e bypass de trade rules.
- **verificar:** 1) Enviar UploadToSteamData com qty maior que inventário real (ex: item com qty=1, enviar qty=999). 2) Verificar se server retorna erro 'insufficient items' ou aceita silenciosamente. 3) Confirmar se qty synca com Steam sem validação de source-of-truth (DB do servidor, não o client).
- **correção:** No servidor: (a) Validar cada UploadToSteamData contra inventário authoritative do DB antes de aceitar. (b) Comparar o_quantity vs qty — rejeitar se delta é impossível (ex: qty cresceu sem concessão de loot). (c) Rate-limit no servidor, não cliente. (d) Log todas as tentativas de desync. (e) Steam sync só após server confirmar integridade.

## [ALTA] Client-Authority + Weak Obfuscation = Item Duplication & Fraud
- **classe:** `TaskbarHero.InventoryProcessBox`
- **problema:** InventoryProcessBox processa transformações (crafting/alchemy) no cliente com validação fraca baseada em ofuscação XOR. Servidor confia em campos o_useItemKeyList criados pelo cliente sem re-validar independentemente se ingredientes existem e se receita é legítima. Permite: (1) Falsificar consumo de ingredientes; (2) Criar items sem gastar recursos; (3) Bypass rate-limiting (apenas client-side); (4) Dupe de items raros via CheatEngine + save local tampering.
- **verificar:** 1. Injetar InventoryProcessBoxRequest com o_useItemKeyList vazio mas resultado válido → servidor aceita? 2. Usar CheatEngine para editar quantidades no save local → persiste? 3. Chamar Process() múltiplas vezes via memória injection → rate-limit é apenas cliente? 4. Monitorar tráfego: se o_itemKey é XOR simples, reverter a chave com 2-3 requisições.
- **correção:** 

## [ALTA] Client-Authoritative Inventory Consumption + Timing Bypass
- **classe:** `TaskbarHero.InventoryConsume`
- **problema:** ValidateInventory/ApplyEffect/RemoveFromInventory executam client-side sem server-side re-validation. Atacante pode: (1) consumir item sem quantidade válida via mod loader (bypassa CodeStage), (2) ignorar cooldown de consumo, (3) duplicar efeitos via replay/resend de InventoryConsumeRequest.
- **verificar:** Interceptar InventoryConsumeRequest e reenviar múltiplas vezes; validar se servidor valida (i) existência do item, (ii) quantidade disponível, (iii) cooldown absoluto, (iv) taxa de consumo contra wall-clock. Verificar se erro de validação reverte o ApplyEffect client-side.
- **correção:** Server-authoritative: servidor MUST re-validate item existence/quantity/cooldown ANTES de aceitar requisição. Rejeitar requests duplicadas via nonce/timestamp. Não confiar em campos ofuscados do cliente (use como detecção apenas); implementar rate-limit de consumo por wallet-time real. Todos os efeitos (heal/buff) aplicados APENAS após confirmação server.

## [ALTA] Client-Authority em Sincronização Steam + Validação Client-Side
- **classe:** `TaskbarHero.InventoryUploadToSteam`
- **problema:** InventoryUploadToSteam resolve validação (ValidateInventoryData) APENAS no cliente antes de serializar/enviar. Cliente também controla estado do inventário (GetInventoryState de snapshot local) e copia campos 'ofuscados' (o_steamId, o_itemKey) antes do envio — ofuscação não é criptografia. Sem HMAC/assinatura, servidor não consegue detectar se itemKey, slot ou quantidade foram alterados mid-transmission ou em cache local.
- **verificar:** 1) Interceptar requisição InventoryUploadToSteamRequest.ToJson() via proxy (mitmproxy/Burp) e alterar itemKey/quantidade; 2) Verificar se servidor REVALIDA integridade dos dados (deve rejeitar ou aplicar sem verificação?); 3) Verificar CacheInventoryData — arquivo de cache local é chave:valor legível?; 4) Testar se CopyToObscured pode ser revertida (desofuscação)
- **correção:** 1) Servidor deve revalidar TUDO (itemKey, slot, steam_id, quantidade) contra estado autorizado do servidor; 2) Usar HMAC(data, server_secret) assinado no cliente, verificado no servidor; 3) Nunca confiar em ValidateInventoryData do cliente; 4) Replay protection (nonce/timestamp com TTL curto); 5) Rate-limit por steam_id no servidor, não no cliente

## [ALTA] Client-Authority em Validação de Propriedade de Item/SteamID
- **classe:** `TaskbarHero.InventoryUploadToSteamRequest`
- **problema:** A estrutura de campos (o_steamId/steamId, o_itemKey/itemKey) sugere validação de mudança no cliente sem autorização server-side. Se o servidor não verifica que o SteamID autenticado == SteamID na requisição, atacante pode transferir/duplicar itens de outro jogador ou sua própria conta multiplamente.
- **verificar:** 1) Envie requisição com outro SteamID que não é seu — se aceitar, é CRÍTICO. 2) Modifique itemKey para item inexistente/já vendido — se aceitar, é duping. 3) Rapidamente envie 100+ requisições idênticas — se aceita sem rate-limit server, economia quebra.
- **correção:** No servidor: (1) Validar que SteamID == Player.AuthenticatedSteamID (via session token, não client param). (2) Verificar em BD que Item realmente existe, pertence ao player, não foi vendido. (3) Usar transação atômica com lock. (4) Rate-limit por SteamID (5 req/min), no backend, não cliente. (5) Log auditoria de cada transfer. TheBackend BaaS deve ter validação de negócio, AntiCheat NÃO valida client-authority.

## [ALTA] Client-Authority Inventory Sync — Dupe via Falsified Upload State
- **classe:** `TaskbarHero.InventoryUploadToSteamResult`
- **problema:** Classe armazena resultado de sync no cliente sem server-side revalidação. Atacante com acesso à memória/save pode falsificar isSuccess=true, manipular o_failedItemIds para simular falha parcial, ou modificar 'removed'/'uploaded' para enganar lógica de retry. Resultado: itens duplicados em Steam trading ou inventário inconsistente.
- **verificar:** 1) Interceptar tráfego InventoryUploadToSteamResult após sync bem-sucedido. 2) Modificar isSuccess=false em save/memória, forçar retry. 3) Se servidor aceita retry como operação nova (não idempotente), items duplicam. 4) Verificar se servidor valida que itens já estão em Steam antes de aceitar 'removed' como true.
- **correção:** Server-side: (A) Manter fonte-da-verdade do inventário no servidor, REVALIDAR contra Steam API após cada sync (não confiar em o_failedItemIds do cliente). (B) Implementar idempotência com upload-token único (evita retry de mesma operação). (C) Rate-limit sync por jogador no servidor (não cliente). (D) Assinar resultado com HMAC se armazenar no cliente (CodeStage não previne tampering em memória). (E) Log server-side de discrepâncias (cliente diz 'falhou', servidor vê 'sucesso').

## [ALTA] Client-Authority em Sincronização de Inventário Steam
- **classe:** `TaskbarHero.InventoryDownloadToSteamResult`
- **problema:** A classe encapsula listas de items 'adicionados' e 'baixados' reportadas pelo CLIENTE. Se o servidor confiar nessas listas sem validar contra Steam API, permite duplication, fabricação de items de alto valor, ou bypass de contas Steam legítimas.
- **verificar:** Interceptar request de InventoryDownloadToSteamResult (Fiddler/mitmproxy); injetar item fake na lista 'added'; confirmar se servidor o persiste sem validar contra Steam API.
- **correção:** Server DEVE: (1) Chamar Steam WebAPI (/IInventoryService/GetInventory) usando Steam AppID/user token; (2) Comparar items reportados pelo cliente contra a resposta de Steam; (3) Aceitar apenas items que Steam confirma. Nunca confiar em listas do cliente em transações de valor.

## [ALTA] Client-Authority em Inventory + Potencial Duping de Items
- **classe:** `TaskbarHero.DownloadToSteamData`
- **problema:** Estrutura com dual representation (legível + ofuscada) em cliente sugere que dados Steam são cacheados e sincronizados como autoridade. Ofuscação é apenas segurança por obscuridade. Se servidor confiar em quantity/steamItemId do cliente sem validação real-time contra Steam API, permite: (1) edição de quantidade via memory editor (Cheat Engine), (2) falsificação de origem do item (contorna rate-limits), (3) claim de items inexistentes no inventário Steam real.
- **verificar:** 1) Usar Cheat Engine pra editar `quantity` em processo Unity rodando (procurar 4-byte values); 2) Sincronizar inventory alterado; 3) Verificar se servidor aceitou sem validar contra Steam API; 4) Testar trading de item com quantity falsificada; 5) Verificar logs do servidor — qual é a autoridade real?
- **correção:** SERVER-SIDE APENAS: (1) Nunca aceitar quantity/steamItemId do cliente — chamar Steam API sempre que inventário importa (trade, shop, quest reward); (2) Manter inventário autoritative no servidor, sincronizar APENAS quando Steam API confirmar; (3) Validar 'from' contra histórico de eventos do jogador (se diz 'shop', verificar se jogador comprou); (4) Rate-limit SERVIDOR, não cliente; (5) Log tudo (origem, antes/depois, código do cliente anticheat); (6) Remover ofuscação trivial—dados sensíveis já têm TLS.

## [ALTA] Encapsulamento de dados inicializáveis em evento client-side sem validação server
- **classe:** `TaskbarHero.InventoryInitResult`
- **problema:** Um InventoryInitResult que encapsula itens, marketData, e DLC slots é tipicamente enviado do servidor ao cliente durante init. Se o cliente puder RE-ENVIAR este resultado modificado (ou se tiver acesso ao construtor), pode injetar itens fictícios, contrafazer preços ou ativar DLC slots sem pagamento.
- **verificar:** 1) Ver se há serialização/desserialização de InventoryInitResult no cliente. 2) Verificar se o servidor re-valida items.count, item IDs, e dlc.activated antes de usar esses dados em transações. 3) Procurar por calls que confiam diretamente em dlc[].isActivated sem re-fetch do servidor.
- **correção:** Nunca confie em dados de init do cliente para autorização de itens/DLC. Servidor deve re-validar TUDO (inventário, ownership, activações DLC) em cada operação (trade, venda, uso de item). Dlc.isActivated deve ser fonte de verdade no servidor, não no cliente.

## [ALTA] Client-Authority Inventory Tampering / Item Duplication
- **classe:** `TaskbarHero.InventoryItemData`
- **problema:** Armazenar dados de item no cliente com duplicação (versão legível + ofuscada) permite reverse-engineer e edição de ambas. Se servidor confiar em itemKey/itemId do cliente sem validar contra banco de dados, atacante consegue criar/duplicar itens raros e quebrar Steam trading.
- **verificar:** 1) CheatEngine: pausar jogo, editar itemId na RAM, resumir e confirmar se servidor aceita; 2) Modular save file de inventário; 3) Tentar fazer trade de item editado; 4) Logs servidor: procurar por validação de itemId contra DB do jogador.
- **correção:** Servidor é autoridade única. Cliente envia APENAS ações ('use X', 'drop Y', 'trade Z'); servidor valida contra DB, aplica efeito, retorna novo estado assinado. Nunca confiar em IDs/valores do cliente. Se inventário persistir no cliente, usar MAC criptográfico (HMAC-SHA256 com key no servidor), não ofuscação.

## [ALTA] Client-Side Inventory State Persistence Without Server Validation
- **classe:** `TaskbarHero.InventoryProcessPendingResult`
- **problema:** O método Apply() persiste mudanças de inventário no cliente baseado em InventoryProcessPendingResult recebido de operação pendente, sem validação cruzada do servidor. Um atacante pode tamper a resposta (isSuccess=true, items adicionados) ou interceptar a chamada para Apply() aplicando mudanças não-autorizadas ao inventário local.
- **verificar:** 1) Interceptar resposta HTTP de operação pendente (ex.: TheBackend API) e mudar isSuccess para true com itens fictícios. 2) Verificar se o cliente chamará Apply() cegamente sem re-validar com servidor. 3) Confirmar que inventário local é atualizado antes de sincronização server-side. 4) Testar dupe: fazer operação legítima, clonar response JSON, enviar 2× para ver se items duplicam.
- **correção:** 1) Servidor REJEITA qualquer mudança de inventário baseada unicamente em Apply() do cliente. 2) Servidor valida CADA operação pendente antes de confirmar (re-query o estado antes/depois). 3) Cliente usa InventoryProcessPendingResult APENAS para UI feedback (sucesso/erro visual), nunca para persistência de dados. 4) Inventário sempre re-sincronizado do servidor após operação (pull-not-push model). 5) Assinar resposta com HMAC(servidor-secret, resultado) — cliente valida antes de Apply(). 6) Implementar rate-limit SERVIDOR-SIDE em operações pendentes (não confiável rate-limit do cliente com CodeStage AntiCheat).

## [ALTA] Client-Authority em Event Tracking de Progresso
- **classe:** `TaskbarHero.ESteamSessionStatType`
- **problema:** Um enum que tipifica eventos de sessão (BossKill, ItemObtain, StageClear, etc) existindo no cliente é padrão de risco: sugere que o cliente está gerando/reportando esses eventos ao servidor. Sem validação robusta server-side, permite progression inflation, item dupe, e achievement spoofing — crítico em jogos com trading.
- **verificar:** Procurar por chamadas HTTP/WebSocket que enviem esse enum ao backend; verificar se servidor replica lógica de validação (realmente matou o boss? tem recurso pra crafting?) ou apenas aceita o evento; tentar gerar evento falso via Frida/MitmProxy; verificar logs de servidor sem validação.
- **correção:** Mover autoridade de eventos para servidor: servidor revalida TODAS as condições (estado do jogo, recursos, cooldowns, lógica de regras) antes de aceitar BossKill/ItemObtain/StageClear/Synthesis. Cliente apenas reporta ações (input do jogador), não outcomes. Rate limit por player_id. Auditar mudanças de estado contra snapshot server.

## [ALTA] Client-Side Marketplace Timestamp Bypass
- **classe:** `TaskbarHero.MarketDataEntry`
- **problema:** O campo 'marketableAfterTimestamp' reside no cliente e não é criptografado. Um atacante pode modificá-lo para vender itens antes do tempo permitido, bypassando completamente a restrição temporal.
- **verificar:** Altere marketableAfterTimestamp para um timestamp no passado e tente vender um item bloqueado temporáriamente. Se aceitar, está vulnerável.
- **correção:** Servidor deve manter timestamp imutável (banco de dados do servidor), ignorar completamente o valor do cliente, e validar contra seu próprio registro antes de qualquer transação de marketplace.

## [ALTA] Client-Reported Inventory Delete Counts Without Server Validation
- **classe:** `TaskbarHero.SteamItemDeleteResult`
- **problema:** A classe armazena contadores de deleção (totalCount, successCount, failCount) que parecem ser reportados pelo cliente ao servidor sem validação contra o estado real do backend/Steam. Permite falsificação de resultado de operação em lote (e.g., reportar 100 itens deletados quando deletou 10, ou vice-versa).
- **verificar:** Interceptar POST do cliente que envia SteamItemDeleteResult; modificar successCount/failCount para valores falsos; verificar se servidor aceita e atualiza inventário/recursos baseado nos valores do cliente em vez de validar contra estado real em TheBackend + Steam API.
- **correção:** Servidor EXECUTA a deleção de itens (não confia no cliente). Valida totalCount contra items reais no Steam/backend. Retorna resultado REAL ao cliente (não aceita resultado do cliente). Operação deve ser idempotent e atomic no servidor com logging de auditoria immutável.

## [ALTA] Client-Authority em Abertura de Caixas — Validação Servidor Ausente
- **classe:** `TaskbarHero.Log.BoxOpenLog`
- **problema:** A classe loga eventos de abertura no cliente sem validação server-side. Um atacante pode reportar falsas aberturas (posições fake, múltiplas vezes, caixas inacessíveis) se o servidor confia apenas neste log para liberar loot e rastrear estado.
- **verificar:** Confirmar: (1) Se servidor valida que há caixa em (x,y,z) e que player pode acessá-la; (2) Se servidor rastreia ownership/estado da caixa independente do log; (3) Se há rate-limit server-side ou apenas cliente; (4) Se mesmo evento duplicado é blindado contra dupe.
- **correção:** Servidor deve: validar existência/acessibilidade da caixa, rastrear estado por ID (não por posição), implementar cooldown + idempotência (ex: UUID do evento), nunca confiar apenas em timestamp/posição do cliente.

## [ALTA] Client-Authority Crafting + Dupe via Evento Não-Validado
- **classe:** `TaskbarHero.Log.CubeAlchemyLog`
- **problema:** Log de alquimia registra transformação de cubos sem validação server-side confirmando que itens de entrada (bfnc) existem no inventário. Permite forjar eventos de crafting e duplicar itens indefinidamente.
- **verificar:** 1) Capture tráfego: cliente envia LogAlchemyEvent com cubo_entrada falso (item não possuído). 2) Verifique se servidor aceita sem validar inventário. 3) Confira se item de saída é concedido apesar de entrada inválida. 4) Teste rapidez: múltiplos eventos simultâneos → multiplicação de itens.
- **correção:** Servidor deve: (a) validar que bfnc (cubo entrada) existe no inventário ANTES aceitar evento; (b) remover input item + criar output em transação atômica; (c) rate-limit crafting por player; (d) assinar eventos do cliente com HMAC pra evitar forgery; (e) audit log todas transformações de receitas raras/valiosas.

## [ALTA] Client-Authority em Lógica de Status Effects
- **classe:** `TaskbarHero.Status.StatusEffect.StatusEffectOverrideType`
- **problema:** Enum no cliente controlando override behavior sugere que cliente escolhe se buffs reseta duração ou se debuffs são ignorados. Jogador pode manipular qual tipo ocorre enviando enum customizado, permitindo dupe de buffs e contorno de debuffs.
- **verificar:** Intercept HTTP/WS: cliente envia StatusEffectOverrideType.InitDuration ao reaplicar buff? Servidor aceita sem re-validar duração? Aplicar mesmo buff 100× — duração fica permanente?
- **correção:** Servidor implementa TODAS lógicas de override server-side. Nunca confiar em tipo enviado pelo cliente. Validar duração contra regras do jogo. Rate-limit aplicação de status effects no servidor.

## [ALTA] Client-Authority em Buff Expiration (Validação Servidor-Side Fraca)
- **classe:** `TaskbarHero.Status.BuffDebuff.BuffExpireConditionType`
- **problema:** Enum BuffExpireConditionType é um contract entre cliente e servidor. Se o cliente adia/falsifica expiração (Time → clock skew, BaseAttackCount → contador local não sincronizado, Stage → mensagem fake), servidor pode confiar cegamente no estado. Jogo idle-RPG com economia Steam = atacantes altamente motivados para buff infinito / dupe.
- **verificar:** ["1. Ver se cliente ENVIA seu próprio estado de buff (msg tipo 'BuffExpired' com enum type) vs servidor recalcula sempre", "2. Procurar 'game server recebe BuffExpireConditionType' — se recebe e valida apenas se enum ∈ {Time,BaseAttackCount,Stage}, é **RED FLAG**", "3. Cheat Engine: testar editar 'attack count' local enquanto buff ativo; se buff não expira, expiração é client-side", '4. Interceptar tráfego de combate: se cliente manda timestamps de expiração (não server), VULNERÁVEL', "5. Procurar 'SetSystemTime' / clock manipulation — buffs por Time são triviais quebrar se baseados em DateTime.Now do cliente"]
- **correção:** ["**Server-side buff timer + state machine:** servidor controla **todos** os timers de buff. Cliente nunca envia 'buff expirou'", 'BaseAttackCount: servidor incrementa contador a cada ataque verificado (não trusted client), servidor expira buff quando contador ≥ threshold', "Stage: servidor notifica cliente 'stage changed' → cliente aplica expiração LOCAL (cosmético), servidor ja expirou no seu lado", 'Validação: se cliente INSISTIR enviar enum/timeout, server ignora—usa seu próprio recálculo: `expire_at = server_now + duration_ms`', 'Rate limit expiração messages no cliente (ex: max 1/s), rate limit no server também (DoS proteção)']

## [ALTA] Client-Authority: Status de Conta Validado no Cliente
- **classe:** `TaskbarHero.StatusSystem.EAccountStatusCategory`
- **problema:** Um enum de status de conta (Battle/Exploration/Reward) cria risco se o cliente envia sua categoria ao servidor e o servidor a usa para validar ações (loot drops, cooldowns, autorização de features). Em idle-RPGs, é comum o cliente mentir sobre seu 'mode' para pular validações ou inflacionar rewards.
- **verificar:** Procurar por: (1) serialização do enum sendo enviada em requests HTTP/gRPC ao servidor; (2) servidor aceitar e usar `EAccountStatusCategory` do cliente sem re-validar independentemente; (3) lógica de recompensas ou rate-limits que dependa deste valor; (4) ausência de server-side journal/audit do histórico de status.
- **correção:** Status de conta = server-of-truth apenas. Cliente recebe categoria como READ-ONLY (para UI). Todas as decisões críticas (loot drop, cooldown, feature unlock) devem ser validadas no servidor com base em histórico de ações verificáveis (kill logs, quest completion timestamp), nunca no valor do enum.

## [ALTA] Client-Authority em Account Buffs - Risco de Exploração de Economia
- **classe:** `TaskbarHero.StatusSystem.EAccountStatus`
- **problema:** O enum EAccountStatus define modificadores que afetam ouro, EXP, drop rates e stats de herói. Se aplicados no cliente sem validação rigorosa servidor-side, permite manipulation direta da economia do jogo (ouro infinito, EXP boost, item dupe via drop rate artificialmente aumentada).
- **verificar:** 1) Procure onde esses enum values são instanciados/aplicados (search 'EAccountStatus' + 'Apply' ou 'Activate'). 2) Confirme se o servidor valida que o buff foi legitimamente adquirido (não fabricado). 3) Teste interceptando um packet de aplicação de buff com Fiddler/Burp e modifique o valor (ex.: AdditionalGold) — se aceita, é exploitável. 4) Verifique se há rate-limiting servidor-side ou se um cliente pode 'aplicar' o mesmo buff múltiplas vezes.
- **correção:** Servidor NUNCA deve aplicar um buff baseado apenas em comando do cliente. Implementar: (a) validação que o buff foi legitimamente 'adquirido' (stored no BD do servidor, não no client); (b) verificação de duração/validade; (c) rate-limiting; (d) re-checksum do estado de buff no servidor. Efeitos de buff (ouro, exp, stats) devem ser calculados e aplicados no servidor, não confiados do cliente.

## [ALTA] Client-side box reward double-claiming via log tampering
- **classe:** `TaskbarHero.Log.GetBoxLog`
- **problema:** A classe GetBoxLog registra eventos de abertura de caixa (bfod=boxId/tipo, bfoe=quantidade, bfof=rewardValue). Como outros logs do jogo, provavelmente persiste localmente em save-file e é sincronizado ao servidor. Se o servidor não validar independentemente (ex: verificando se a caixa foi realmente desbloqueada/consumida no servidor), um jogador pode: (1) editar o save-file para duplicar entradas de log, (2) enviar multiple sync requests com o mesmo GetBoxLog, ou (3) manipular os campos de recompensa antes do envio.
- **verificar:** Monitorar requisições POST para `/api/sync-logs` ou similar. Verificar se: (a) servidor re-calcula reward vs. cliente, (b) há idempotência por box-id+timestamp, (c) campos bfod/bfoe/bfof são re-validados server-side contra resource state. Tentar abrir uma caixa, editar save com dnSpy/Cheat Engine, enviar log duplicado e verificar se reward é duplicado.
- **correção:** Implementar server-side validation: (1) Manter registro server-side de caixas abertas por user+timestamp. (2) Rejeitar GetBoxLog se box-id já foi processado. (3) Calcular rewards no servidor, não confiar em bfof. (4) Assinar logs com HMAC(user_secret) para detectar tamper no trânsito. (5) Usar StageBoxRequestAbuseGuard para rate-limit, mas estenda para rate-limit por reward-value, não apenas contagem.

## [ALTA] Client-Authoritative Combat + Weak Rate-Limiting → Amplified Farming
- **classe:** `TaskbarHero.StatusSystem.AccountStatus`
- **problema:** Combat resolution (kills, boss chest drops) acontece NO CLIENTE. Servidor confia no evento reportado SEM validar taxa contra wall-clock real. Accelerando Time.timeScale (~6×), kills são reportados 6× mais rápido; servidor aplica chance-de-drop por kill sem sanidade check. Resultado: farm de itens econômicos amplificado, fluxo pra Steam trading afeta outros jogadores.
- **verificar:** 1) Elevar Time.timeScale no cliente e medir kills/min reportados vs wall-time real; 2) Analisar StageBoxRequestAbuseGuard (client-side, contornável via MelonLoader/HarmonyX); 3) Verificar se servidor tem rate-limit por conta no endpoint de baú; 4) Confirmar que StageBoxRequestAbuseGuard é a ÚNICA barreira.
- **correção:** SERVER-SIDE: (1) Implementar rate-limit em wall-clock (não game-time) no endpoint de baú; (2) Validar tempo mínimo de clear (HP_mob ÷ DPS_player_max); (3) Rejeitar/limitar kills acima de taxa fisicamente plausível; (4) Remover confiança em eventos crus de kill — derivar baús de progresso validado no servidor; (5) Telemetria de timeScale/taxa anômala.

## [ALTA] Client-Authority em Stats de Modificação de Item
- **classe:** `TaskbarHero.Skill.UniqueModData`
- **problema:** UniqueModData armazena EquipModType e até 5 valores numéricos que definem o efeito da modificação. Se enviados ao servidor sem revalidação de tipo, range ou whitelist, permitem stat inflation, item tampering via save-game editor (Cheat Engine, UAFFS) e potencial dupe.
- **verificar:** Capturar uma requisição de equip/craft no servidor: os valores de Value1-Value5 vêm do cliente? Há validação server-side contra minerais de EquipModType permitidos? Ou o servidor apenas persiste o que o cliente mandou?
- **correção:** 1) Servidor mantém whitelist de EquipModType × range válido para cada Value. 2) Revalida TODOS os campos antes de salvar/aplicar bonus. 3) Tie UniqueModData a item ID + timestamp do servidor (immutable após geração). 4) Rate-limit criação de mods server-side, não client.

## [ALTA] Client-Authority em AttributeLevel + Save Tamperável
- **classe:** `TaskbarHero.EasySaveData.AttributeSaveData`
- **problema:** A classe serializa atributos numéricos (Level) localmente sem proteção criptográfica ou assinatura. Se o servidor aceita estes valores como verdade, um jogador edita o arquivo de save e aumenta arbitrariamente força/defesa/vida/ouro.
- **verificar:** 1) Localize save em AppData/players ou ~/.config. 2) Edite com hex editor ou JSON (Level: 1→999). 3) Relance jogo. Se servidor reflete mudança sem recomputa, é exploração trivial. Teste trading de items com atributos falsificados.
- **correção:** Servidor: (1) Nunca confie em attributeLevel do cliente — computar server-side (Level = XP_acumulada / tabela_escala). (2) Validar contra eventos de combate/quest (auditlog). (3) Criptografar+assinar saves com HMAC/Ed25519 (chave privada servidor). (4) Rejeitar saves com assinatura inválida. (5) Enforçar rate-limit de atributo no servidor, não cliente.

## [ALTA] Client-Authority em AccountStatusCategoryData — Progressão/Achievements Editáveis
- **classe:** `TaskbarHero.StatusSystem.AccountStatusCategoryData`
- **problema:** A classe gerencia achievements/estatísticas da conta no cliente (marked 'system: player') com métodos como GetOrAddEntry() e ClearCategory() sem indicação de validação server-side. Servidor aparentemente sincroniza via SyncToServer() aceita dados do cliente sem revalidar integridade.
- **verificar:** 1) Editar arquivo save local JSON (trocar achievements de false→true ou incrementar stats); 2) Chamar GetOrAddEntry() com achievement falso via debugger; 3) Verificar se LoadFromServer() revalida ou apenas sobrescreve local com remoto; 4) Testar se ClearCategory() no cliente reflete no servidor sem validação.
- **correção:** Servidor DEVE: (1) Manter source-of-truth de achievements/stats no BD (nunca aceitar valores brutos do cliente); (2) Revalidar CADA achievement quando cliente sincroniza (ex: 'KillCount=1000' → verificar se matou realmente 1000 mobs pelo logs); (3) Usar HMAC/assinatura digital pra dados persistidos localmente (detect tampering); (4) Rate-limit GetOrAddEntry no servidor se aceita submissão de novas categorias; (5) Logs de auditoria: quem desbloqueou qual achievement e quando.

## [ALTA] Client-Authority: Exp/Level tamperáveis sem validação server-side
- **classe:** `TaskbarHero.EasySaveData.CubeLevelSaveData`
- **problema:** EasySaveData persiste Level e Exp localmente sem encriptação/assinatura. Um jogador pode editar o save e aumentar Exp/Level infinitamente — se o servidor apenas lê esses valores sem re-validar progressão, qualquer um faz dupe/bypass.
- **verificar:** Intercepte tráfego do cliente; edite o save local; suba um XP sem interagir com o servidor; confirme se o servidor aceita o valor sem questionar a progressão (intervalo Exp esperado vs Exp enviado).
- **correção:** Server-side validation obrigatória: (1) servidor calcula Exp esperado baseado em eventos auditados (kills, quests completadas); (2) client envia evento, servidor calcula novo Exp e retorna diferença; (3) nunca confie em Level/Exp do cliente, use como hint apenas para UI local.

## [ALTA] Client-side progression data tamperable — sem validação server-side
- **classe:** `TaskbarHero.EasySaveData.HeroSaveData`
- **problema:** Campos como HeroLevel, HeroExp, AbilityPoint, equippedItemIds e unlockedAttributeGroupKeys são armazenados no cliente (EasySaveData) sem aparente validação/assinatura. Atacante pode editá-los via memory hacker (Cheat Engine) ou editor de save, permitindo level-up ilimitado, dupe de items, equipar gear sem requisito, ou unlock indevido de habilidades.
- **verificar:** 1) Usar Cheat Engine pra aumentar HeroLevel localmente; 2) Enviar comando ao servidor com nível falsificado e observar se é aceito; 3) Adicionar item inexistente a equippedItemIds e tentar equipar; 4) Confirmar se servidor valida que item existe no inventário antes de aplicar.
- **correção:** Servidor deve: 1) Armazenar HeroLevel/HeroExp/AbilityPoint APENAS server-side; enviar ao cliente apenas pra render UI. 2) Validar cada equip: confirmar item existe no inventário autêntico antes de aplicar. 3) Verificar requisitos (level, unlock, cooldown) server-side, nunca confiar em cliente. 4) Rate-limit server-side (ex: claim reward máx 1×/5min). 5) Assinar dados críticos com HMAC(server_secret) ou encriptar save file.

## [ALTA] Salvamento Local Tamperável + Client-Authority de Moeda
- **classe:** `TaskbarHero.EasySaveData.CurrencySaveData`
- **problema:** EasySaveData armazena Quantity no cliente sem integridade criptográfica, permitindo edição direta de arquivo/memória. Servidor sem validação rigorosa cria oportunidade de duplication de currency e exploração de economia de jogo.
- **verificar:** 1) Localizar arquivo de save do cliente (PlayerPrefs/AppData); 2) Editar valor de Quantity manualmente (ex: 99999999); 3) Reabrir jogo e sincronizar com servidor; 4) Confirmar se servidor aceitou valor inflado sem auditoria de origem legítima da moeda.
- **correção:** 1) Servidor é canonical currency state (jamais confia em valores do cliente); 2) Hash/HMAC assinado do save com chave secreta do servidor; 3) Rate-limit + validação de delta razoável (quanto a quantidade pode mudar em X tempo); 4) Log de auditoria de todas as mudanças de currency; 5) Anti-cheat detecta desincronização; 6) Se moeda é tradeable, validar integridade antes de permitir transação Steam.

## [ALTA] Client-Authority em Desbloqueio e Ownership de Item
- **classe:** `TaskbarHero.EasySaveData.InventorySaveData`
- **problema:** IsUnlock e IsUnlockedByRune são booleans serializados no cliente que definem acesso a conteúdo. Se servidor confia destes valores durante sincronização sem re-validar (quest feita? runas gastas?), jogador edita arquivo de save ou memória e falsifica desbloqueios. ItemUniqueId sem validação de ownership + Steam trading = dupe de item com valor real.
- **verificar:** Localize save em AppData/LocalLow/[game]/ ou ~/.local/share. Edite JSON/protobuf: IsUnlock→true, IsUnlockedByRune→true, ItemUniqueId→ID raro. Inicie + sincronize com servidor. Se servidor aceita sem erro, é vulnerável.
- **correção:** Servidor re-valida TUDO: (1) IsUnlock = recalcula baseado em quest_completed + required_level no DB; (2) IsUnlockedByRune = checa se jogador gastou runas suficientes (server-side journal); (3) ItemUniqueId = lookup na DB, rejeita IDs inválidos/não-no-inventário. Nunca confie em flags do cliente.

## [ALTA] Client-Authority de Contadores e Flags Críticas de Item
- **classe:** `TaskbarHero.EasySaveData.ItemSaveData`
- **problema:** Campos como EnchantCount, DecorationAppliedTotalCount, IsBlocked e IsServerPendingItem sugerem que o cliente serializa/controla contadores de modificações e flags de sincronização. Se o servidor aceita esses valores sem re-validação, permite duping de items, bypass de limites de encantamento, e falseamento de estado.
- **verificar:** 1) Edite um ItemSaveData local aumentando EnchantCount de 3→100 e confirme se servidor aceita. 2) Altere UniqueId de um item equipado pra duplicar no inventário. 3) Mude IsBlocked:false→true no save e veja se servidor honra. 4) Inspecione logs do servidor: está re-validando EnchantData vs banco de dados ou confiando no cliente?
- **correção:** Servidor DEVE: (1) Ignorar EnchantCount, totalizadores e flags de sync — recalcular do histórico/transações persistidas. (2) Validar EnchantData contra schema permitido + limites por raridade/nível. (3) UniqueId gerado servidor-side, não confiável do cliente. (4) IsBlocked/IsChaotic/IsServerPendingItem são atributos do servidor, não do cliente — recalcular a cada load. (5) Assinar/hash ItemSaveData se mantiver localmente, recusar se MAC falhar.

## [ALTA] Client-Authority em Resultado de Encantamento + Duping via TxId
- **classe:** `TaskbarHero.EasySaveData.PendingEnchantSaveData`
- **problema:** O campo 'Enchant' (resultado do encantamento) está armazenado no cliente e presumivelmente enviado ao servidor para persistência sem re-cálculo. Um atacante pode: (1) editar o save EasySaveData para um Enchant superior, (2) duplicar itens enviando o mesmo TxId em múltiplas requisições se o servidor não implementar idempotência, (3) contornar cooldown alterando TimestampTicks.
- **verificar:** No servidor: intercepte POST /enchant (ou similar). Envie dois requests com mesmo TxId — se ambos forem processados, há duping. Altere o valor de 'Enchant' no cliente (EasySaveData é frequentemente text JSON/plaintext) — se o servidor aceitar sem recalcular, há client-authority.
- **correção:** Servidor DEVE: (1) recalcular Enchant determinísticamente a partir de RecipeType + seed serverSide, ignorando valor do cliente; (2) validar MaterialUniqueId/GearUniqueId contra inventário do jogador; (3) implementar idempotência usando TxId com deduplicação no banco; (4) rate-limit SERVIDOR-side usando timestamp do servidor, não do cliente.

## [ALTA] Client-Authority em Valor de Modificador de Stat (UniqueModStat.Value)
- **classe:** `TaskbarHero.Skill.UniqueModStat`
- **problema:** ItemEnchantSaveData serializa Value (modificador) no cliente e PendingEnchantSaveData valida apenas IDs+timestamp, nunca o valor. Servidor recebe TxId e aplica modificadores sem recalcular a receita, permitindo cliente inflar stats arbitrariamente.
- **verificar:** 1) Editar save file EasySave de ItemEnchantSaveData, alterar Value (15→150); 2) Enviar PendingEnchantSaveData com Value editado; 3) Verificar se servidor aplica o valor malicioso ou rejeita. Se aplica, vulnerável.
- **correção:** Servidor DEVE re-calcular o output da receita (RecipeType + MaterialKey + Tier) antes de aplicar. Nunca confiar em Value que vem do cliente. Assinar/HMAC os dados de encantamento no cliente e validar assinatura no servidor.

## [ALTA] Client-authority em enchantment data + save tampering + material dupe
- **classe:** `TaskbarHero.EasySaveData.ItemEnchantSaveData`
- **problema:** ItemEnchantSaveData é serializado localmente (EasySave = JSON editável). Atacante pode: (1) elevar Tier/Value arbitrariamente editando save; (2) consumir materiais múltiplas vezes salvando antes do consumo; (3) enviar recipes/materials inválidos ao servidor se não há revalidação full.
- **verificar:** 1) Editar save file local, aumentar Tier pra 999, recarregar - persiste?  2) Tentar duplicar MaterialKey salvando antes de consumo  3) Fazer POST de enchantment com RecipeType/ModType inexistentes - servidor aceita?
- **correção:** Servidor DEVE: (1) revalidar Tier ∈ [1..MAX_VALID]; (2) revalidar materials existem no inventory do player E subtraí-los atomicamente no servidor (não confia em save); (3) whitelist de RecipeType/ModType válidos; (4) consumo de material é state-change APENAS no servidor, nunca no cliente; (5) rate-limit enchantment server-side, não client-side

## [ALTA] Client-side authority sobre ItemUniqueId no stash de trading
- **classe:** `TaskbarHero.EasySaveData.TradingStashSaveData`
- **problema:** EasySave persiste ItemUniqueId como JSON/binary editável. Atacante pode alterar diretamente qual item está armazenado (ou duplicar). Se o servidor confirma trades apenas consultando este campo sem re-validar posse real, permite dupe de items valiosos e exploits de economia.
- **verificar:** 1) Capturar save file (Documents/TaskbarHero/... ou AppData/LocalLow). 2) Decodificar EasySave (JSON ou binary). 3) Editar ItemUniqueId para um item raro. 4) Carregar save. 5) Ofertar em trade. 6) Verificar se servidor aceita item inexistente ou cria dupe.
- **correção:** Nunca confiar em ItemUniqueId do cliente. No trade: servidor consulta inventário autorizado do PLAYER A no DB, valida se item existe E pertence a A, remove atomicamente, só ENTÃO confirma trade. EasySave é apenas display. Adicionar hash HMAC (chave server-side) da save se quiser anti-tamper local.

## [ALTA] Client-Authority Game Progression + Save Tampering (OWASP A01/A08)
- **classe:** `TaskbarHero.EasySaveData.CubeRecipeSaveData`
- **problema:** CubeRecipeSaveData armazena progresso crítico localmente sem assinatura/integridade. Campo MaxUnlockRecipeKey editável no save; servidor não valida ao carregar. Permite bypass de progression e dupe de itens.
- **verificar:** Editar save file localmente (elevar MaxUnlockRecipeKey). Carregar jogo. Verificar se todas receitas desbloqueadas sem ação legítima de gameplay.
- **correção:** Server recalcula progresso baseado em ações legítimas (quests, combats, achievements). Nunca confiar em MaxUnlockRecipeKey do cliente. Implementar HMAC-SHA256 do save. Sincronizar dados críticos com servidor (cloud save). Rate-limit crafting server-side.

## [ALTA] Client-Authority Inventory Tampering — Bypass de Desbloqueio & Duping de Items
- **classe:** `TaskbarHero.EasySaveData.RemakeTradingStashSaveData`
- **problema:** RemakeTradingStashSaveData é serializado no cliente (EasySave) e carregado sem validação server-side. Atacante edita ItemUniqueId e IsUnLock localmente → servidor aceita blindamente, permitindo duping de items e desbloqueio grátis de slots.
- **verificar:** Localize Application.persistentDataPath/EasySave/... → extraia RemakeTradingStashSaveData (JSON/binary) → edite IsUnLock=true ou copie ItemUniqueId → reinicie e observe se servidor sincroniza SEM erro. Se aceitar, vulnerável.
- **correção:** NUNCA confie em inventário do cliente. Servidor: (1) mantém source-of-truth de stash, (2) ao receber mudança: valida posse do item + checksum HMAC-SHA256 assinado, (3) nega qualquer ItemUniqueId que não existe em player inventory, (4) rate-limita mudanças stash (máx 5/min). Remova RemakeTradingStashSaveData do sync server — deixe como cache local apenas.

## [ALTA] Client Authority em Stash (Inventário)
- **classe:** `TaskbarHero.EasySaveData.StashSaveData`
- **problema:** Cliente controla Index (slot arbitrário), ItemUniqueId (item arbitrário) e IsUnLock (desbloqueio arbitrário), permitindo dupe de itens, escrita em slots alheios e bypass de pré-requisitos de desbloqueio.
- **verificar:** Enviar StashSaveData com Index fora do range, ItemUniqueId fabricado ou IsUnLock=true sem ter obtido o item; verificar se servidor persiste a mudança sem validação.
- **correção:** Servidor valida: (1) Index pertence ao jogador; (2) ItemUniqueId existe no inventário deste jogador (anti-dupe: verificar se já possui esse item_id neste slot); (3) IsUnLock requer pré-requisitos (moeda gasta, level, quest) — cliente envia intenção, servidor valida e persiste.

## [ALTA] Client-Authority Vulnerability: Progression Data Stored Client-Side
- **classe:** `TaskbarHero.EasySaveData.RuneSaveData`
- **problema:** RuneKey e Level armazenados em EasySaveData (cliente) sem validação server-side, permitindo edição direta do save file, duping e progression spoofing. AntiCheat protege código executável mas não valida estado persistente.
- **verificar:** 1) Editar save file localmente (Level=999+), sincronizar e verificar se servidor aceita; 2) Tentar duping via backup/restore de RuneSaveData; 3) Monitorar anomalias (nível impossível para tempo de jogo); 4) Rastrear trades de runes com IDs/levels inválidos
- **correção:** Server-side source-of-truth obrigatória: cliente envia apenas ações ('upgrade rune X to level Y'), servidor valida permissões/custos/limites e aplica. Remover progression data de EasySaveData ou criptografar+assinar. Implementar rate limiting, validação de anomalias e audit log no backend.

## [ALTA] Client-Authority em Seleção de Material de Craft
- **classe:** `TaskbarHero.UI.CraftingMaterialSlotArrow`
- **problema:** m_slotIndex sendo manipulável no cliente permite que atacante selecione materiais inválidos, duplicados ou contorne validação de custo. Se o servidor confia neste índice para validar craft, resulta em dupe de item e quebra de economia.
- **verificar:** Interceptar POST de craft via proxy, alterar material_slot_index para índice inválido/fora-de-range/duplicado. Se servidor aceita e crafta, é vulnerável. Testar também modificação em memória via Unity debugger.
- **correção:** Servidor DEVE re-validar completamente: (1) recuperar inventário real do BD, (2) confirmar que slot_index existe E tem quantidade suficiente, (3) recomputar custo total, (4) NUNCA confiar em valores do cliente para validação de craft. Implementar server-authoritative inventory state.

## [ALTA] Client-Authority em Claim de Mail + Falta Validação Server-Side de Propriedade
- **classe:** `TaskbarHero.Manager.MailBoxPoolManager`
- **problema:** MailBoxPoolManager sincroniza slots visuais via SyncMailBoxWithBackend mas OnClaimButtonClicked valida apenas no cliente (CanClaimItem) antes de enviar InventoryProcessBoxRequest. Servidor não re-valida se o jogador é proprietário do mail, permitindo double-claim, reivindicação de mails de outros jogadores, ou bypass se MelonLoader desativa checks client-side.
- **verificar:** 1) Interceptar POST para /api/claimMail com itemInstanceId de mail pertencente a outra conta; 2) Verificar se servidor rejeita ou aceita; 3) Testar múltiplos claims concorrentes do mesmo mailId (race condition). Se servidor aceita sem validar ownership, confirma vuln.
- **correção:** Implementar server-side: (a) verificar token/sessão do jogador contra mailOwnerID na BD; (b) marcar mail como 'claimed' atomicamente antes de adicionar item ao inventário; (c) rate-limit claims por conta no backend, não apenas client-side StageBoxRequestAbuseGuard.

## [ALTA] UI Recipe Selection sem re-validação server-side → possível craft não autorizado de receita bloqueada
- **classe:** `TaskbarHero.UI.CubeComboBoxImageCharger`
- **problema:** A classe CubeComboBoxImageCharger gerencia seleção visual de receitas no dropdown sem indicação de validação server-side antes do craft. Se o servidor confia no ID da receita vindo do cliente (via OnComboBoxSelectionChanged → crafting request) sem re-verificar se a receita está desbloqueada (IsRecipeUnlocked) ou disponível, atacante pode craft receitas bloqueadas enumerando IDs. Crítico em economia de Steam market.
- **verificar:** 1) Interceptar OnComboBoxSelectionChanged + enviar crafting request com RecipeType/ID não desbloqueado (craft receita de fase futura). 2) Verificar se servidor aceita ou rejeita. 3) Se aceitar: dupe de items raros → breach econômico. Debugar: dump memoria + modificar m_recipeMenu para incluir receitas com IsRecipeUnlocked=false.
- **correção:** Servidor DEVE re-validar na endpoint de crafting: (a) CubeRecipeSaveData.IsRecipeUnlocked(requestedRecipeId, userAccount) ✓ antes de aplicar crafting logic; (b) Assinar receita com server timestamp + HMAC pra evitar replay de receitas antigas; (c) Rate-limit de crafting por wall-clock time (não timeScale).

## [ALTA] Client-authoritative Steam reconnection enable bypass via Time.timeScale + missing wall-clock validation
- **classe:** `TaskbarHero.UI.SteamDisconnectPopUp`
- **problema:** Timer de reconexão é decrementado localmente sem validação server-side de tempo real. Atacante eleva Time.timeScale → timer artificial cai rápido → falsificar reconexão bem-sucedida sem intervalo real. Servidor não valida wall-clock → amplifica farm de baús/itens.
- **verificar:** Elevar Time.timeScale para ~6×, aguardar pop-up de desconexão, observar se timer decresce em ~1/6 do tempo real; reconectar artificialmente; observar se backend aceita farm acelerado como na PoC de TBH_VULN_DISCLOSURE.md
- **correção:** Server-side: (1) Validar tempo real transcorrido entre desconexão/reconexão usando wall-clock, rejeitar reconexões sem intervalo mínimo plausível; (2) Rate-limit por conta em /api/reconnect e endpoints de farm usando relógio real (não Time.timeScale); (3) Validar progressão reportada pós-reconexão contra DPS teórico máximo + duração real da fase.

## [ALTA] Encantamentos não validados server-side (Risk Dependency)
- **classe:** `TaskbarHero.UI.EnchantSlotViewer`
- **problema:** EnchantSlotViewer é UI-only (benigna per se), MAS se o servidor não re-valida toda aplicação de encantamento contra inventário real do player, está vulnerável a tamper de saves, dupe via race-condition e bypass de rate-limit client-only.
- **verificar:** 1) Intercepte requisição de aplicação de encantamento (Charles/Burp). 2) Modifique 'enchantmentType' ou 'level' pra um inválido/ultra-raro. 3) Se servidor aceita sem erro → vulnerável. 4) Teste spam-click rate-limit (clique 10x rápido) e veja se servidor aceita todos.
- **correção:** Server-side: (1) Re-valida que player tem item + materiais; (2) Re-valida tipo/nível encantamento vs whitelist; (3) Rate-limit real (token bucket 1 encantamento/5s); (4) Atomic transaction (BEGIN → UPDATE item → UPDATE materials → COMMIT)

## [ALTA] Client-Authority em Seleção de Tipo de Síntese — Crafting não re-validado server-side
- **classe:** `TaskbarHero.UI.SynthesisItemTypeComboBoxButton`
- **problema:** PopulateButtons inicializa tipos de item baseado no inventário local do cliente sem re-validação no servidor. UpdateButtonStates ativa/desativa filtros conforme disponibilidade calculada no cliente. Attacker pode tamperar memória/inventory local, selecionar tipos bloqueados, e craftar itens inválidos.
- **verificar:** 1) Editar inventory local (ex: adicionar 999 materiais via CheatEngine) 2) Ativar combobox e verificar se tipos normalmente bloqueados aparecem 3) Craftar item inválido 4) Confirmar se servidor aceita a síntese sem re-validar quantidade/tipo no backend
- **correção:** Servidor DEVE re-validar antes de executar síntese: (a) verificar se player tem quantidade exata de materiais no BD, (b) validar se recipe/tipo está desbloqueado para player, (c) aplicar dano ao inventário APÓS confirmar disponibilidade, (d) usar transação DB pra atomic craft+debit. Client-side é apenas UI/UX.

## [ALTA] Client Authority em Cálculo de Stats & Dano
- **classe:** `TaskbarHero.UI.UI_Status`
- **problema:** UI_Status exibe DPS, dano de ataque, crítico, armor e cooldown que, em um jogo que 'resolve muita coisa no cliente', provavelmente são calculados/sincronizados do lado do cliente. Se o servidor confia em valores de `attack_dmg`, `max_hp`, `cri_dmg` ou `cool_time_reduction` vindos do cliente para resolver combate, trades ou progressão, atacante edita memória/save e duplica dano ou bypassa cooldown.
- **verificar:** Interceptar POST/sync entre cliente e servidor; procurar por: `{attack_dmg: X, max_hp: Y, dps: Z}` em payloads de combate/compra/venda. Se servidor executa dano baseado em `attack_dmg` cliente, ou valida inimigo morte com `damage` do player, é confirmado.
- **correção:** Servidor calcula TODOS os stats (DPS, dano, crítico, armor, cooldown) baseado em `{level, equips[], skills[], buffs}`. Cliente sincroniza apenas `level` e `equips`. Servidor valida cada ação: 'Player level 30 não pode fazer 99999 de dano'. AntiCheat (CodeStage) não substitui validação server-side em lógica crítica.

## [ALTA] Client-Authority State Tampering + Cooldown/Purchase Bypass
- **classe:** `EArrangeState`
- **problema:** EArrangeState parece ser um enum que o cliente pode manipular ou enviar ao servidor sem validação server-side. Nomes como ArrangeCooldown, NotPurchased, NeedPurchaseDLC, CanNotArrangeBecauseActBoss indicam que validações críticas são apenas client-side. Isso permite: (1) dupe de itens via falsificação de NotPurchased→ReadyArrange, (2) rate-limit bypass de ArrangeCooldown→ReadyArrange (spam de arranjos), (3) contorno de DLC/condições, (4) unfair advantage em Steam trading.
- **verificar:** 1. Capture cliente-servidor (Fiddler/mitmproxy), modifique EArrangeState de NotPurchased/ArrangeCooldown para ReadyArrange na request — se servidor aceita, vulnerável. 2. Force estado localmente (debugger/save editor), faça nova ação — se não rejeita, é puramente client-side. 3. Tente arranjar sem DLC/condição — se funciona, servidor não valida.
- **correção:** Server-side authority: (1) Servidor é fonte de verdade (timestamps, ownership, DLC status em DB). (2) Cliente envia ACTION, não estado. (3) Servidor valida: ownership, cooldown (timestamp servidor), DLC ativo, condições satisfeitas. (4) Servidor retorna estado novo e autorizado. (5) Rate-limit servidor-side + logging de anomalias de arranjo frequente.

## [ALTA] Client-Authority Inventory Sync: Slider Value Trusted Server-Side
- **classe:** `TaskbarHero.UI.SteamInventoryFixer`
- **problema:** Slider parameter (m_slider) é enviado ao servidor sem server-side re-validação de limites. Atacante intercepta requisição HTTP, modifica sliderValue para valor impossível (ex: 999999), e servidor aceita quantidade fraudulenta, causando duplicação ou inflação de inventário.
- **verificar:** 1. Interceptar POST /api/fix-inventory com Burp Suite durante clique em m_fixingTriggerButton. 2. Modificar parâmetro do slider (ex: sliderValue: 100 → 100000). 3. Se servidor responde HTTP 200 e aceita valor, vulnerabilidade confirmada. 4. Verificar se TheBackend executa validação servidor-side ou apenas aceita valor do cliente.
- **correção:** 1. Remover confiança no valor do cliente. 2. Servidor deve ter CANONICAL inventory state (fonte de verdade). 3. Validar slider range NO SERVIDOR após receber requisição (min/max hardcoded no backend, não enviado pelo cliente). 4. Implementar server-side rate-limiting (por userId + timestamp). 5. Assinar/hash dados de inventário com chave privada server-side para detectar tampering.

## [ALTA] Client-Authority em Cálculo de Drops Probabilísticos
- **classe:** `TaskbarHero.Data.DropInfoData`
- **problema:** Classe DropInfoData expõe pesos (Weight), tipos de recompensa e condições de drop que o cliente pode acessar e processar. Se o cliente calcula qual drop ocorreu e envia ao servidor (vs servidor calculando server-side com seed), permite manipulação de probabilidades via memory editing ou config modification.
- **verificar:** Capturar requisição HTTP/UDP quando herói mata inimigo: servidor recebe ['dropID'] (vulnerável) ou ['seed+enemyID+heroState'] que ele re-processa? Se recebe drop já decidido, o AntiCheat não vai detectar mudanças em valores de Weight.
- **correção:** Servidor gera drop determinístico (HMAC-SHA256 de seed+encounter+timestamp) sem confiar em dados do cliente. Cliente recebe APENAS {dropID, itemID, quantity} já validado. Remover Weight/REWARDTYPE da DropInfoData acessível client-side ou mover cálculo pra servidor.

## [ALTA] Client-Authority em ItemGroupInfoData: Risco de Duplicação/Transferência Ilícita de Itens
- **classe:** `TaskbarHero.Data.ItemGroupInfoData`
- **problema:** ItemKey e ItemGroupKey armazenados no cliente e transmitidos sem validação servidor-side permitem duplicação de itens raros, alteração de propriedade, e contrabando em trading. Servidor não valida se o jogador realmente possui o item antes de confirmar transações.
- **verificar:** Memory patcher/save editor: alterar ItemKey para um item valioso inexistente no inventário; tentar tradear esse item falsificado; verificar se servidor processa transação sem validar propriedade contra banco de dados.
- **correção:** Servidor: (1) manter inventory como fonte-da-verdade em DB, nunca confiar em ItemKey do cliente; (2) validar cada ItemKey contra registro do jogador autenticado antes de qualquer operação; (3) usar transações atômicas com rollback se item desaparecer entre verificação e execução; (4) auditar trades com logs imutáveis.

## [ALTA] Client-Authority Inventory — Servidor confia em CostForUnLock e Index do cliente
- **classe:** `TaskbarHero.Data.InventoryInfoData`
- **problema:** A classe é enviada do cliente para o servidor sem validação. Um atacante pode alterar CostForUnLock para 0 (bypass de unlock gratuito), Index para valores inválidos, ou duplicar items enviando múltiplas vezes. O servidor não valida integridade, autorização, recurso suficiente do jogador ou existência do item.
- **verificar:** 1) Intercepte a requisição de unlock (Fiddler/Charles) e altere CostForUnLock:0. 2) Verifique se unlock é permitido gratuitamente. 3) Envie Index=-1 ou 999999. 4) Envie múltiplos requests idênticos → verifica-se dupes. 5) Analise se o servidor valida contra sua própria cópia dos dados (base de dados).
- **correção:** Servidor é source-of-truth: manter Index e CostForUnLock APENAS no banco. Cliente envia APENAS ID do jogador + tentativa de unlock. Servidor valida: (1) Item existe, (2) Custo é correto, (3) Jogador tem recursos, (4) Não foi desbloqueado, (5) Rate-limit por jogador. Responda ao cliente APENAS com sucesso/falha, nunca com dados brutos.

## [ALTA] Client-Authority em Inventário — Ausência de Validação Servidor
- **classe:** `TaskbarHero.Data.InventoryInfoDataMap`
- **problema:** InventoryInfoDataMap é um dicionário local que armazena itens do jogador sem validação lado-servidor. Um cliente pode Add/Set itens arbitrários, duplicar raros, ou falsificar quantidades sem que o servidor valide de verdade, permitindo dupe e fabricação de items ilegais.
- **verificar:** Capturar requisição de sync de inventário (POST/PUT ao backend), modificar count/itemID antes de enviar, e verificar se o servidor aceita valores impossíveis ou já duplicados; tentar Add direto ao mapa local sem envio ao servidor.
- **correção:** Servidor DEVE validar cada transação: Add só com item que existe em loot-table/quest-reward; Remove só com consumo real; Count incrementa/decrementa de forma atômica com auditoria; client envia apenas intenção ("use item X"), servidor executa e retorna novo estado autorizado. Nunca confiar no estado do InventoryInfoDataMap do cliente como verdade de autoridade.

## [ALTA] Client-Authority em GRADE, IsCanExchangeMarketable e DropCooldown
- **classe:** `TaskbarHero.Data.ItemInfoData`
- **problema:** A classe armazena metadata crítica (raridade, tradabilidade, cooldown de drop) localmente. Se o servidor confiar nesses valores sem validar contra fonte de verdade (servidor), permitem dupe de items raros, desbloqueio de trading em items bound, e farm acelerado.
- **verificar:** 1) Editar ItemInfoData.json / save local (GRADE: 5→6, IsCanExchangeMarketable: false→true, DropCooldown: 86400→0); 2) Enviar ao servidor (trade/sell/drop); 3) Se aceito sem error, é exploitável. Confirmar com Wireshark/proxy que o cliente envia os valores e servidor não revalida contra database.
- **correção:** Servidor é fonte de verdade: (a) em cada requisição, recarregar item metadata do DB, não confiar em cliente; (b) validar GRADE/raridade antes de marketplace/trade; (c) enforçar DropCooldown server-side (timestamp de último drop), ignorar cliente; (d) tradabilidade (IsCanExchangeMarketable) validar contra account flags (steam-bound, marketplace-restricted, etc), não campo do item local.

## [ALTA] Client Authority em Equipamento: Enum permite ao cliente definir slots sem validação
- **classe:** `TaskbarHero.Data.EItemParts`
- **problema:** Enum de slots é inerte, mas o contexto 'resolve MUITA coisa no CLIENT' sugere: cliente envia qual item equipar em EItemParts.MAIN_WEAPON, EItemParts.HELMET, etc., e servidor confia sem validar posse/elegibilidade. Permite: dupe (mesmo item em múltiplos slots), equipar itens inexistentes, bypass de requerimentos (level/quest).
- **verificar:** Interceptar pacote de equipagem (Fiddler/Burp): enviar slot MAIN_WEAPON com item_id=999999 (inexistente). Se servidor aceita, ou tenta equipar item já em outro slot — confirmado. Verificar se servidor valida: (1) jogador possui item, (2) item não está em outro slot, (3) reqs de level/classe.
- **correção:** Servidor NUNCA confia em EItemParts do cliente. No server: player.equipItem(item_id, slot_enum) deve: verificar item_id na DB do jogador, validar slot é legal (não enum injection), desunequip slot anterior, validar reqs do item, persistir em DB, notificar cliente com estado authoritative. Cliente é só viewer.

## [ALTA] Client-Authority em Multiplicadores de Stats de Item
- **classe:** `TaskbarHero.Data.ItemLevelScaleInfoDataMap`
- **problema:** ItemLevelScaleInfoDataMap mapeia multiplicadores/adições de stats no cliente. Se servidor confiar em valores enviados pelo cliente (sem recalcular) ou aplicar localmente para transações críticas (venda, trading, battle rewards), permite exploração via Cheat Engine ou memory patching.
- **verificar:** Procure: servidor aceita stats do cliente em requests de venda/trading sem validar contra item_id+level. Ou: cliente calcula dano/defesa final e servidor confiam nesse valor para battle resolution ou economia.
- **correção:** Servidor sempre recalcula stats a partir de (item_id, level, rarity). Cliente envia apenas identifiers. Servidor retorna stats assinados; cliente rejeita valores que falham verificação. Nenhuma lógica crítica (dano, venda, dupe check) roda apenas no cliente.

## [ALTA] Client-Authority em Recompensas (RewardGold/RewardExp)
- **classe:** `TaskbarHero.Data.MonsterInfoData`
- **problema:** A classe define RewardGold e RewardExp como dados estáticos carregados no cliente. Se o servidor confia no valor reportado pelo cliente quando um monstro é derrotado, um atacante pode editar MonsterInfoData.json (plaintext/ScriptableObject) no disco e receber 10–1000× a recompensa legítima. Padrão crítico em idle-RPG onde prêmios são a mecânica central.
- **verificar:** 1) Procurar como TaskbarHero.MonsterInfoData é carregado (Resources/StreamingAssets?). 2) Rastrear fluxo de 'monstro derrotado' → 'recompensa enviada ao servidor'. 3) Confirmar se servidor aceita {gold: X, exp: Y} do cliente ou se revalida contra dados canônicos server-side. 4) Tentar editar MonsterInfoData.json localmente, derrotar monstro, observar se ouro recebido é o alterado.
- **correção:** Servidor é fonte de verdade: manter MonsterInfoData canônico no backend. Cliente envia apenas {monsterId}. Servidor busca RewardGold/RewardExp de sua DB, aplica modificadores (battle multipliers), calcula crédito final. Cliente exibe resultado, mas não autoriza.

## [ALTA] Client-Authority em Cálculo de Recompensas (Ouro Alquimia / XP Cubo)
- **classe:** `TaskbarHero.Data.ItemLevelScaleInfoData`
- **problema:** ItemLevelScaleInfoData calcula escalas de recompensas (AlchemyGoldScale, CubeExpScale) no cliente e fornece métodos GetAdjustedAlchemyGold/GetAdjustedCubeExp que retornam valores multiplicados. Sem validação server-side, permite editar escalas localmente (Cheat Engine) para multiplicar ouro/exp arbitrariamente.
- **verificar:** Procure chamadas de GetAdjustedAlchemyGold/GetAdjustedCubeExp → confirme se valores vão direto pro servidor sem recalcular. Teste editando AlchemyGoldScale com memory patcher; verifique se servidor aceita recompensas dobradas.
- **correção:** Servidor armazena tabela IMUTÁVEL {ItemLevel → {ouro, exp}}. Cliente envia apenas itemID + qnt coletada. Servidor recalcula rewards de forma independente com validação de plausibilidade (nível vs drops esperados).

## [ALTA] Client-authority Pet Stat Modification
- **classe:** `TaskbarHero.Data.PetStatInfoData`
- **problema:** Se valores são salvos/enviados sem validação server-side, atacante edita arquivo de save ou memoria pra modificar estatísticas de pet arbitrariamente (dupe, stat overflow, aplicar mods indevidos).
- **verificar:** Procurar: (1) métodos públicos set em Value/StatType; (2) serialização sem checksum; (3) servidor aceita dados do cliente sem re-validar; (4) sem rate-limit no servidor pra modificações; (5) MODTYPE pode ser string arbitrária.
- **correção:** Server-side validation: servidor valida CADA mudança de stat, recalcula a partir de source-of-truth (banco de dados), rejeita valores fora de range, implementa rate-limit, assina dados com HMAC-SHA256.

## [ALTA] Client-Authority em Multiplicadores de Economia (AlchemyGoldScale/CubeExpScale)
- **classe:** `TaskbarHero.Data.ItemTypeScaleInfoData`
- **problema:** A classe armazena multiplicadores de recursos (ouro, XP) que parecem ser editáveis no cliente e possivelmente calculados client-side. Se o servidor confia nestes valores ao processar recompensas, um atacante pode usar Cheat Engine para modificar escalas em memória, multiplicando recursos infinitamente.
- **verificar:** (1) Abrir jogo com Cheat Engine; (2) Localizar AlchemyGoldScale ou CubeExpScale em memória; (3) Editar valor (ex: 1.0 → 100.0); (4) Completar uma ação que gera recurso → Se recebeu 100× mais recurso, vulnerabilidade confirmada. (5) Tentar vender items duplicados no marketplace/Steam trading.
- **correção:** Servidor NUNCA deve confiar em escalas do cliente. Recalcular TODAS as escalas server-side baseado em configuração autorizada (item level, player tier, progression). Validar que recurso_recebido ≈ base_reward × server_calculated_scale, com tolerância mínima (±5%). Implementar rate-limit server-side por player/hora. Log de auditoría para detectar anomalias.

## [ALTA] Client-Authority de Status Effects — Buffs sem validação servidor
- **classe:** `TaskbarHero.Data.StatusEffectInfoDataMap`
- **problema:** StatusEffectInfoDataMap armazena buffs/debuffs localmente. Se o servidor não valida e persiste cada efeito aplicado (duração, stacks, intensidade), o cliente pode criar/estender indefinidamente, bypass de cooldowns, ou gerar recursos em excesso (especialmente crítico em jogo idle onde buffs amplificam farm). Economia tradeable fica comprometida.
- **verificar:** Conferir se /api/claim-rewards, /api/save-progress, /api/trade validam PERMANENTEMENTE no servidor quais buffs estão ativos. Se duração/stacks são calculados apenas no cliente com `ClientTime.deltaTime`, é vulnerável a: memory edit, date-skew, serialization tamper.
- **correção:** Servidor é source-of-truth: manter tabela `player_buffs(player_id, effect_id, applied_at_timestamp, duration_seconds)`. Cliente aplica apenas efeito visual. A cada request crítico (claim, trade), servidor recalcula `active_buffs = WHERE applied_at + duration > now()`. Nada do cliente é confiável para rewards.

## [ALTA] Síntese (Crafting) validada no Cliente — Duplicação e Manipulação de Drops
- **classe:** `TaskbarHero.Data.SynthesisDropInfoDataMap`
- **problema:** Mapeamento de síntese→drops no cliente permite modificação de taxa de sucesso, duplicação de itens via múltiplas tentativas, e garantia de drops específicos. Em jogo com Steam trading, dupes causam inflação econômica real.
- **verificar:** 1) Localizar SynthesisDropInfoDataMap em savefile/memória (Cheat Engine). 2) Editar taxa de sucesso para 100% ou modificar drops esperados. 3) Testar se síntese funciona sem consumir recursos / sem cooldown server-side. 4) Verificar se drops são sorteados no cliente ou validados pelo servidor.
- **correção:** Mover lógica de síntese pro servidor: servidor recebe 'pedir síntese', valida recursos/cooldown, sorteia sucesso+drops, retorna resultado. Cliente nunca calcula nem vê dados crus de mapeamento. Usar rate-limit server-side, não client-side.

## [ALTA] Client-Authority em Síntese de Itens (EItemSynthesisType não validado)
- **classe:** `TaskbarHero.Data.EItemSynthesisType`
- **problema:** Se o cliente envia o tipo de síntese ao servidor, um atacante pode forjar requisições com valores inválidos (out-of-range, negative, ou types inexistentes) para bypassar lógica de custo, cooldown ou raridade. Enum definitions no cliente são trivialmente contornáveis.
- **verificar:** Interceptar requisição de síntese; enviar EItemSynthesisType=999 ou -1; se servidor processa sem erro, validação falhou. Monitorar logs do BaaS por tentativas de enum inválido.
- **correção:** Servidor valida SEMPRE: `if (!Enum.IsDefined(typeof(EItemSynthesisType), clientType)) return HttpStatusCode.BadRequest;`. Rate-limit, custo e cooldown verificados apenas no servidor, nunca confiados do cliente. Usar whitelist explícita de tipos permitidos por classe de item.

## [ALTA] Síntese (Crafting) com Autoridade do Cliente — Drops/Itens Não-Validados
- **classe:** `TaskbarHero.Data.SynthesisDropInfoData`
- **problema:** A classe SynthesisDropInfoData é um DTO que centraliza ItemLevel, GRADE, RecipeTier e DropKey. Se o servidor confia nos valores enviados pelo cliente sem re-validar contra a receita original e tier permitido, um jogador pode editar DLL/memória pra craftar items de nível/raridade arbitrária ou bypass de receitas.
- **verificar:** 1. Inspecionar fluxo: cliente faz POST /craft/{SynthesisDropKey} com SynthesisDropInfoData completo? 2. Servidor valida se ItemLevel ≤ MaxLevelForRecipeTier e GRADE ≤ MaxGradeForTier, ou apenas insere o item como-é? 3. Verificar se existe server-side lock que força recalcular/rederivir os valores a partir de SynthesisRecipeInfoData.
- **correção:** Servidor nunca deve confiar em ItemLevel/GRADE/RecipeTier vindos do cliente. Em vez disso: (a) Cliente envia APENAS o SynthesisDropKey; (b) Servidor carrega SynthesisRecipeInfoData e SynthesisDropInfoData de uma tabela imutável (config protegida); (c) Valida material/tier do jogador vs receita; (d) Calcula ItemLevel/GRADE via fórmula determinística (não reversa-engenhável); (e) Gera item no servidor e envia apenas o item final pro cliente.

## [ALTA] Client-Authority em StatModGroupInfoData + Save Tampering
- **classe:** `TaskbarHero.Data.StatModGroupInfoData`
- **problema:** A classe armazena buffs/debuffs com metadados completamente sob controle do cliente (bgqa=valor, bgqb=duração, bgpz=ID). Sem validação server-side, jogador edita save local alterando multiplicadores de stat, duração infinita, ou injetar IDs inválidos de modificadores.
- **verificar:** 1) Localize arquivo de save (Unity: %appdata%/[GameName] ou ~/.local/share/[Game]). 2) Exporte StatModGroupInfoData (provavelmente JSON/MessagePack). 3) Edite bgqa para 999 (se é multiplicador) ou bgqb para -1 (duração). 4) Reload jogo. 5) Se buffs aplicam sem revalidação server-side: CONFIRMADO. 6) Verifique se servidor está realmente re-authenticating dados ou apenas confiando em cliente.
- **correção:** **CRITICO — Server-side only:** (1) Tabela imutável no servidor com IDs de buffs pré-autorizados + limites de valores (ex: 'ATK_BUFF_LEVEL_1' = multiplicador fixo 1.1, não editável). (2) Servidor VALIDA: 'bgpz está em whitelist? bgqa está dentro do intervalo permitido para esse buff? bgqb < duração máxima?' (3) Armazena dados de buff NO SERVIDOR, não no cliente. (4) Save local apenas armazena referência (buff_id + timestamp aplicado), servidor controla lógica de expiração. (5) Assinatura de save: HMAC/RSA da estrutura completa com chave privada servidor. (6) Anti-dupe: servidor verifica se mesmo buff já ativo antes de aplicar.

## [ALTA] Client-Authority + Save Tampering em HunterQuickLoader
- **classe:** `TaskbarHero.Combat.HunterQuickLoader`
- **problema:** Campo 'bgtf' com nomes ofuscados e métodos sem corpo visível sugerem padrão de validação apenas client-side. Típico: servidor envia valor → cliente edita → servidor confia. Combinado com save tamperable, permite dupe de ouro/itens.
- **verificar:** ['Ver corpo de mlz, mma, mni, mnj — fazem lógica de jogo ou apenas UI?', 'Verificar se servidor re-valida bgtf contra player stats server-side', 'Testar: editar valor local (PlayerPrefs ou save file) → servidor aceita?', 'Rate-limit: existem cooldowns server-side ou apenas client-side?']
- **correção:** ['Server-side re-validação: antes de aceitar ação, consultar player.gold actual no servidor', 'HMAC sign + server verify: assinar bgtf com chave secreta', 'Rate-limit no servidor, não no cliente', 'Anti-cheat: validação de deltas (se mudança de ouro é implausível, reject)']

## [ALTA] Client-Authority em Combate + Rate-Limit Client-Side
- **classe:** `TaskbarHero.Combat.KnightRevengeAtk`
- **problema:** KnightRevengeAtk, como classe de combate, participa do fluxo kill→baú que é inteiramente client-authoritative. O servidor não valida kills contra wall-clock real, permitindo manipulação via Time.timeScale. Rate-limit (StageBoxRequestAbuseGuard) é contornável via MelonLoader/Harmony.
- **verificar:** 1. Decompile KnightRevengeAtk com dnSpy/Il2CppInspector; 2. Procure por chamadas a UI.StageBox, InventoryProcessBoxRequest.ToJson ou eventos de kill; 3. Verifique se há sincronização com servidor de timestamp/taxa real
- **correção:** 

## [ALTA] Client-Authority em Dano de Ataque (KnightBaseAtk) — Edição de Memória + Rate-Limit Bypass via TimeScale
- **classe:** `TaskbarHero.Combat.KnightBaseAtk`
- **problema:** O campo `bgth` de KnightBaseAtk (damage/state do ataque) reside no cliente (Assembly-CSharp.dll) e é editável em memória via Cheat Engine / MelonLoader. O servidor confia no evento de dano reportado sem validação de plausibilidade (taxa contra relógio real, DPS teórico máximo). Combinado com TimeScale bypass (vide disclosure), permite ~6×+ amplificação de farm ilimitado.
- **verificar:** 1) Usar Cheat Engine pra editar `bgth` (ou injetar hook MelonLoader); 2) disparar ataque com valor inflado; 3) observar se servidor aceita dano sem rejeitar; 4) confirmar exp/ouro/baú gerado em taxa anormalmente alta. 5) Verificar logs server de 'kills/seg' — devem exceder max físico da fase.
- **correção:** 

## [ALTA] Client-Authority Ressurreição + Rate-Limit Apenas Cliente
- **classe:** `TaskbarHero.Combat.PriestResurrection`
- **problema:** Habilidade de ressurreição valida cooldown/custo/pré-condições apenas no cliente, permitindo bypass via packet replay, dupe de efeito, ou reutilização sem mana. Servidor sem validação independente é authority gap crítico.
- **verificar:** (1) Intercepte requests com Fiddler/mitmproxy; (2) repita mensagem de ressurreição 10× sem delay; (3) modifique savedata localmente para resetar cooldown; (4) confirme se servidor aceita sem validar pré-condições (aliado morto? custo debitado? combate ativo?); (5) teste rate-limit ausente server-side.
- **correção:** Servidor deve validar TODAS: (a) último timestamp de ressurreição + cooldown global; (b) mana/recurso realmente disponível; (c) aliado ID existe e status=dead; (d) combate ativo; (e) rate-limit hard com token bucket; (f) idempotência via transação UUID; (g) log de tentativas suspeitas.

## [ALTA] Client-Side Heal Amount Não Validado
- **classe:** `TaskbarHero.Combat.PriestHeal`
- **problema:** GetHealAmount() calcula cura baseado em stats do cliente sem validação servidor. Atacante pode modificar assembly (remover obfuscação IL) ou injetar valor de cura aumentado no evento Execute enviado ao servidor.
- **verificar:** Interceptar tráfego de execute_heal_action; tentar enviar heal_amount > base stat * 2; verificar se servidor aceita; testar se mesmo com HP baixo consegue curar aliado infinitamente.
- **correção:** Servidor SEMPRE recalcula heal_amount from scratch usando stats auditados do alvo (banco de dados). Nunca confie em valor enviado pelo cliente. Valide: cooldown server-side, recurso consumido (mana), alvo está vivo e aliado, cap de healing por tick.

## [ALTA] Client-Authority Damage + Movement Validation (ActBossDashAttack)
- **classe:** `TaskbarHero.Combat.ActBossDashAttack`
- **problema:** A classe executa dano, validação de cooldown e cálculo de deslocamento no CLIENTE. Um hacker pode editar parâmetros de movimento (velocidade/distância), multiplicar dano, bypassar cooldown, e explorar alcance inválido.
- **verificar:** 1. Decompile o APK com dnSpy; 2. Localize mlz (Execute) — vê se ApplyDamage é chamado cliente-side?; 3. Procure por CanExecute — o cooldown é checado apenas com Time.time local?; 4. Teste: pause o jogo, edite bgxi campo em Cheat Engine, retome — o ataque acontece fora de cooldown?; 5. Teste movimento: com Cheat Engine, aumente velocidade do dash 10x ou distância 1000x — atravessa paredes?
- **correção:** 1. REMOVA todo cálculo de dano do cliente. Envie apenas 'ExecuteDashAttack()' para servidor; 2. Servidor valida: cooldown (compara ServerTime vs lastAttackTime), distância (verifica posição real do boss vs alvo), CALCULA dano base (não cliente), aplica hit; 3. Cliente recebe apenas resposta do servidor ('DamageApplied: 150' ou 'Cooldown active') e renderiza efeitos; 4. Rejeite qualquer evento do cliente que contenha valores numéricos (damage, movement delta, cooldown override); 5. Implemente replay validation: servidor re-calcula toda attack sequence localmente para detectar tampering

## [ALTA] Client-Authority Combat Damage + Target Selection
- **classe:** `TaskbarHero.Combat.ActBossMonsterStab`
- **problema:** Seleção de alvo e cálculo de dano executados no cliente. Um atacante pode modificar valores de dano, escolher alvos inválidos (out-of-range, aliados, invulneráveis), ou enviar múltiplas ações simultâneas sem validação server-side.
- **verificar:** Capturar request HTTP/WebSocket de ActBossMonsterStab; variar parâmetros (targetId, damage, bgxy coordinates). Enviar com alvo fora do range, com boss morto, ou spam 100× em 1s. Se servidor aceita sem validação, é vulnerável.
- **correção:** No servidor: (1) Validar que boss/attacker existe e está vivo. (2) Validar target existe e está em range + não é imortal. (3) Recalcular dano server-side (ignore valor do cliente). (4) Rate-limit por actor (1 ataque/X ms). (5) Log todas as ações pra auditoria.

## [ALTA] Client-Authority — Validação Fraca de Ação de Boss
- **classe:** `TaskbarHero.Combat.ActBossMassiveVoidCall`
- **problema:** Classe invoca ação de boss do cliente sem validação robusta server-side. Cliente pode disparar múltiplas vezes, fora de contexto de combate, ou com spawn count/dano dos Void modificados via memory-tampering.
- **verificar:** Testar se servidor valida: (1) combate ativo, (2) turno/timing correto, (3) spawn count regenerado server-side ou editável no cliente via debugger
- **correção:** Servidor deve validar autorização (combate ativo, boss vivo), regenerar spawn count + dano das criaturas server-side (nunca confiar cliente), aplicar rate-limit server-side

## [ALTA] Client-Authority Hit Detection em AoE — Servidor confia em área de ataque do cliente
- **classe:** `TaskbarHero.Combat.MassiveVoidCallArea`
- **problema:** Controladores de AoE em idle-RPG frequentemente executam hit detection (quem foi atingido), dano e rate-limiting no cliente. Se MassiveVoidCallArea calcula quem foi atingido e o servidor apenas aplica o resultado sem revalidar geometria/posições, permite spawn de múltiplas instâncias, dano falso ou bypass de cooldown editando m_effectObjects ou memória.
- **verificar:** 1) Procurar m_effectObjects sendo enviado ao servidor ou usado pra calcular dano; 2) Rastrear RPC/payload quando skill é disparada — servidor valida posições de inimigos e área? ou confia cegamente no cliente? 3) Tentar dispara skill 2× em 1s — cooldown é só local ou servidor rejeita? 4) Inspecionar memória (Cheat Engine) se m_effectObjects é editável em runtime.
- **correção:** 1) Servidor revalida TUDO: posição do player, posição de cada inimigo, se estão realmente no raio. 2) Rate-limit no servidor por timestamp de último uso. 3) Servidor calcula dano — cliente só envia comando 'cast', nunca hits/dano. 4) AntiCheat deve sinalizar tentativas de editar m_effectObjects ou rate de invocação anormal.

## [ALTA] Client-side probability resolution sem validação server-side
- **classe:** `TaskbarHero.Combat.AnimaitionProbabiltyData`
- **problema:** Se o cliente usa `probability` para determinar hit/miss e envia o resultado ao servidor sem recálculo independente, permite edição de `probability=1.0` ou falsificação de eventos de sucesso, convertendo todas as skills em 100% de chance.
- **verificar:** Debugar: cliente envia qual dado ao servidor após skill? (ex: {skillKey, targetId, result:'hit'}) e servidor apenas aplica dano sem re-validar RNG localmente? Se sim, é explorado via Cheat Engine (editar probability ou interceptar request).
- **correção:** Servidor recalcula RNG + probabilidade com seed server-side; cliente recebe APENAS resultado final (MISS|HIT|CRIT). Nunca deserializar ou confiar em probability do cliente.

## [ALTA] Client Authority em Timers de AOE + Rate-Limiting Ausente
- **classe:** `TaskbarHero.Combat.RepeatingAOESpawner`
- **problema:** RepeatingAOESpawner permite que o cliente controle m_spawnInterval e m_totalDuration sem validação server-side, permitindo gerar AOEs em frequência arbitrária (ex: interval=0) ou duração ilimitada, causando DoS. Posições (m_spawnDirection, m_spawnRangeRaw) também são calculadas no cliente sem validação de bounds/autoridade.
- **verificar:** 1) Capturar tráfego de rede durante skill ativo e alterar m_spawnInterval para 0 → verifica se AOEs spawnam sem throttle. 2) Verificar se servidor recalcula timers ou apenas aceita valores do cliente. 3) Tentar enviar múltiplas StartSpawning em sequência rápida e medir se servidor aplica rate-limit.
- **correção:** Server-side: (1) Armazenar intervalo/duração definido no servidor (SkillInfoData), não confiar em cliente. (2) Aplicar throttle server-side com timestamp validado (ex: require 'now - lastSpawnTime >= serverDefinedInterval'). (3) Validar posições de AOE contra bounds/alcance legais antes de aplicar dano. (4) Ignorar pacotes de spawn que violem rate-limit. Usar servidor como fonte de verdade para timing.

## [ALTA] Client-side spawn interval para projéteis de congelamento — DoS econômico
- **classe:** `TaskbarHero.Combat.SnowstormArea`
- **problema:** O campo ProjectileSpawnIntervalTime é controlado e executado apenas no cliente via Update. Um atacante pode modificar este intervalo em memória ou via decompilação+recompilação, fazendo spawnar centenas de projéteis por frame em vez de por intervalo, causando dano massivo indetectável. Sem validação server-side da frequência real de aplicação do efeito de congelamento, um cliente modificado pode congelar o mesmo inimigo 100× por segundo.
- **verificar:** 1) Fazer dump de memória do jogo enquanto SnowstormArea está ativo e confirmar que ProjectileSpawnIntervalTime é um float editável em tempo real sem proteção AntiCheat. 2) Modificar o intervalo (ex: de 0.5s para 0.001s) e verificar se o servidor aceita o aumento de dano aplicado sem questionar a taxa de eventos. 3) Verificar logs do servidor: há rate-limiting ou validação de quantos eventos de 'ApplyFreezing' um cliente pode reportar por segundo?
- **correção:** Mover a lógica de spawn para servidor: cliente recebe prefab visual apenas (posição, animação), servidor valida interval, calcula dano real, aplica efeito de congelamento e envia feedback ao cliente. Rate-limit server-side: máximo N events de congelamento por player por segundo. Assinar timestamps de eventos com chave de sessão servidor para detectar replaydrops.

## [ALTA] Turreta executa ataques client-side sem validação servidor
- **classe:** `TaskbarHero.Combat.MeleeTurret`
- **problema:** Alcance (m_detectRangeRaw), delays (m_attackDelayTime, m_spawnDelayTime), contagem de ataques (m_maxAttackCount) e tipo-alvo (m_targetDamageableType) são resolvidos localmente. Atacante pode editar memória para: remover delays (spam ataque), aumentar alcance infinitamente, contornar limite de ataque, atacar NPCs/estruturas proibidas. Sem validação servidor, dano é aplicado baseado apenas em evento clientside.
- **verificar:** 1) Monitorar tráfego: RPC de ataque envia apenas confirmação ou envia hit+damage+target? 2) Memory-cheat (Cheat Engine): editar m_attackDelayTime→0, verificar se ataque acelera. 3) Editar m_detectRangeRaw→999, verificar se turreta detecta além do limite esperado. 4) Se não há RPC validando cada ataque no servidor, é client-authority.
- **correção:** Servidor deve validar CADA ataque: distância real entre turreta e alvo (serverside), cooldown via servidor, tipo-alvo permitido, cap de dano. Cliente envia apenas comando 'atacar'; servidor aprova/nega e aplica resultado. Usar rollback se client reportar hit que servidor rejeitou.

## [ALTA] Client-Authority em Combat Stats (AttackSpeed, DetectRadius)
- **classe:** `TaskbarHero.Combat.TurretSkillData`
- **problema:** Se AttackSpeed, DetectRadius e Target são armazenados e controlados apenas no cliente, um jogador pode modificar memória/bytes para atacar muito mais rápido, detectar inimigos de raio infinito ou alvos inválidos. Em idle-RPG multiplayer, isso quebra a economia (farm infinito) e resulta em dupe.
- **verificar:** Inspecionar TurretSkillData.cs para ver se é [Serializable] público; verificar se valores são enviados pro servidor ANTES de aplicar ataque, ou se servidor apenas ACEITA o evento 'ataque'.
- **correção:** 

## [ALTA] Hit Detection e Status de Congelamento Client-Side
- **classe:** `TaskbarHero.Combat.Projectile.FrostBolt`
- **problema:** Se FrostBolt usa OnCollisionEnter para detectar impacto e envia evento 'congele o inimigo X' pro servidor, o cliente controla quem é atingido. Servidor não valida distância, LoS ou se alvo era alcançável → bypass de range, congelamento em qualquer NPC, possível dupe se aplicado múltiplas vezes.
- **verificar:** 1) Capturar tráfego de evento quando FrostBolt impacta; 2) Verificar se servidor valida posição do atirador/alvo; 3) Tentar lançar contra alvo fora de range (lua/cheat engine); 4) Tentar congelar o mesmo NPC 1000× em 1s.
- **correção:** Servidor autoriza HIT: calcular trajetória do projétil, validar distância/LoS, aplicar dano+congelamento server-side com timeout. Cliente só anima/áudio. Rate-limit 1 FrostBolt/seg por jogador.

## [MEDIA] Client-Authority em ItemSynthesisType
- **classe:** `TaskbarHero.SynthesisItemResource`
- **problema:** A classe armazena ItemSynthesisType (tipo de síntese) no cliente. Se este valor for enviado ao servidor sem validação (ou com validação apenas client-side), permite forjar sínteses não autorizadas ou de itens inacessíveis.
- **verificar:** 1) Rastrear donde ItemSynthesisType é enviado (POST /synthesis ou similar). 2) Verificar se o servidor re-valida: o tipo existe? o jogador tem permissão? tem os recursos necessários? Se o servidor confiar no cliente ou validar apenas format (não negócio), é vulnerável.
- **correção:** Servidor deve manter lista authoritative de sínteses permitidas por classe/nível. Ao receber requisição de síntese, validar: (a) tipo existe no whitelist, (b) jogador tem level/items/resources, (c) re-calcular custo/resultado server-side, nunca confiar em valores do cliente.

## [MEDIA] Client-Side Inventory Count Display (Weak Client-Authority Pattern)
- **classe:** `TaskbarHero.CubeCraftingSlot`
- **problema:** CubeCraftingSlot exibe text_count sem indicação de validação server-side. Em idle-RPGs com trading, valores de inventário no cliente são trivialmente editáveis via Unity Inspector. Se o servidor aceita a contagem reportada pelo cliente no momento do craft, dupe é possível.
- **verificar:** 1) Editar text_count via Inspector/Cheat Engine antes de craftar; 2) Tentar craftar com quantidade > inventory real; 3) Verificar se servidor rejeita ou aceita a ação. Se aceita = vulnerabilidade confirmada.
- **correção:** Servidor deve: 1) Manter source-of-truth de inventário no backend; 2) Validar 'player realmente tem X items?' antes de each craft action; 3) Deduzir recursos do estado server, não do client; 4) Se cache local necessário, criptografar/assinar; 5) Implementar rate-limiting server-side.

## [MEDIA] Falta de validação server-side em coleta de items do correio (MailSlot.OnClaimButtonClicked)
- **classe:** `TaskbarHero.MailSlot`
- **problema:** OnClaimButtonClicked envia um request de coleta ao servidor (provável POST /api/mail/claim) sem garantias server-side de que o item pertence ao jogador, não foi já coletado, ou há rate-limiting. Um cliente malicioso pode clicar múltiplas vezes no botão, modificar o request (ex.: forjar mailId), ou enviar requests em paralelo para duplicar items.
- **verificar:** 1) Inspecionar MailSlot.OnClaimButtonClicked para verificar se usa HTTP interceptor/proxy para redirecionar POST; 2) Tentar clicar rapidamente (10+× em <1s) no mesmo item e verificar se backend aceita coleta múltipla; 3) Usar MelonLoader para hotpatch StageBoxRequestAbuseGuard e remover rate-limit local, depois coletar — se funciona, backend não valida; 4) Forjar mailId em POST /api/mail/claim com ID de outro jogador (se IDs sequenciais/previsíveis).
- **correção:** 1) Backend valida idempotência: verificar se mailId já foi reivindicado (flag claimed_at != null, ou remove de lista após coleta); 2) Rate-limit server-side por jogador + endpoint (ex.: max 10 claims/min por conta); 3) Validar ownership: backend confirma que mailId pertence a userId antes de processar; 4) Usar token/nonce one-time no POST para evitar replay; 5) Não depender de StageBoxRequestAbuseGuard (client-side) como única barreira.

## [MEDIA] Client-Authority em EStageState + Rate-limit apenas cliente
- **classe:** `TaskbarHero.EStageState`
- **problema:** A máquina de estados de fase (spawn → battle → reorganização) é resolvida no cliente e reporta eventos de kill/baú ao servidor sem validação de taxa contra relógio real. Rate-limit protetor (StageBoxRequestAbuseGuard) é apenas client-side e sensível a manipulação de Time.timeScale.
- **verificar:** 1. Acelerar jogo com Time.timeScale=6 2. Farm de baús completar ~6× mais rápido 3. Verificar se servidor valida kills/sec contra wall-clock (não deve) 4. Confirmar que desativar StageBoxRequestAbuseGuard via mod (HarmonyX) remove proteção de taxa
- **correção:** Mover autoridade para servidor: validar taxa de kills contra wall-clock real, rejeitar clears abaixo do tempo mínimo teórico (baseado em HP mob + DPS máximo), fazer concessão de baú server-authoritative com derivação do progresso validado, não apenas de eventos de kill crus.

## [MEDIA] Source-Type Spoofing — Cliente pode falsificar origem de itens
- **classe:** `EItemGetSourceType`
- **problema:** Um enum de origem (`EItemGetSourceType`) num jogo com Steam trading é crítico. Se o cliente declara a origem de um item (ex: 'MonsterDieReward' vs 'BackendPost') sem validação server-side, um atacante pode falsificar itens, bypassar cooldowns/rate-limits originários do cliente, ou criar dupes ao transferir pro Steam com origem falsa.
- **verificar:** 1) Capturar tráfego do cliente ao obter/tradar itens → procurar por campos type/source não-assinados; 2) Modificar a origem localmente (no save do jogo ou via Cheat Engine) e mandar pro servidor; 3) Se servidor aceita qualquer origem sem validar contra log do evento real (loot drop, mail, etc), há dupe/bypass.
- **correção:** Servidor deve: (a) nunca confiar no campo `source` do cliente — recalcular a origem baseado em audit log interno (did player kill monster X at time Y?, did we send a mail?, etc); (b) assinar/encriptar dados críticos no cliente; (c) validar origem antes de qualquer operação de trade/steam; (d) rate-limit e dedupagem server-side.

## [MEDIA] Event-driven attribute sync sem validação server-side
- **classe:** `TaskbarHero.AttributeCache`
- **problema:** Callbacks OnAttributeChange notificam mudanças de atributos (HP, MP, ATK, DEF, etc) disparados do cliente; se o servidor aplica efeitos de skills/buffs baseado nestes eventos sem validar contra estado esperado do servidor, permite amplificação de dano ou duração anômala de buffs.
- **verificar:** Capturar tráfego de eventos quando skill é usada; confirmar se servidor aplica multiplicador de ATK sem verificar se o skill foi realmente castado dentro da duração/cooldown válido. Acelerar timeScale e checar se buff persiste além do esperado no servidor.
- **correção:** Mover autoridade de stat/buff para servidor: servidor calcula ATK = baseATK + buffs_validados(skill_history_server); cliente apenas renderiza. Rejeitar updates de atributo que não derivam de ação verificada no servidor (kill, spell cast, buff aplicado pelo servidor).

## [MEDIA] Client-Authority em Validação de Inventário
- **classe:** `TaskbarHero.EBackendPostRewardResult`
- **problema:** A presença de um enum específico 'NotEnoughInventorySpace' sugere que validação de lógica de negócio (capacidade do inventário) está sendo feita no cliente. Isso permite que um atacante contorne a validação editando o cliente para ignorar a verificação de espaço ou manipulando requisições para aceitar recompensas sem esse check.
- **verificar:** Verifique se: (1) o cliente valida espaço em inventário ANTES de enviar requisição de recompensa ao servidor; (2) se um cliente modificado pode enviar 'ReceiveReward' mesmo com inventário cheio; (3) se o servidor novamente valida o espaço antes de aplicar a recompensa.
- **correção:** Mova toda validação para o servidor: o servidor recebe 'ReceiveReward', tenta adicionar ao inventário, e responde com NOVO ESTADO DO INVENTÁRIO (ou erro genérico 403). Nunca confie em checks do cliente. O enum pode ficar apenas como tipo de resposta, mas a lógica de 'há espaço?' deve ser 100% server-side e não executável pelo cliente.

## [MEDIA] Client-side DLC ownership caching + ausência de re-validação server-side
- **classe:** `TaskbarHero.VerifyDlcResult`
- **problema:** A verificação de propriedade de DLC é feita uma vez contra servidor e cacheada localmente em disco. O campo isSuccess é uma flag booleana simples — um mod loader pode setta-la como true sem possuir o DLC, permitindo acesso a conteúdo protegido e itens trade.
- **verificar:** MelonLoader + HarmonyX: interceptar VerifyDlcResult.GetDlcStatus(), forçar retorno true; tentar acessar content/ability protegida por DLC; servidor não rejeita = vulnerável.
- **correção:** Validar DLC no servidor a cada acesso (ex.: POST /api/use-dlc-feature com signature), nunca confiar em flag booleana local. Mover autoridade de DLC para backend (tipo role-based como StageBoxRequestAbuseGuard deveria ter feito).

## [MEDIA] Client-Authority: Classificação de inimigo aceita sem validação server-side
- **classe:** `TaskbarHero.Log.EMonsterLogType`
- **problema:** Enum de logging do cliente que classifica tipos de inimigos (Monster/Boss/ActBoss). Se o servidor confia nessa classificação para distribuir recompensas, XP, ou loot, o cliente pode falsificar tipo e colher ganhos de ActBoss derrotando Monster normal. Típico em idlers com economia aberta.
- **verificar:** 1) Capturar network traffic de combat win → procurar por enum value ou nome de tipo de inimigo no payload. 2) Modificar para ActBoss valor antes do send. 3) Verificar se recompensa muda (loot rarity, XP multiplier, ouro). 4) Checar se servidor re-valida o tipo baseado em inimigo real no DB.
- **correção:** Servidor NUNCA acessa `EMonsterLogType` direto. Em vez disso: (1) Cliente envia apenas `enemyId`. (2) Servidor lookup do tipo real via DB. (3) Recompensas calculadas server-side. Logging pode usar tipo server-computed.

## [MEDIA] Client-Authority em Efeitos de Status
- **classe:** `TaskbarHero.Status.StatusEffect.StatusEffectType`
- **problema:** A enumeração sugere padrão onde cliente envia eventos de efeito de status (Freeze, Ignite, etc.) e servidor os aplica sem validação. Em idle-RPG com client-heavy resolution, permite explorador aplicar qualquer efeito em qualquer alvo (congelar boss, stunear adversário indefinidamente) ou ignorar delays/cooldowns.
- **verificar:** Debugar tráfego cliente→servidor durante combate: procurar por `StatusEffectType.Freeze` ou similar em payloads. Se aparecer fora de uma RPC assinada/validada, é vulnerável. Testar: enviar `ApplyStatusEffect(enemyId, StatusEffectType.Stun)` com token inválido — se stun aplicar mesmo assim, confirmado.
- **correção:** Servidor valida TUDO: (1) Jogador tem skill/spell que aplica esse efeito? (2) Skill está em cooldown? (3) Jogador tem recursos (mana/stamina)? (4) Alvo pode receber (imunidade, já afetado?)... Enum fica só como feedback visual client-side; autoridade é sempre server.

## [MEDIA] Client-Authority em BuffStackType — Stackability Spoof
- **classe:** `TaskbarHero.Status.BuffDebuff.BuffStackType`
- **problema:** Se o servidor aceita o tipo de stack (OnlyOne vs Stackable) como enviado pelo cliente, um atacante pode modificar a classe em memória (memory patching, Cheat Engine, decompile+recompile) para forçar buffs exclusivos a se tornarem stackáveis, permitindo múltiplas instâncias simultâneas do mesmo efeito (ex.: +5 Força aplicado 10× = +50).
- **verificar:** 1) Interceptar tráfego de rede (Fiddler/mitmproxy) quando buff é aplicado — se BuffStackType é transmitido do cliente. 2) Usar Cheat Engine para modificar `value__` em tempo real e verificar se efeitos se acumulam. 3) Verificar logs do servidor — ele respeita o tipo vindo do cliente ou tem validação hardcoded?
- **correção:** Servidor mantém matriz immutável [BuffId → StackType] gerada server-side; IGNORA qualquer StackType vindo do cliente. Validação no Apply: se buffer X é OnlyOne e já ativo, rejeita nova aplicação. Logs auditam tentativas de violação.

## [MEDIA] Client-Authority em Cálculo de Modificadores de Stat
- **classe:** `TaskbarHero.Skill.ModStatExchangeType`
- **problema:** O enum ModStatExchangeType expõe diferentes escalas de divisão (Raw_Divide1000, Raw_Divide100, Divided) que parecem ser configuráveis/aplicáveis pelo cliente. Se o servidor confia no tipo de divisão escolhido pelo cliente sem revalidar, um atacante pode escolher Raw_Divide100 ao invés de Raw_Divide1000 para obter 10× mais bônus — ou tamperar o valor bruto antes da operação de troca.
- **verificar:** 1) Interceptar requisição de exchange/modificação de stat (ex: /api/applyModifier). 2) Clonar a payload, alterar field 'modType' de Raw_Divide1000 para Raw_Divide100. 3) Se o servidor aceita e aplica bônus 10× maior, está vulnerável. 4) Verificar se servidor recalcula/valida o valor bruto + divisor antes de persistir.
- **correção:** No servidor: (a) Nunca confie no tipo de modificador enviado pelo cliente. Armazene meta-dados da transação no servidor (qual modificador é válido para este item/personagem). (b) Valide: tipo de modificador ∈ {tipos pré-aprovados para este item}, valor final ∈ [min_esperado, max_esperado]. (c) Recalcule valor_final = valor_bruto / escala_servidor (não escala do cliente). (d) Log auditado de quem aplicou qual modificador quando.

## [MEDIA] Receitas não-desbloqueadas selecionáveis sem validação server-side
- **classe:** `TaskbarHero.UI.MainRecipeComboBoxButton`
- **problema:** Se m_unlockEffect é apenas visual e não há validação server-side de desbloqueamento ao confirmar crafting, o cliente pode selecionar/usar receitas premium que não deveria ter acesso, gerando items ilicitamente.
- **verificar:** 1) Interceptar requisição de crafting ao servidor após selecionar receita locked via MainRecipeComboBoxButton. 2) Se InventoryProcessBoxRequest contém apenas recipe_id sem verificação server-side do status unlock (via player level/progression), exploração é viável. 3) Verificar se há validação de CraftingRecipeInfoData.Validate() no backend antes de aplicar ApplyRecipe.
- **correção:** Servidor DEVE validar antes de aplicar crafting: (a) receita existe + é valid; (b) receita está desbloqueada para este player (level, progression, quests); (c) player tem materiais necessários (não confiar em CanCraft client-side). Adicionar rate-limit server-side por player+receita. Usar validação server-authoritative, não client-side checks como m_unlockEffect.

## [MEDIA] Client-Authority em Equipamento de Skills — Possível Abuse de Slots
- **classe:** `TaskbarHero.UI.ActiveSkillEquipPanel`
- **problema:** O método lfe (clique em slot) transmite decisão de equipar skill PELO CLIENTE. Sem validação server-side de 'esse herói tem esse skill?', 'requisitos atendidos?', 'existe skill com esse ID?', o servidor pode receber e aplicar equipamentos inválidos ou bloqueados.
- **verificar:** Interceptar POST/WebSocket para EquipSkill: (1) equipar skill que o herói não possui; (2) equipar skill de nível acima; (3) equipar skill inexistente (ID -1 ou 99999); (4) spam equipar/desequipar 100× em <1s. Se server aceita sem reject, vuln confirmada.
- **correção:** Server: (1) validar `players[uid].skills.contains(skillId)` antes de aplicar; (2) validar requisitos (nível, classe); (3) implementar rate-limit por conta (máx 1 equipamento/seg); (4) não confiar em cache cliente (bfzp/bfzq) — re-validar skills autorizados a partir do save persistente.

## [MEDIA] Client-Authority: Cache dessincronizado + refund sem validação servidor
- **classe:** `TaskbarHero.UI.AttributeSlot`
- **problema:** Campo _cache armazena valor de atributo/nível no cliente; método SetLevel/PlayRefundEffect podem ser invocados sem validação server-side de pré-requisitos. Se refund/levelup são confirmados ao servidor sem checar se pontos suficientes existem, permite dupe de upgrades.
- **verificar:** 1) Capturar tráfego servidor: chamar /upgrade-attribute sem pontos suficientes — servidor rejeita? 2) Chamar PlayRefundEffect/SetLevel via reflection/mod — persiste após reload? 3) Verificar se servidor valida 'level antigo' antes de aplicar refund.
- **correção:** Server-side: (1) cada upgrade/refund enviado deve validar pontos, nível máximo, e se o pré-requisito do atributo foi atingido. (2) cache cliente é apenas leitura — derivar estado real do servidor. (3) rejeitar upgrades cuja sequência timeline não faz sentido (ex.: downgrade pra level 5 depois upgrade pra 10 em 100ms). (4) não depender de UI state (isSelect, _cache) para lógica; enviar apenas intent+IDs, validar no backend.

## [MEDIA] Client-Authority Cooldown Bypass via StatusPanelSkillSlot
- **classe:** `TaskbarHero.UI.StatusPanelSkillSlot`
- **problema:** StatusPanelSkillSlot.ApplyCanEquipEffect mostra disponibilidade baseada em estado local (IsReady/cooldown de SteamSlotData). Se servidor não re-valida o timer ao receber CastSkill, atacante edita memória do cliente (o_cooldownUntil) para resetar cooldown e ativa skills rapidamente, bypassando rate-limit de servidor.
- **verificar:** (1) Capturar requisições cliente→servidor durante skill cast via proxy; (2) Verificar se servidor valida cooldown novamente ou apenas aplica novo cooldown; (3) Usar Frida/dnSpy para editar o_cooldownUntil no cliente e tentar executar skill — se sucesso, é vulnerable.
- **correção:** Servidor deve SEMPRE revalidar o_cooldownUntil no seu próprio timestamp ao processar CastSkill/ActivateSkill, descartando o valor enviado pelo cliente. Rate-limit também deve ser server-side.

## [MEDIA] Client-Authority em ações de inventário (manipulação de itemID/index)
- **classe:** `TaskbarHero.UI.InventorySlot`
- **problema:** InventorySlot gerencia o índice e estado de item no cliente. O fluxo de EquipItemToGearSlot (método companion) envia ações de equip/move ao servidor sem validação robusta server-side de propriedade do item ou permissão.
- **verificar:** Tentar equipar item com index falso (e.g., -1 ou fora do range) ou manipular itemTooltipTrigger para injetar dados; verificar se servidor valida que itemID existe no inventário do jogador antes de aplicar equip.
- **correção:** Servidor deve validar: (1) itemID existe no inventário do jogador, (2) classe/level do item combina com herói, (3) rate-limit por wall-clock, não client-side. Mover lógica de validação de equip/move para backend; cliente apenas dispara requisição, nunca confia no resultado local sem confirmação.

## [MEDIA] AutoOpenBox susceptível a Time.timeScale acceleration (client-authority)
- **classe:** `TaskbarHero.UI.OpenBoxStats`
- **problema:** A classe UpdateAutoOpenTimer provavelmente decrementa timers com Time.deltaTime, permitindo que Time.timeScale elevado acelere abertura automática de baús. Sem validação server-side da taxa de abertura contra wall-clock real, permite farm acelerado de recompensas.
- **verificar:** 1) Elevar Time.timeScale ~6× em jogo / 2) Medir taxa real de abertura (relógio) vs rate esperado / 3) Confirmar que servidor não rejeita abertura anormalmente rápida. 4) Procurar por validação temporal no endpoint de resposta do baú (lista negra de tempo-de-fase muito curto, taxa kills/baús por segundo de relógio real).
- **correção:** 

## [MEDIA] Abertura automática de caixas sem rate-limit validado no servidor
- **classe:** `TaskbarHero.UI.StageBox`
- **problema:** Método OpenAll pode permitir burst de abertura sem throttle de servidor. Cliente processa animação e efeitos localmente, mas se RPC ao servidor não validar rate-limit, permite bypass.
- **verificar:** Chamar OpenBox/OpenAll 50× em 5 segundos. Servidor aceita todas ou faz rate-limit? Logs revelam quantas transações o backend registrou?
- **correção:** Validar (userID + timestamp + last_open_time) no servidor com cooldown. Rejeitar RPC se OpenBox foi chamado < N segundos atrás. Usar Redis/cache.

## [MEDIA] Client-Authority em Síntese: Tipo não validado server-side
- **classe:** `TaskbarHero.UI.SynthesisItemImageChanger`
- **problema:** SynthesisItemImageChanger recebe SetSynthesisItemType(int type) do cliente sem validação server-side se o tipo está desbloqueado. Padrão idêntico à vulnerabilidade de baú já disclosed: cliente seleciona tipo, cliente valida, servidor executa sem questionar—permitindo crafting de tipos não desbloqueados.
- **verificar:** 1) Interceptar tráfego de síntese (Fiddler/Burp) e enviar tipo=999 inválido/não-desbloqueado. 2) Verificar se servidor aceita e cria item resultado sem validar se tipo é permitido. 3) Confirmar com código: handler de síntese backend NÃO revalida UnlockableSynthesisRecipe antes de ConfirmAlchemy.
- **correção:** Server-side: revalidar SynthesisRecipeKey + ItemSynthesisType + UnlockStatus do jogador ANTES de ConfirmAlchemy. Rejeitar se tipo não está na lista de receitas desbloqueadas. Aplicar mesmo padrão do fix de baú (wall-clock validation + server-authority).

## [MEDIA] Client-Authority em Remoção de Itens do Inventário
- **classe:** `TaskbarHero.UI.NoticeRemoveInvalidItem`
- **problema:** A classe gerencia quais items deletar (m_deleteSlots) e confirma localmente (m_checkButton), sugerindo que o cliente determina e autoriza a remoção. Se o servidor confiar nesse evento sem revalidar critérios de 'inválido', um atacante pode manipular m_deleteSlots via proxy/memória modificada para deletar items valiosos ou rarros.
- **verificar:** Interceptar requisição de remoção com Fiddler/Burp, editar m_deleteSlots para incluir item válido (não-inválido), observar se servidor deleta. Alternativa: modificar memória Unity em runtime (Cheat Engine) e executar botão.
- **correção:** Servidor REVALIDA TODOS os critérios de 'inválido' (rarity, level, cooldown, duplicate ownership) antes de deletar. Nunca confiar na decisão do cliente. Implementar signature/hash do item + servidor calcula hash esperado e compara. Log auditoria de todas as remoções de item.

## [MEDIA] Potencial Client-Authority: Dados de Skill Carregados de Fonte Não-Validada
- **classe:** `TaskbarHero.UI.SkillDescriptionPanel`
- **problema:** SetSkillInfo carrega dados de skill (nome, nível, atributos) em campos da UI sem validação mencionada. Se essa origem é cache local (PlayerPrefs, arquivo de save) editável, o cliente pode exibir dados falsos que influenciam gameplay (dano calculado, cooldown falso). Padrão comum em jogos com BaaS fraco: confiar em cache local para exibição e cálculo.
- **verificar:** 1) Verificar se SetSkillInfo busca dados de PlayerPrefs.Get*() ou PlayerData carregado localmente. 2) Procurar se m_skillLevel é usado por SkillManager.CalculateDamage() ou GetCooldown(). 3) Confirmar se servidor re-valida skill data antes de processar cast/equip. 4) Testar se editar player.json (ou save local) reflete dano alterado no jogo.
- **correção:** Implementar server-side validation: (1) Servidor é source-of-truth para skill level/atributos, cliente só exibe. (2) Antes de processar qualquer ação de skill (cast, equip, upgrade), servidor valida contra DB. (3) Implementar hash/signature em save local para detectar tamper. (4) Usar CodeStage AntiCheat para flagear edições de memória de m_skillLevel em runtime.

## [MEDIA] Client-Authority em Classificação de Itens
- **classe:** `TaskbarHero.Data.EItemType`
- **problema:** Enum de tipo de item no cliente sugere que tipos (STAGEBOX/MATERIAL/GEAR) podem ser enviados ao servidor ou usados em validações locais. Se o servidor confia na classificação do cliente sem re-validar contra DB (ex: crafting, trading, equipar), permite spoofing de tipo de item (cliente mente sobre qual tipo um item é).
- **verificar:** 1) Buscar onde EItemType é serializado/enviado em mensagens pro servidor (networking code, PlayerData save). 2) No servidor (backend), confirmar se há validação: `Item.type_real = DB.lookup(item_id); if(Item.type_real != cliente_type) REJECT`. 3) Testar: enviar craft de MATERIAL mas declarar tipo GEAR — verificar se servidor aceita.
- **correção:** Servidor SEMPRE valida tipo contra DB próprio. Cliente nunca envia tipo de item — servidor responde com confirmação (ex: {item_id, server_type, server_rarity}). Tipos/raridades/efeitos calculados 100% server-side. Cliente só renderiza o que servidor disse.

## [MEDIA] Client-Authority no Tipo de Monstro — Farming de Rewards de Boss
- **classe:** `TaskbarHero.Data.EMonsterType`
- **problema:** Se o servidor confia num EMonsterType enviado pelo cliente para distribuir rewards/progresso, um atacante pode falsificar derrotas de BOSS como MONSTER, duplicando rewards de boss (XP, itens, ouro) sem limite de rate.
- **verificar:** 1) Capture request de derrota de monstro (ex.: via proxy); 2) Mude MONSTER→BOSS; 3) Repita 100×; 4) Verifique se rewards de boss foram creditados sem validação server-side.
- **correção:** Servidor deve (a) gerar/rastrear tipo de monstro apenas server-side, (b) validar que EMonsterType recebido do cliente bate com o ID/state do monstro salvo no server, (c) rate-limit BOSS by time/día independente do cliente, (d) gravar audit log de derrotas de boss.

## [MEDIA] Client-authority em SkillBuffType.ApplyBuff()
- **classe:** `TaskbarHero.Data.SkillBuffType`
- **problema:** Métodos ApplyBuff()/RemoveBuff() podem ser chamados pelo cliente sem validação de servidor, permitindo o cliente falsificar estado de buff (Normal ↔ Buff) e sincronizar esse estado falso. Em jogo com trading, possibilita venda de buffs falsos ou duração indefinida.
- **verificar:** Intercepte tráfego /api entre cliente e TheBackend ao ativar buff; veja se servidor valida (1) duração/expiração, (2) source da ativação, (3) state antes de confirmar. Se servidor aceita sem validação, é confirmado.
- **correção:** Servidor deve: (1) calcular e armazenar buff expiration server-side; (2) rejeitar ApplyBuff se cooldown/stack limit violado; (3) validar que só ações autorizadas (skill cast, item uso) disparam ApplyBuff (não direto do cliente); (4) usar timestamps de servidor, nunca client-supplied.

## [MEDIA] Client-Authority de Modificadores de Stats (buff/debuff) - Duração e Aplicação Controladas no Cliente
- **classe:** `TaskbarHero.Data.StatModInfoData`
- **problema:** StatModInfoData.UpdateDuration() e RemoveModifier() executam no cliente, sem revalidação server-side. Permite buff infinito, stacking não validado e duração manipulada.
- **verificar:** 1) Verificar se servidor revalida modificadores ANTES de aplicar stats na defesa/dano; 2) Confirmar se cliente pode não chamar RemoveModifier() (breakpoint em mjz/mko); 3) Testar se ApplyModifier() pode ser chamado múltiplas vezes sem sincronização.
- **correção:** Servidor é source-of-truth: servidor aplica/remove modificadores. Cliente APENAS exibe status. Servidor revalida todos os stats em cada ação crítica (ataque/defesa/habilidade).

## [MEDIA] Client-Authority no Carregamento de Armadilha + Rate-Limit Apenas no Cliente
- **classe:** `TaskbarHero.Combat.HunterChargeTrap`
- **problema:** O método `mng` (Update) acumula tempo de carga do lado cliente; se o servidor recebe evento de ativação (mnh: OnActivate) sem validar o tempo decorrido contra relógio real (wall-clock), permite bypass via Time.timeScale acelerado ou edição direta dos campos bgtd/bgte. Dano aplicado (mma: ApplyEffect) não é validado contra carga real.
- **verificar:** 1) Acelerar Time.timeScale × 6 e ativar trap → servidor aceita kill/damage acelerado? 2) Editar bgtd/bgte na memória (MelonLoader/HarmonyX) e ativar trap antes de tempo → aplica dano máximo? 3) Existe rate-limit de ativações no servidor ou apenas no cliente (StageBoxRequestAbuseGuard)?
- **correção:** 1) Servidor valida tempo decorrido contra wall-clock, rejeita ativações impossíveis. 2) Servidor calcula dano da trap baseado no tempo decorrido verificado, não em valores enviados pelo cliente. 3) Rate-limit de ativações de trap deve ser server-side com bloqueio real, não client-side. 4) Derivar concessão de drops/baú do progresso validado no servidor, não de eventos de kill crus.

## [MEDIA] Client-Authoritative Shield — Ativação Permanente / Absorção Não-Validada
- **classe:** `TaskbarHero.Combat.KnightAegisField`
- **problema:** A habilidade resolve estado (duração, absorção de dano, cooldown) no cliente. Os métodos Activate/ApplyShield não incluem validação server-side de parâmetros, permitindo ativar o shield permanentemente ou aumentar sua efetividade via mod (MelonLoader/HarmonyX).
- **verificar:** Hook em ApplyShield com HarmonyX; capturar chamadas e modificar duration/absorption antes de enviar ao backend. Ou enviar Activate repetido sem Deactivate → o servidor deve rejeitar, mas provavelmente não faz.
- **correção:** Servidor é autoridade: cliente envia comando 'activate_shield', servidor calcula (duração = config[classe] + buffs servidor, absorção = HP*factor servidor), e retorna parâmetros ao cliente. Cliente renderiza apenas. Rate-limit: máx N ativaçõs/minuto por conta (wall-clock, não game-time).

## [MEDIA] Skill de efeito congelante (IceOrb) com estado de cliente — sem validação server-side de aplicação
- **classe:** `TaskbarHero.Combat.WizardIceOrb`
- **problema:** A classe WizardIceOrb encapsula uma habilidade mágica de ataque/controle (congelamento). Como tipo de combat action client-resolved, o servidor presumivelmente confia no evento de aplicação de efeito enviado pelo cliente. Ausência de server-side authority sobre quando/se o efeito foi realmente aplicado permite bypass de mecânicas de cooldown, consumo de mana e stuns fraudulentos.
- **verificar:** 1) Enviar manualmente uma ação IceOrb sem cooldown estar pronto (ex.: com modificação de tempo client-side). 2) Verificar se o servidor rejeita ou aplica anyway. 3) Observar se inimigos congelam sem que o efeito deveria ter sido aplicado. 4) Checar telemetria de lag — clientes offline conseguem aplicar skill?
- **correção:** 

## [MEDIA] Client-Authority em Cooldown + Dano de Ataque de Boss
- **classe:** `TaskbarHero.Combat.ActBossJumpAttack`
- **problema:** Campos bgxq/bgxr (dano) e cooldown são resolvidos no cliente. Time.timeScale acelera cooldown; Cheat Engine edita modificadores de dano. Servidor não valida taxa real de ataque vs wall-clock.
- **verificar:** Elevar Time.timeScale (ex: 6×); monitorar ataques de jump reportados ao servidor. Editar bgxr via Cheat Engine e confirmar que dano aumenta no servidor.
- **correção:** Server-side: (1) validar taxa de ataques contra relógio real, rejeitar se >max_atk/sec; (2) recalcular dano (bgxq/bgxr) no servidor baseado em stats do boss validados; (3) limpar cooldown server-side, não client-side

## [MEDIA] Client-side spawn interval and offset permitem manipulação de DPS em combate
- **classe:** `TaskbarHero.Combat.BoneFallArea`
- **problema:** Os campos m_projectileSpawnIntervalTime e m_boneSpawnOffset residem no cliente e controlam taxa e posição de spawn. Sem validação server-side, um player pode reduzir o intervalo (Cheat Engine) ou ajustar offset para gerar projéteis mais rápido ou fora da área intendida, ganhando vantagem não-autorizada.
- **verificar:** Dumpar IL2CPP, modificar m_projectileSpawnIntervalTime em runtime (diminuir valor), observar se taxa de projéteis aumenta. Verificar se servidor aceita/processa eventos de spawn sem rate-limit ou validação de intervalo canônico.
- **correção:** Servidor mantém intervalo de spawn canônico e estado de projéteis; cliente apenas renderiza. Validar cada evento de spawn server-side contra taxa máxima permitida. Offset deve ser determinístico (baseado em ID da área), não editável.

## [MEDIA] Client-Authority em Soul Snatch: efeito de debuff/dano não validado server-side
- **classe:** `TaskbarHero.Combat.ActBossSoulSnatch`
- **problema:** ActBossSoulSnatch resolve Initialize→CanExecute→Execute→ApplyEffect inteiramente no cliente. O servidor recebe apenas o evento de execução e confia nele; não há validação de pré-condições (distância, mana, cooldown real) ou do dano/debuff aplicado. Um cliente modificado pode executar o snatch fora de cooldown, com dano aumentado, ou em targets inválidos.
- **verificar:** Capturar trace de rede quando Execute() é acionado; verificar se o servidor valida timestamps reais entre execuções, se recomputa dano baseado em stats conhecidos do boss, ou apenas aplica o efeito reportado. Também testar: usar MelonLoader para chamar Execute() diretamente sem passar por CanExecute().
- **correção:** Servidor deve: (1) manter cooldown real por conta; (2) recalcular dano/debuff apenas servidor-side baseado em estado validado; (3) rejeitar eventos fora da janela temporal válida; (4) não executar ApplyEffect sem validar que o alvo está em range/combate.

## [MÉDIA] Client-Authority em Classificação de Inimigos
- **classe:** `TaskbarHero.EBoxType`
- **problema:** Enum de tipo de inimigo (NORMAL/BOSS/ACTBOSS) definido no cliente. Se servidor confiar que cliente está classificando corretamente o tipo derrotado, permite dupe de recompensas: cliente marca inimigo NORMAL como BOSS para receber loot/XP superior ou contaminação de estatísticas de combate.
- **verificar:** 1) Interceptar requisição de derrota de inimigo (ex: JSON com EBoxType) via proxy/Wireshark; 2) Modificar type de NORMAL→BOSS antes enviar; 3) Verificar se recompensas mudaram; 4) Repetir com mesmo inimigo; 5) Checar logs servidor se validou ou aceitou cegamente.
- **correção:** Servidor deve ter tabela authoritative de inimigos com tipo+loot pré-definido. Ao receber 'derrotei inimigo ID X', validar: (A) ID X existe, (B) tipo armazenado ≠ tipo do cliente, (C) recompensas vêm da tabela server, NÃO do cliente. Remover enum de tipo do lado cliente ou usar só pra renderização.

## [MÉDIA] Client-Authority em Troca de Inventário
- **classe:** `TaskbarHero.InventoryExchangeRequest`
- **problema:** Campos o_* (old) vs sem prefixo sugerem que o cliente afirma seu estado anterior de inventário. Se servidor não valida que o cliente REALMENTE POSSUI os itens em o_resultItemId/o_itemKey, há risco de troca fraudulenta ou dupe.
- **verificar:** Envie múltiplas InventoryExchangeRequests com o_resultItemId=1 mas resultItemId=999 (falso). Monitore servidor-logs: se aceitar sem verificar posse real do item 1, ou se dupe o 999, há vuln.
- **correção:** Servidor deve: (1) verificar o_* contra inventário autêntico do jogador no banco antes de aceitar troca; (2) validar regras de craft/troca; (3) usar transaction atômica; (4) rate-limit requisição por player/min

## [MÉDIA] Client-Authority em Seleção de Tipo de Síntese
- **classe:** `TaskbarHero.UI.SynthesisItemTypeChangerButton`
- **problema:** O componente armazena m_synthesisItemType que, presumivelmente, é enviado ao servidor como parte da requisição de síntese. Se o servidor não valida qual tipo o jogador realmente pode sintetizar, o cliente pode escolher tipos restritos ou contornar pré-requisitos.
- **verificar:** Interceptar requisição POST de síntese com proxy (Burp/Fiddler). Mudar synthesisItemType para um tipo inválido, banido ou restrito a outros players. Se síntese processa normalmente, vulnerabilidade confirmada.
- **correção:** Servidor deve re-validar: (1) se tipo de síntese existe e é público, (2) se jogador tem pré-requisitos (nível, itens, skills), (3) se tem recursos suficientes (antes de processar). Nunca confiar em valores do cliente escolhido na UI.

## [MÉDIA] Campos Obfuscados em Classe de Interação — Possível Transmissão Não-Validada
- **classe:** `TaskbarHero.UI.HeroPreviewSkillSlot`
- **problema:** Campos `bfzx` e `bfzy` com nomes genéricos/obfuscados numa classe que gerencia INTERAÇÃO de skill equipável (não só exibição) sugerem armazenamento de IDs críticos. Se esses valores são transmitidos ao servidor em eventos de equipagem/desequipagem, há risco de manipulação em memória + falta de validação server-side.
- **verificar:** 1) Inspecionar bytecode/decompile: o que `bfzx`/`bfzy` realmente armazenam? 2) Procurar métodos como `OnEquipClick()`, `SendEquipRequest()` — eles enviam esses campos direto pro servidor? 3) No server: ao receber equip_skill_id, ele valida se o player realmente possui essa skill + tem slots livres, ou só confia no ID?
- **correção:** Server nunca confie em IDs vindo do cliente. Ao receber 'equipar skill X': (1) verifique posse no banco de dados, (2) verifique autorização/desbloqueio, (3) verifique limites (slots, cooldown), (4) apenas então aplique. Nunca use valor do cliente como source-of-truth.

## [MÉDIA] Cooldown de Refresh Contornável (Client-Side Authority)
- **classe:** `TaskbarHero.UI.UI_Mailbox`
- **problema:** RefreshCoolTime é gerenciado no cliente e facilmente bypassável via memory editing (Cheat Engine). Sem validação server-side, permite spam ilimitado de requisições de refresh, potencialmente levando a dupe de recompensas ou sobrecarga do servidor.
- **verificar:** Memory scanner localiza RefreshCoolTime em heap, zera valor, clica Refresh repetidamente. Se funciona sem delay, cooldown está só no cliente. Capture request POST e resenda sem esperar timer.
- **correção:** Move controle de cooldown para servidor: servidor armazena last_refresh_timestamp por jogador, rejeita requests prematuramente com HTTP 429. Cliente apenas renderiza countdown UI baseado em timestamp server recebido.

## [MÉDIA] Enum Deserialization sem Validação + Client Authority no Damage Type
- **classe:** `TaskbarHero.Data.EDamageAttribute`
- **problema:** Se o servidor aceita EDamageAttribute do cliente sem validar com Enum.IsDefined(), um atacante envia um inteiro fora do range válido (ex: -1, 999), causando index out-of-bounds em arrays/lookups do servidor ou permitindo que o cliente escolha arbitrariamente qual tipo de dano aplicar (bypassing resistências ou multiplicadores esperados).
- **verificar:** 1) Procurar recebimento de damageType no servidor sem Enum.IsDefined() check. 2) Verificar se existe array indexado por (int)damageType sem bounds validation. 3) Confirmar se cliente pode escolher EDamageAttribute livremente num ataque (vs servidor forçá-lo baseado em arma/skill).
- **correção:** Servidor: (1) validar sempre com `if (!Enum.IsDefined(typeof(EDamageAttribute), value))` antes de cast; (2) usar whitelist de tipos de dano permitidos por arma/skill (server-authoritative); (3) se áreas usam arrays, dimensionar pra Enum.GetValues().Length e nunca confiar em valor do cliente.

## [MÉDIA] Client-Authority em AttributeKey - Potencial Boost de Stats/Atributos
- **classe:** `TaskbarHero.Data.AttributeLoadData`
- **problema:** Se AttributeKey é serializado/carregado localmente e o servidor não revalida qual chave de atributo cada jogador possui, um attacker pode editar o arquivo save ou interceptar o payload pra falsificar atributos (ex.: trocar 'defense' por 'attack', ou valores de atributo). Em idle-RPGs com TheBackend + trading, isso permite stat-boost, dupes indiretas, ou economia quebrada.
- **verificar:** 1. Procurar onde AttributeLoadData é desserializado (arquivo save local? JSON do servidor?). 2. Verificar se servidor revalida AttributeKey contra o progresso legítimo do jogador (ex.: pode ter atributo 'legendary_defense' no level 5?). 3. Testar editar o arquivo save local ou interceptar request de load pra injetar AttributeKey novo. 4. Checar se valores de atributo são recalculados server-side ou confiados do cliente.
- **correção:** Server-side: (1) Servidor RECALCULA todos os atributos do jogador quando loads, nunca confia em AttributeKey enviado pelo cliente. (2) Whitelist de atributos válidos por level/progression. (3) Revalida antes de qualquer operação crítica (trade, upgrade, damage calc). (4) Log anomalias (atributos impossíveis). (5) Se uses save file local, encripta+HMAC, não JSON puro.

## [MÉDIA] Client-Side Combat Authority com Campos Obfuscados
- **classe:** `TaskbarHero.Combat.SoulSnatchEffect`
- **problema:** Classe contém 6 campos obfuscados (bham, bhan, bhao, bhap, bhaq, bhar) em pipeline de combate com TheBackend BaaS e Steam trading. Em contexto de idle-RPG, efeitos visuais parametrizáveis frequentemente trigram eventos de servidor não-validados (dano confirmado, loot gerado) — presença de obfuscação sugere logic crítico ocultado que pode ser client-authoritative.
- **verificar:** 1) Desassemble SoulSnatchEffect: campos bham* contêm RPC calls ou referencias a objetos que comunicam com servidor? 2) Audit da cadeia de recompensas: servidor valida dano/XP **antes** de confiar em conclusão de efeito visual? 3) Rate-limit implementado server-side pra eventos de combate ou só no cliente? 4) Teste: editar m_pullDuration serializado no savegame/asset — afeta timing de recompensas?
- **correção:** 1) Move ALL damage/loot/XP calculation server-side; cliente apenas serializa inputs. 2) Valide duração total de combate server-side (não trust client timing). 3) Implement server-side rate-limiting + cooldown pra eventos de combate, com audit trail. 4) Remove campos obfuscados ou documente publicamente sua função (ferramenta anti-análise é red-flag). 5) Jogo com Steam trading: implement anti-dupe checks (verify item history on trade).

## [CRÍTICA] Client-Authority de Dano + Cooldown + Distância (Múltiplos Vetores)
- **classe:** `TaskbarHero.Combat.ActiveSkill`
- **problema:** ActiveSkill calcula/valida dano, cooldown, mana e distância 100% no cliente. Atacante pode editar valores de dano via Cheat Engine, fazer bypass de cooldown com reset instantâneo, ignorar validação de skillCastDistance (range), e contornar mana/charges—tudo sem re-validação server-side. Em jogo com trading/economia, isso permite dupe de kills, trivialização de boss, e manipulação de ranking.
- **verificar:** 1) Capturar tráfego: skill-cast message contém dano final? 2) Editar memória com Cheat Engine: ApplySkillEffect.AttackDamage += 10000x 3) Resetar cooldown via debugger 4) Testar cast fora de range (skillCastDistance=999) 5) Verificar se servidor aceita e persiste o dano editado
- **correção:** 1) SERVIDOR recalcula dano (base skill + stats do player + buffs), nunca confia em dano do cliente 2) Servidor valida cooldown state authoritatively (timestamp last-cast + cooldown duration) 3) Servidor valida distance(player, target) via coordenadas server-truth 4) Servidor verifica mana/charges antes de aplicar 5) Skill effect só é aplicado se TODOS os requisitos passaram server-side. Cliente é visual/predictive only
