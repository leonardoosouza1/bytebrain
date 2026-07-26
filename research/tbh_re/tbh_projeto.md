# TaskbarHero — projeto recriado (Claude de-ofuscou por contexto)
> 558 classes TaskbarHero.* anotadas · campos reais + métodos inferidos

## ?  (1)
### TaskbarHero.Log.GetBoxLog
?
- campos: bfod, bfoe, bfof

## boss  (7)
### TaskbarHero.Combat.ActBossBoneFall
Define um ataque do boss que faz ossos caírem sobre o jogador em padrão de queda
- campos: bgxd
  - mml: Initialize - prepara os pontos de spawn e velocidades iniciais dos ossos
  - mlz: Execute - executa a lógica principal (dispara ossos em loop)
  - mma: OnComplete - limpa referências e finaliza o ataque
  - mqy: SpawnBones - cria instâncias dos projectis de osso com trajetória
### TaskbarHero.Combat.ActBossJumpAttack
Implementa o ataque de salto do chefe boss, causando dano em área ao impactar
- campos: bgxq, bgxr
  - Execute: inicia o ataque de salto com animação e trajetória
  - CalculateDamage: calcula dano baseado nos campos bgxq (força) e bgxr (modificador)
  - GetCooldown: retorna tempo de recarga do ataque
  - OnImpact: aplica dano e knockback aos alvos atingidos
  - IsReady: verifica se o ataque está disponível para ser executado
### TaskbarHero.Combat.ActBossMonsterStab
Ação de combate executada por um boss: executa um golpe/punhalada contra um alvo com seleção e cálculo de dano.
- campos: bgxy
  - mml: Executa o ataque de punhalada no alvo
  - mlz: Inicializa parâmetros do ataque (dano, range, custos)
  - mma: Seleciona ou valida o alvo do ataque
  - msa: Aplica efeito/dano ao alvo após resolução
### TaskbarHero.Combat.ActBossSetDamageReductionMode
Encapsula a lógica de ativar e gerenciar um modo de redução de dano defensivo do boss.
  - Activate: Ativa o modo de redução de dano, possivelmente ajustando o multiplicador defensivo
  - Deactivate: Encerra o modo de redução, restaurando o damage model padrão
  - GetDamageReduction: Retorna o multiplicador ou percentual de redução de dano ativo
  - Update: Mantém/processa o efeito durante o frame (cooldown, duração, stacking)
  - OnDamageReceived: Hook que intercepta dano recebido e aplica a mitigação
### TaskbarHero.Combat.ActBossSoulSnatch
Ação de chefe que captura e absorve a alma do alvo em combate.
- campos: bgyf
  - Initialize: Validar condições pré-execução
  - CanExecute: Checar mana/distância/estado
  - Execute: Disparo da captura
  - ApplyEffect: Dano + debuff aplicado
  - Cleanup: Reset de estado
### TaskbarHero.Combat.ActBossVoidThunder
Ação especial de boss: ataque encontro trovão com elemento vazio infligindo dano + debuff.
  - Execute: Inicializa e dispara o ataque trovão no alvo/área
  - CalculateDamage: Calcula dano baseado em stats do boss e defesa do jogador
  - ApplyVoidEffect: Aplica efeito de debuff/stun de vazio no alvo
  - GetCastTime: Retorna duração/delay de castagem do ataque
### TaskbarHero.Combat.Projectile.ActBossVoidWave
Projectil de onda do vácuo lançado pelo boss, expande e causa dano em área
  - mml: Atualiza posição e expansão do projectil cada frame
  - mlz: OnTriggerEnter — detecta entidades atingidas
  - mma: DealDamage — aplica dano aos inimigos na área
  - nao: Destroy — limpa projectil após duração ou impacto

## combate  (94)
### TaskbarHero.Data.ACTIVATIONTYPE
Enum que define tipos de ativação de habilidades/ataques (ataque base, contagem, cooldown, contínuo)
- campos: value__, BASEATTACK, BASEATTACK_COUNT, COOLDOWN, CONTINUOUS
  - GetActivationType: retorna o tipo de ativação baseado no valor
  - HasCooldown: verifica se o tipo requer cooldown
  - IsContinuous: testa se a ativação é contínua
  - GetCooldownDuration: retorna duração do cooldown
  - IsBaseAttack: verifica se é ataque base
### TaskbarHero.Combat.AOEHitLineEffect
Efeito visual de acerto em linha que aplica dano em área, ativado quando ataque de linha atinge múltiplos inimigos
- campos: m_aoe, m_lineController, bgyv
  - mtj: Aplica dano AOE e anima efeito da linha de impacto
### TaskbarHero.Combat.ActBossDashAttack
Ação de ataque do boss com deslocamento rápido em combate
- campos: bgxi
  - mml: Initialize — Prepara parâmetros do dash (velocidade, duração, distância)
  - mlz: Execute — Executa o deslocamento e aplica dano ao alvo
  - mma: CanExecute — Valida se o boss pode executar (cooldown, distância mínima)
  - mre: Cleanup — Reseta estado e efeitos visuais do ataque
### TaskbarHero.Combat.ActBossMassiveVoidCall
Ação de combate do boss que invoca múltiplas criaturas Void para atacar o jogador em larga escala.
  - Execute: Dispara a sequência completa da convocação massiva
  - InitializeVoidSpawns: Prepara posições e parâmetros das criaturas a invocar
  - AnimateVoidCall: Renderiza efeito visual/animação da convocação
  - ApplyDamagePattern: Calcula e aplica dano de cada criatura ao alvo
### TaskbarHero.Combat.ActBossSpawnUndead
Ação de combate onde um boss invoca criaturas não-mortas (zumbis) no campo de batalha
  - Execute: Executa o spawn dos zumbis no campo de batalha
  - GetSpawnCount: Retorna quantos zumbis serão invocados
  - GetCooldown: Retorna cooldown/recarga da skill em turnos
  - ValidateTargets: Valida se há espaço/tiles para invocar os zumbis
  - GetManaCost: Retorna custo de energia/mana da ação
### TaskbarHero.Combat.ActBossSpawnVoidTurret
Ação de combate que spawna uma torrente void (turret) durante o ataque do boss.
- campos: bgym, bgyn
  - Initialize: Configura parâmetros iniciais do spawn (delay, posição, ativo)
  - Execute: Executa o spawning da void turret no timing correto
  - CalculateSpawnPosition: Calcula coordenadas de spawn baseado em bgym/bgyn
  - UpdateTurretBehavior: Controla movimento/ataque da torrente depois de spawn
  - Cleanup: Desativa e libera referências da torrente ao terminar
### TaskbarHero.Combat.ActiveSkill
Gerencia execução, animação e estado (cooldown/mana) de um skill ativo no combate
- campos: skillSo, skillCache, skillCastDistance, bhbu, bhbv, bhbw, bhbx, bhby, AnimClipName, ActionTimeName, animaitionProbabiltyDatas, bhbz, bhca, bhcb, bhcc, bhcd
  - CastSkill: inicia o cast do skill com verificações de distância e disponibilidade
  - IsSkillReady ou CanCastSkill: valida se o skill pode ser usado (cooldown/mana/charges)
  - UpdateCooldown: decrementa cooldown e atualiza estado do skill
  - PlayAnimation: executa clip de animação do skill (AnimClipName)
  - ApplySkillEffect: aplica dano (AttackDamage) e efeitos secundários no alvo
### TaskbarHero.Combat.ArcherBarrageAtk
Habilidade de combate que dispara uma rajada de múltiplas flechas do arqueiro contra um alvo.
- campos: bgsq, currentTarget
  - Execute: Inicia a sequência de barrage
  - FireProjectile: Lança cada flecha individual da rajada
  - CalculateDamage: Calcula dano total do ataque
  - PlayAnimation: Toca a sequência visual de ataque (bgsq)
  - OnTargetHit: Processa acerto e aplica efeitos ao alvo
### TaskbarHero.Combat.ArcherBaseAtk
Sistema base de ataque do arqueiro que define dano, velocidade e conta de projéteis para skills derivadas
- campos: bgsr, bgss, bgst
  - mma: Execute — executa o ataque base contra alvo(s) atual
  - mlz: Initialize — configura o ataque com valores de bgsr/bgss/bgst
  - mmq: GetTargetList — valida e retorna alvo(s) no range do ataque
  - .ctor: Constructor — inicializa campos de dano/velocidade/contagem
### TaskbarHero.Combat.Projectile.ArcherBaseProjectile
Classe base para projéteis de arqueiro que gerencia trajetória, colisão e dispara eventos de acerto crítico.
- campos: OnCriticalHit
  - Launch: inicializa e dispara o projétil com velocidade/direção
  - OnTriggerEnter: detecta colisão com alvo e processa dano
  - CalculateDamage: calcula dano final incluindo crítico
  - TriggerCriticalHit: dispara o evento OnCriticalHit para lógica adicional
  - Despawn: remove o projétil da cena após impacto/timeout
### TaskbarHero.Combat.Projectile.ArcherPiercingArrow
Projétil de seta perfurante que atravessa múltiplos inimigos aplicando dano contínuo
- campos: bhem
  - mml: Initialize - configura posição, velocidade e direção inicial da seta
  - mlz: UpdateTrajectory - atualiza movimento e posição a cada frame
  - mma: ApplyPiercingDamage - aplica dano a inimigos atingidos (registra em bhem para não acertar 2x)
  - nae: OnOutOfBounds - remove seta quando sai da tela ou atinge limite de penetração
### TaskbarHero.Combat.ArcherSpreadShotAtk
Habilidade de arqueiro que dispara múltiplos projéteis em padrão de spread/dispersão controlado por ângulo
- campos: atkCount, angle
  - mlz: Initialize() — configura atkCount e ângulo inicial
  - mmv: CalculateSpread() — distribui ataques em leque por angle
  - mmw: SpawnProjectiles() — cria múltiplos projéteis (x atkCount)
  - mmx: Execute() — dispara ataque (fire/trigger)
  - mma: ApplyEffect() — aplica dano/efeito visual aos projéteis
### TaskbarHero.Data.BackGroundContainer
Container que armazena e organiza elementos visuais de fundo de um palco/fase em múltiplas camadas (chão, fundo principal, 3 camadas paralaxe)
- campos: StageKey, FloorObject, FloorBackGround, Layer1Object, Layer2Object, Layer3Object, m_allObjects
  - Initialize: cria e posiciona todos os objetos FloorObject, FloorBackGround, Layer1-3 do StageKey
  - GetAllObjects ou PopulateAllObjects: preenche m_allObjects com todas as referências
  - DestroyBackground: limpa/destrói todos os GameObjects do container
  - SetupSortingOrder: ordena Z ou sorting layers para renderização correta
  - LoadFromAssets: carrega prefabs do fundo baseado no identificador StageKey
### TaskbarHero.Combat.BoneFallArea
Gera projéteis de osso em intervalos regulares em uma área de combate.
- campos: m_boneProjectile, m_boneSpawnOffset, m_projectileSpawnIntervalTime, bgzm
  - Update: Decrementa timer de intervalo e dispara spawn quando atinge zero
  - SpawnBone: Instancia projétil de osso com m_boneSpawnOffset aplicado
  - ResetSpawnTimer: Reseta m_projectileSpawnIntervalTime após cada disparo
  - InitializeProjectile: Configura velocidade/direção do projétil spawned
  - OnTriggerEnter: Ativa/desativa a geração de ossos baseado em proximidade
### TaskbarHero.SO.BuffEffectData
Dados configuráveis para efeitos visuais e regras de stacking de buffs/debuffs em combate
- campos: BuffKey, EffectObject, EffectStackType, EffectTransformType
  - GetBuffKey: retorna identificador único do buff
  - InstantiateEffect: cria GameObject do efeito visual na posição do alvo
  - GetStackBehavior: retorna se buff substitui, acumula ou ignora duplicatas
  - ApplyTransform: posiciona/rotaciona efeito conforme EffectTransformType
  - CleanupEffect: destrói GameObject do efeito quando buff expira
### TaskbarHero.Status.BuffDebuff.BuffExpireConditionType
Enum que define as condições sob as quais buffs/debuffs expiram em combate (por tempo, contagem de ataques ou mudança de estágio).
- campos: value__, Time, BaseAttackCount, Stage
  - ToString(): retorna nome legível da condição de expiração
  - IsExpired(float elapsed, int attackCount, int currentStage): verifica se buff deve expirar
  - GetExpirationText(): descreve em texto quando o buff vai expirar
  - GetConditionValue(): retorna o valor numérico associado à condição
### TaskbarHero.Data.BuffGroupInfoData
Armazena dados de um grupo de buffs com três componentes de valor numérico (x, y, z).
- campos: bgmx, bgmy, bgmz
  - ApplyBuffGroup: aplica o conjunto de buffs a um alvo
  - GetTotalEffect: calcula o efeito combinado dos três componentes
  - UpdateBuffValues: atualiza x, y, z com novos valores
  - IsBuffActive: verifica se algum buff do grupo ainda está ativo
  - RemoveExpiredBuffs: limpa buffs que expiraram
### TaskbarHero.SO.BuffGroupSO
Define um grupo de buffs/efeitos que podem ser aplicados ao acertar inimigos ou ao longo do tempo em uma área.
- campos: UseOnHitEffect, OnHitEffect, UseDurationEffect, DurationEffect, AreaOfEffect
  - Apply: Aplica os efeitos do buff ao alvo
  - HasOnHitEffect: Verifica se o buff tem efeito ao acertar
  - HasDurationEffect: Verifica se o buff tem efeito com duração
  - GetAreaOfEffect: Retorna o raio de alcance da área
  - Initialize: Configura o grupo de buffs com os efeitos
### TaskbarHero.Data.BuffInfoData
Armazena dados de buffs/debuffs aplicados ao jogador ou entidades (tipo, duração, magnitude, ícone, flags).
- campos: bgna, bgnb, bgnc, bgnd, bgne
  - GetBuffType: retorna o tipo/ID do buff armazenado em bgna
  - GetDuration: retorna duração em ticks ou segundos (provável bgnb)
  - GetMagnitude: retorna poder/valor do efeito (provável bgnc)
  - IsActive: verifica se o buff ainda está ativo
  - ApplyEffect: aplica o buff ao alvo usando os dados armazenados
### TaskbarHero.Data.BuffInfoDataMap
Mapa de dados que armazena e acessa informações de buffs (efeitos temporários) do jogo.
  - GetBuffInfo: recupera dados de um buff específico pelo ID
  - AddBuffInfo: registra um novo tipo de buff no mapa
  - HasBuff: verifica se um buff existe no mapa
  - RemoveBuff: remove um buff do mapa
  - GetAllBuffs: retorna todos os buffs cadastrados
### TaskbarHero.Status.BuffDebuff.BuffStackType
Define o tipo de acúmulo de buffs — se apenas um ativo por vez (OnlyOne) ou múltiplos simultâneos (Stackable).
- campos: value__, OnlyOne, Stackable
  - GetStackBehavior: Retorna se buff segue modelo OnlyOne ou Stackable
  - CanStack: Valida se um novo buff pode ser adicionado baseado no tipo
  - ToString: Serializa o tipo de stack (usado em debug/logs)
  - CompareTo: Ordena buffs por prioridade de aplicação
  - Validate: Verifica se value__ é um tipo de stack válido
### TaskbarHero.Status.BuffDebuff.BuffType
Enum que define os tipos de modificadores de status (Buff = bônus, Debuff = penalidade)
- campos: value__, Buff, Debuff
  - IsBuff: Retorna true se type == Buff
  - IsDebuff: Retorna true se type == Debuff
  - GetColor: Retorna cor para renderização UI (verde/vermelho)
  - GetUIIcon: Retorna sprite ou ícone do tipo
  - GetDisplayName: Retorna nome localizado do modificador
### TaskbarHero.Combat.ContinuousSkill
Classe base para habilidades de combate que persistem ativas por duração, executando efeitos contínuos a cada frame.
  - mwv: Update/Tick — aplicar lógica contínua (dano, cooldown, duração)
  - mma: Start/Activate — iniciar habilidade e seus efeitos
  - mne: Stop/End — encerrar habilidade e limpar estado
  - moy: IsActive/CanContinue — verificar se habilidade ainda está ativa
### TaskbarHero.Effect.CreatedEffect
Representa um efeito visual/status criado em cena, gerencia seu pooling e permite remoção posterior via chave.
- campos: RemoveEffectKey, CreatedEffectPoolable, CreateEffectTransform
  - Init(RemoveEffectKey, CreateEffectTransform): Inicializa o efeito com chave de remoção e posição
  - OnRemove(): Desativa/retorna o efeito ao pool quando chamado via RemoveEffectKey
  - GetEffectTransform(): Retorna Transform do efeito criado
  - SetPoolable(CreatedEffectPoolable): Armazena referência ao objeto poolable do efeito
### TaskbarHero.UI.CubeExtractionOptionButton
Botão de UI interativo que gerencia seleção visual e exibição de um slot de encantamento (tier, decoração, ícone) com feedback de hover e estado.
- campos: m_inscriptionSprite, m_engravingSprite, m_decorationSprite, m_tierImage, m_enchantIconImage, m_backgroundImage, bghw, m_button, m_defaultSlotSprite, m_selectedSlotSprite, m_hoverEdgeObject, m_selectViewObject, m_emptyEnchantSlotObject, m_enchantType, m_enchantSlotIndex, bghx
  - lzi: SetEnchant — Carrega dados do encantamento e atualiza tier/ícone
  - lzj: ShowEmpty — Ativa visuals para slot vazio (m_emptyEnchantSlotObject)
  - lzk: UpdateVisuals — Refresh de sprites e imagens baseado em m_enchantType
  - lzl: SelectButton — Ativa m_selectViewObject e troca sprite selecionado
  - lzm: DeselectButton — Desativa m_selectViewObject e volta sprite padrão
### TaskbarHero.Combat.SpecialStructure.DamageAbsorbComponent
Escudo temporário que absorve dano inimigo dentro de um raio, com duração limitada
- campos: SpawnOffset, ProtectDistance, bhdu, bhdv, bhdw, bhdx, bhdy, existLifeTime, originLifeTime, bhdz, bhea, bheb
  - mzp: OnDamageReceived - reduz pool de absorção conforme dano aplicado
  - mzq: UpdateAbsorption - atualiza decaimento temporal do escudo
  - mzr: GetCurrentAbsorbAmount - retorna quanto dano o escudo ainda consegue absorver
  - mzs: DestroyAbsorber - remove escudo quando quebrado ou timeout
  - mzt: ApplyDamageToNearby - distribui parte do dano absorvido pra inimigos próximos
### TaskbarHero.DamageInfo
Encapsula dados completos de um evento de dano no sistema de combate, incluindo origem, valor, tipo, atributo e efeitos associados.
- campos: Attacker, OriginDamage, IsCritical, FloatingDamageText, DamageAttribute, DamageType, PlayHitFeedBack, PlayHitSound, HitEffects
  - Apply: Aplica o dano ao alvo com todos os efeitos, som e feedback
  - GetFinalDamage: Calcula valor final considerando crítico e modificadores de atributo
  - PlayEffects: Reproduz feedback visual/háptico e som de acerto baseado em DamageType
  - ShowFloatingText: Exibe valor de dano flutuante na tela do alvo
  - CreateDamageInfo: Factory/construtor que monta instância a partir de parametrizados
### TaskbarHero.Combat.DamageableType
Enum que classifica tipos de entidades que podem receber dano no sistema de combate (herói, monstro, estrutura).
- campos: value__, Hero, Monster, Structure
  - ToString: retorna nome do tipo de entidade (Hero/Monster/Structure)
  - GetHashCode: retorna hash do valor enum para uso em dicionários
  - Equals: compara igualdade entre valores de DamageableType
  - CompareTo: ordena valores enum por seu valor numérico subjacente
  - HasFlag: verifica presença de flag (se houver atributo [Flags])
### TaskbarHero.Combat.DetectType
Enum que define estratégias de detecção e spawn de inimigos em combate (raio central ou laterais)
- campos: value__, MiddleDetectRadius, SpawnLeftDirection, SpawnRightDirection
  - GetDetectRadius(): retorna o raio de detecção correspondente ao tipo
  - GetSpawnDirection(): retorna a direção de spawn (esquerda ou direita)
  - IsMiddleDetect(): verifica se é detecção central
### TaskbarHero.EBoxType
Enum que classifica tipos de entidades no combate: inimigos normais, bosses, ou bosses ativos.
- campos: value__, NORMAL, BOSS, ACTBOSS
  - IsBoss: retorna true se o tipo é BOSS ou ACTBOSS
  - GetDifficulty: retorna multiplicador de dificuldade (1x normal, 3x boss)
  - ToString: retorna nome legível do tipo de box
  - Parse: converte string ('BOSS') em valor de enum
  - IsActive: diferencia ACTBOSS de BOSS (boss em combate vs inativo)
### TaskbarHero.Data.EDamageAttribute
Enum que define os tipos de atributo de dano aplicáveis em combate (físico, fogo, frio, raio, caos).
- campos: value__, Physical, Fire, Cold, Lightning, Chaos, AllElement, None
  - ToString: retorna o nome do tipo de dano como string
  - GetName: obtém o nome legível do atributo de dano
  - GetValues: retorna array com todos os tipos de dano disponíveis
  - Parse: converte string para o enum correspondente
  - HasFlag: verifica se o atributo inclui um tipo específico
### TaskbarHero.Data.EDamageType
Enumera tipos de dano no sistema de combate (físico, mágico, área, invocação, DOT e armadilha)
- campos: value__, None, Melee, Projectile, AOE, Summon, DOT, Trap
  - GetDamageTypeName: retorna nome legível do tipo de dano
  - IsPhysicalDamage: verifica se é dano físico direto (Melee/Projectile/Trap)
  - CanAffectMultipleTargets: verifica se afeta área (AOE/Summon)
  - GetResistanceModifier: retorna modificador de resistência do inimigo
  - GetAnimationType: retorna animação visual associada ao tipo
### TaskbarHero.Data.EDropType
Enumeração que define estratégias de distribuição de drops/loot em combate (por peso aleatório, por classe do jogador, ou variantes DLC)
- campos: value__, EachDropOneWeight, SelectOneByClass, EachDropOneWeight_DLCVariant
  - GetWeight: calcula peso de drop para o tipo especificado
  - IsClassRestricted: verifica se drop é limitado a classe específica
  - ApplyDLCModifier: aplica ajustes de raridade/quantidade para conteúdo DLC
  - SelectDropForClass: escolhe drop apropriado baseado na classe do jogador
### TaskbarHero.Data.EGearUniqueMod
Enum de modificadores únicos (Unique Mods) aplicáveis a equipamentos, cada um alterando mecânica de habilidades específicas (cooldown, efeitos, contagem de projéteis, atributos de party).
- campos: value__, None, ShieldChargeKillCooldown, SkewerShotBleedingStrike, ArrowRainCriticalCooldown, FlameHydraBerserk, IceOrbFreezeToCold, SnowstormEnhanceFrozenEnemy, SorcererLightningShock, WrathOfHeavenHeal, ExplosiveBoltHalf, ChargeTrapExplosiveCooldown, CrossbowTurretCooldown, CrossbowTurretAddAmount, AxeSpinBleedingChance, SlayerLowHpAttackSpeed
  - ToString: retorna nome do modificador
  - GetHashCode: hash para lookup em dicionário
  - CompareTo: comparação para ordenação de mods
  - Parse: converte string em enum value
  - GetValues: retorna array de todos os mods disponíveis
### TaskbarHero.Log.EMonsterLogType
Classificar tipos de inimigos para registro/logging (normal, boss, boss de ato) em encontros de combate
- campos: value__, Monster, Boss, ActBoss
  - ToString: retorna nome legível do tipo de monstro
  - IsActBoss: verifica se é boss de ato
  - IsBoss: verifica se é qualquer tipo de boss
  - GetRewardMultiplier: obtém multiplicador de rewards baseado no tipo
  - CreateLogEntry: cria entrada de log do encontro
### TaskbarHero.Data.EMonsterType
Enum que classifica tipos de monstros em combate (normal vs boss)
- campos: value__, MONSTER, BOSS
  - GetName: retorna nome string do tipo de monstro
  - GetValue: retorna valor inteiro do enum (0=MONSTER, 1=BOSS)
  - ToString: converte enum para string para debug/UI
  - CompareTo: compara dois tipos de monstro por valor
  - HasFlag: verifica se tipo tem flag específica (se usasse Flags attribute)
### TaskbarHero.Data.EParamExchangeType
Enum que define tipos de parâmetros conversíveis em combate, com fatores de escala (÷100 ou ÷1000) para cálculos de dano.
- campos: value__, Raw_Divide1000, Raw_Divide100, Divided, DamageAttribute
  - GetRawValue: retorna value__ bruto do parâmetro
  - GetScaledValue: retorna Divided ou Raw_Divide100/1000 conforme tipo
  - IsDamageType: verifica se DamageAttribute está ativo
  - ConvertExchangeRate: aplica fator divisor apropriado ao parâmetro
  - ToDisplayString: formata para UI (ex: vida, mana, dano)
### TaskbarHero.Data.ESTAGEDIFFICULTY
Enum que define cinco níveis de dificuldade progressivos do jogo (Normal até Torment)
- campos: value__, NORMAL, NIGHTMARE, HELL, TORMENT, COUNT
  - GetValue: retorna valor inteiro do nível de dificuldade
  - ToString: converte enum para string do nome
  - FromValue: converte inteiro para valor do enum
  - IsEndgame: verifica se dificuldade é NIGHTMARE ou superior
  - GetMultiplier: retorna multiplicador de dano/HP baseado no nível
### TaskbarHero.EStageEnterResultType
Enum dos resultados possíveis ao tentar entrar em um estágio de combate
- campos: value__, Success, FailReasonEndStage, FailReasonNeedSoulStone, FailReasonNeedChestSpace, Failed
### TaskbarHero.EStageState
Máquina de estados que controla as fases sequenciais de um estágio de combate: spawn de inimigos, batalha com herói e reorganização entre waves
- campos: value__, NONE, MONSTERSPAWN, BATTLE, REORGANIZATION
  - GetNextPhase: avança para o próximo estado da sequência
  - IsPhaseActive: verifica se um estado específico está ativo
  - ResetPhase: reseta para NONE entre estágios
  - GetPhaseDuration: retorna tempo da fase em segundos
  - OnPhaseChanged: callback acionado ao trocar estado
### TaskbarHero.Data.EStageType
Enum que classifica tipos de estágios: NORMAL (fases comuns) e ACTBOSS (fases com chefes)
- campos: value__, NORMAL, ACTBOSS
### TaskbarHero.Effect.EffectController
Controla efeitos visuais de status e impacto em combate (shock, cura, congelamento, atordoamento, sangramento, hit effects).
- campos: m_modelSprite, m_hitEffectTransform, m_boundCenterTransform, m_boundBottomTransform, m_headUpEffectTransform, m_unitCollider, ShockObject, m_shockObjectScaleCorrectionRatio, HealEffectObject, FreezeEffectObject, m_freezeEffectAnimator, StunEffectObject, m_damageReductionHitEffect, BleedEffectCorrectByBoundY, BleedEffectObject, bfpt
  - ApplyShockEffect: exibe objeto de shock no transform do personagem e escala conforme ratio
  - ApplyHealEffect: ativa efeito de cura acima da cabeça via m_headUpEffectTransform
  - ApplyFreezeEffect: ativa animator de freeze e exibe objeto congelado
  - ApplyBleedEffect: spawna efeito de sangramento corrigido pela posição inferior (BoundY)
  - PlayHitEffect: anima efeito de impacto no transform de hit com redução de dano visual
### TaskbarHero.ObjectPooler.EffectPoolable
Poolable de efeito visual que segue um alvo por tempo limitado e se recicla automaticamente
- campos: LifeTime, m_effectSpriteRenderer, FollowTarget, bfkl, bfkm
  - Initialize/Setup: Configura LifeTime, FollowTarget, habilita SpriteRenderer ao ativar do pool
  - Reset/Cleanup: Reseta estado, desabilita renderer, retorna ao pool quando LifeTime expira
  - UpdatePosition (em LateUpdate): Sincroniza posição do renderer com FollowTarget
  - DecrementLifeTime: Reduz LifeTime a cada frame (bfkl provavelmente é contador ou flag 'tempo expirou')
### TaskbarHero.Effect.EffectStackType
Enum que define o comportamento de stacking quando múltiplos efeitos do mesmo tipo são aplicados (acumular, apenas um ativo, ou nenhuma regra especial)
- campos: value__, UseStack, UseOnlyOne, None
  - ToString: retorna o nome string do tipo de stacking
  - GetValue: retorna o valor inteiro subjacente do enum
  - ShouldStack: verifica se um novo efeito pode coexistir com instâncias existentes
  - ShouldReplace: determina se o novo efeito substitui o efeito anterior
  - CanApplyMultiple: valida se múltiplas instâncias são permitidas para este tipo
### TaskbarHero.Effect.EffectTransformType
Enum que define pontos de ancoragem para renderizar efeitos visuais de combate (impacto, HUD sobre cabeça, centro do corpo, pés).
- campos: value__, HitEffectPointType, BoundCenterType, HeadUpType, FootCenterType
  - GetAnchorTransform: retorna a transformação do ponto de ancoragem no personagem
  - ApplyEffectAt: aplica o efeito visual na posição especificada
  - IsHeadElement: verifica se é efeito de cabeça (HeadUpType)
  - GetWorldPosition: calcula posição mundial do ponto de ancoragem
  - ShouldFollowBone: determina se o efeito deve seguir o osso durante animação
### TaskbarHero.Combat.Projectile.ExplosionProjectile
Projétil que explode ao colidir, causa dano em área (AoE) usando raio de explosão.
- campos: WillDetectType, explosionRadius, bhgc
  - OnTriggerEnter: Detecta colisão com objetos do tipo WillDetectType
  - Explode: Ativa explosão e calcula dano em raio
  - DealDamageInRadius: Aplica dano aos inimigos dentro de explosionRadius
  - GetEnemiesInRadius: Busca todos os alvos detectáveis no raio de explosão
  - PlayExplosionEffect: Spawna partículas/som/efeito visual da explosão
### TaskbarHero.Tutorial.GuidingMonsterWave
Gerencia ondas de monstros guiados durante a fase tutorial do jogo.
- campos: bfll, bflm, bfln, bflo, bflp
  - SpawnWave: Instancia monstros em padrão/posição inicial
  - UpdateWave: Atualiza posições e comportamento dos inimigos a cada frame
  - CompleteWave: Marca onda como finalizada quando todos monstros morrem/saem
  - GetWaveProgress: Retorna quantidade de monstros restantes ou % progresso
  - SetWavePattern: Define movimento/agressividade dos monstros da onda
### TaskbarHero.Combat.HeroActiveSkill
Gerencia execução e ciclo de vida de uma habilidade ativa de combate do herói.
- campos: bgtc
  - mml: Execute/Cast - dispara a skill ativa
  - mne: CanUse/IsAvailable - valida cooldown e pré-requisitos
  - mnf: ApplyEffect/OnCast - aplica efeitos e dano
### TaskbarHero.Log.HeroDieLog
Registra e persiste dados sobre morte do herói (timestamp, causa, contexto) para analytics e save-states
- campos: bfog, bfoh
  - RecordDeath: Captura momento da morte com contexto (inimigo, posição, HP restante)
  - SaveDeathLog: Persiste dados em disco/serialização
  - GetLastDieInfo: Retorna detalhes da morte anterior para replay ou estatísticas
  - ClearLog: Limpa registros obsoletos ou ao novo run
### TaskbarHero.Combat.HunterBaseAtk
Sistema que gerencia o ataque base (BaseAtk) de um Hunter no loop de combate
  - mma: Executa ou dispara o ataque base
  - mlz: Calcula dano ou aplica efeitos do ataque
### TaskbarHero.Combat.HunterChargeTrap
Gerencia o carregamento e ativação de uma armadilha de caçador em combate
- campos: bgtd, bgte
  - mlz: Initialize — configura o estado inicial da armadilha
  - mng: Update — acumula/gerencia o tempo de carga
  - mnh: OnActivate — dispara/detona a armadilha quando carregada
  - mma: ApplyEffect — aplica dano ou status ao alvo atingido
### TaskbarHero.Combat.HunterCrossbowTurret
Torreta de besta controlada que dispara projéteis contra inimigos durante combate
  - mlz: FireCrossbowProjectile - dispara um projétil de besta em direção ao alvo
  - mma: UpdateTurretRotation - atualiza rotação da torrta para seguir/apontar ao inimigo
### TaskbarHero.Combat.Projectile.HunterExplosiveBolt
Projétil explosivo lançado por caçadores que causa dano em área ao impacto
- campos: bhen
  - mlz: Lógica de movimento/trajetória do projétil até impacto
  - mma: Aplicação de dano múltiplo em área ou detonação da explosão
### TaskbarHero.Combat.Projectile.HunterFrostBolt
Projectil de ataque de raio de gelo disparado por caçadores que se move até o alvo e aplica dano com efeito de congelamento
- campos: bheo
  - mlz: Move — atualiza posição do projectil a cada frame em direção ao alvo
  - mma: OnHit — detecta colisão, aplica dano e efeito de congelamento ao inimigo atingido
### TaskbarHero.Combat.HunterShockBolt
Skill de ataque especial do Hunter que dispara um projétil de choque contra inimigos, infligindo status Shock.
- campos: bgtg
  - mlz: Execute/Cast — dispara ou ativa o shock bolt direcionado ao alvo
  - mma: ApplyEffect/OnHit — aplica o status de choque quando o projétil acerta o inimigo
  - bgtg (field): duration ou shockIntensity — configuração de duração do shock ou intensidade do efeito
### TaskbarHero.Combat.KnightAegisField
Habilidade defensiva de cavaleiro que cria um campo de proteção (aegis) ao redor do personagem.
  - Activate (mlz): Ativa o campo de proteção
  - ApplyShield (mnj): Aplica redução/absorção de dano na área
  - Deactivate (mma): Desativa ou termina o efeito do campo
### TaskbarHero.Combat.KnightDashAtk
Habilidade de ataque do Cavaleiro que combina movimento rápido (dash) com golpe ofensivo contra inimigos
- campos: bgtp, bgtq, bgtr, bgts, bgtt, bgtu
  - Activate: inicia e executa o dash+ataque
  - Update: atualiza movimento e colisões durante execução
  - OnHit: aplica dano e knockback ao alvo acertado
  - CalculateDamage: computa dano com multiplicadores e stats
  - PlayEffect: reproduz animação e efeitos visuais do ataque
### TaskbarHero.Combat.KnightStrongAtk
Classe que implementa o ataque especial forte (Strong Attack) do cavaleiro no sistema de combate do jogo TaskbarHero.
- campos: bgug
  - mml: OnCastBegin — inicializa o ataque (posição, direção, consume `bgug` como recurso)
  - mlz: CalculateDamage — computa dano baseado em stats do cavaleiro
  - mma: OnUpdate — atualiza hitbox/animação durante execução
  - mol: OnCastEnd — finaliza o ataque e reseta o estado
### TaskbarHero.MODSOURCE
Enum que categoriza a origem/fonte de um modificador de atributo ou stat no personagem
- campos: value__, BASE, ITEM, ATTRIBUTE, PASSIVE, AccountStatus, StatusEffect, BuffSkill, ENVIROUNMENT
  - GetModifierType: retorna tipo da fonte (item, passive, statuseffect, etc)
  - Apply: aplica modificador baseado na origem
  - GetPriority: define ordem de precedência na aplicação de stats
  - IsTemporary: verifica se modificador é permanente ou temporário
  - StackWith: define regras de stacking com outras fontes
### TaskbarHero.Combat.MeleeTurret
Turreta de combate corpo-a-corpo que detecta alvos no alcance e executa ataques automaticamente com animação e cooldown.
- campos: m_animator, m_detectRangeRaw, m_maxAttackCount, m_attackAnimationSpeed, m_spawnDelayTime, m_attackDelayTime, m_targetDamageableType, bhcq, bhcr, bhcs, bhct, bhcu, bhcv, bhcw, bhcx, bhcy
  - myp: FindTargetInRange — detecta inimigos dentro do raio m_detectRangeRaw
  - myq: IsTargetValid — valida se alvo é do tipo m_targetDamageableType
  - myr: CanAttackNow — verifica se cooldown expirou e ataques < m_maxAttackCount
  - mys: DealDamageToTarget — aplica dano ao alvo detectado
  - edx: UpdateAttackCooldown — tick do timer m_attackDelayTime
### TaskbarHero.Monster
Gerencia unidades inimigas (monstros) com stats, tipo, IA de combate e ciclo de vida
- campos: bcwq, bcwr, cache, MonsterType, bcws, bcwt, bcwu, bcwv, bcww
  - Update: Loop de IA/comportamento por frame
  - TakeDamage: Recebe dano e atualiza HP
  - Attack: Executa ataque contra alvo
  - SetMonsterType: Inicializa tipo (afeta stats bcwq-bcww)
  - CalculateAI: Decide ação (atacar/mover/skill)
### TaskbarHero.Combat.MonsterActive
Gerencia estado, comportamento e combate de um monstro ativo em batida
- campos: bgyu
  - mma: Inicializar/Spawnar monstro com parâmetros
  - mml: Atualizar IA/comportamento a cada frame
  - mlz: Executar ataque contra jogador
  - mtg: Receber dano e atualizar saúde
  - mth: Morrer/Despawnar quando saúde <= 0
### TaskbarHero.Data.MonsterInfoData
Define atributos estáticos de um tipo de monstro (stats, recompensas, assets).
- campos: MonsterKey, MonsterName, MONSTERTYPE, RewardGold, RewardExp, SkillKey, AttackDamage, AttackSpeed, MaxLife, MovementSpeed, DeadSoundKey, PrefabPath, AnimatorPath
  - GetMonsterStats: retorna AttackDamage, AttackSpeed, MaxLife, MovementSpeed
  - GetRewards: retorna RewardGold + RewardExp
  - GetPrefabInstance: carrega/instancia o prefab do caminho PrefabPath
  - GetAnimator: retorna ou carrega animator do AnimatorPath
  - PlayDeadSound: toca áudio DeadSoundKey ao morrer
### TaskbarHero.Data.MonsterLoadData
Armazena dados de carregamento de monstros (prefab, animação) para instanciação em combate
- campos: MonsterKey, MonsterPrefab, MonsterAnimatorController
  - GetMonsterPrefab: retorna a referência do prefab do monstro para ser instanciado
  - GetAnimatorController: retorna o controller de animação associado ao monstro
  - GetMonsterKey: retorna o identificador único do monstro
  - LoadMonsterData: carrega/valida os dados do prefab e animador da classe
  - CreateMonsterInstance: utiliza os dados para instanciar o monstro na cena
### TaskbarHero.Manager.MonsterSpawnManager
Gerencia spawn, listagem e morte de monstros em ondas de combate, com suporte a bosses e monstros sumonados
- campos: m_spawnPositionMonster, MonsterList, DeadMonsterUnit, SummonedMonsterList, IsForceEnterBossWave, OnMonsterDie, OnBossMonsterSpawned, bfug, bfuh, bfui, bfuj, bfuk
  - SpawnMonster: Instancia um monstro na posição de spawn
  - SpawnBossMonster: Spawna um monstro boss (dispara OnBossMonsterSpawned)
  - RemoveDeadMonster: Move monstro morto para DeadMonsterUnit (pool/cache)
  - EnterBossWave: Ativa flag IsForceEnterBossWave e inicia fase de boss
  - UpdateMonsterList: Atualiza listas MonsterList e SummonedMonsterList por frame
### TaskbarHero.OpeningDirection
Controla a cinemática de entrada de herói em combate com efeitos visuais (tremor de câmera, dispersão de partículas, landing)
- campos: openingObject, landingEffect, npcSettings, monsterSettings, heroEntranceDelay, shakeDuration, shakeIntensity, scatterDuration, scatterSpeed, scatterRotationCount, scatterRotationDuration, scatterTargetScale, destroyOffsetX, bdes, bdet, bdeu
  - Initialize: configura efeitos, delays e settigns de combatentes
  - PlayHeroEntrance: executa animação de entrada do herói com delay configurável
  - ApplyShakeEffect: aplica tremor de câmera por duração e intensidade
  - PlayScatterEffect: executa dispersão visual (rotação + escala alvo) em duração
  - Cleanup: destroi objetos e offset visual após conclusão
### TaskbarHero.Combat.Projectile.Projectile_Arrow
Gerencia projétil de flecha com alvo específico no sistema de combate
- campos: specificTarget
  - Launch: Dispara flecha em direção ao specificTarget
  - Update: Atualiza posição e movimento do projétil
  - OnTrigger: Detecta colisão com o alvo
  - ApplyDamage: Aplica dano ao alvo atingido
  - Destroy: Remove projétil após impacto
### TaskbarHero.RESULTGRADETYPE
Enumeração que classifica resultados de ações (combate/skill) em 5 tiers relativos a um grade de referência
- campos: value__, Lower2Grade, Lower1Grade, SameGrade, Higher1Grade, Higher2Grade
  - GetGradeOffset: retorna o delta numérico (-2, -1, 0, +1, +2)
  - IsSuccessful: retorna true se grade >= SameGrade
  - IsCritical: retorna true se >= Higher1Grade
  - GetRewardMultiplier: aplica multiplicador de XP/loot baseado no tier
### TaskbarHero.UI.ReadyArrangePanel
Painel UI de preparação/arranjo de equipe pré-combate com visualização de DPS esperado.
- campos: m_dpsText, m_dpsTitle, m_previewText, bfza
  - lef: Carregar estado salvo do arranjo anterior
  - leg: Gerar/recalcular DPS baseado em equipamento selecionado
  - leh: Atualizar preview de texto quando equipment muda
  - lei: Inicializar componente com valores padrão do herói
### TaskbarHero.Combat.RepeatingAOESpawner
Spawna AOEs (danos em área) repetidamente em uma direção por uma duração limitada, com intervalo regular entre spawns.
- campos: m_aoePrefab, m_spawnDirection, m_spawnRangeRaw, m_totalDuration, m_spawnInterval, bgzx, bgzy, bgzz, bhaa, bhab, bhac
  - StartSpawning: Inicia o ciclo repetido de geração de AOEs
  - SpawnAOE: Instancia uma AOE na posição calculada
  - UpdateTimer: Controla intervalo de spawn e duração total
  - StopSpawning: Encerra o padrão ao atingir totalDuration
  - CalculatePosition: Calcula offset baseado em spawnDirection e range
### TaskbarHero.Data.SLOTTYPE
Enum que define os tipos de slots para equipamento/barra de ações (ataque base vs habilidade)
- campos: value__, BASEATTACK, SKILL
  - ToString: Retorna nome do tipo de slot como string
  - IsBaseAttack: Verifica se é slot de ataque base
  - IsSkill: Verifica se é slot de habilidade
  - GetSlotDamageType: Retorna tipo de dano do slot
  - CanEquip: Valida se item pode ser equipado neste tipo de slot
### TaskbarHero.Data.ScriptableObjectDataContainer
Gerenciador central Singleton que carrega e cacheia todos os dados do jogo (heróis, monstros, skills, buffs, gear, atributos) em memória via ScriptableObjects
- campos: Instance, m_logGradeSo, LogColorSettings, m_buffGroupDataPath, BuffGroupData, bgmo, SkillData, bgmp, UniqueModData, bgmq, m_gearUniqueModDataPath, bgmr, m_heroLoadDataPath, HeroLoadData, bgms, HeroIdleSpriteData
  - LoadAllData: inicializa e carrega todos os dados persistentes do disco
  - LoadHeroData: carrega sprites, stats e abilities dos heróis
  - LoadMonsterData: carrega atributos e comportamento de inimigos
  - LoadSkillBuffAndGearMods: carrega catalogo de skills, buffs e modificações de gear
  - ValidateDataIntegrity: valida se todas as referências estão carregadas corretamente
### TaskbarHero.Combat.SkillBehaviorContext
Contexto que armazena estado e restrições para execução de skills em combate (owner, distância, cooldown, ativação).
- campos: SkillOwner, SkillCastDistance, MaxCoolDown, ActivationValue, SkillCache, ActiveSkill, PlayerNormalAttackCountWhenInit
  - Initialize: configurar owner, distância máxima e cooldown inicial
  - CanCastSkill: validar se skill pode ser lançada (distância, cooldown, pontos de ativação)
  - CastSkill: executar skill ativa e aplicar cooldown
  - UpdateCooldown: decrementar/resetar cooldown da skill
  - GetActiveSkillFromCache: recuperar skill ativa do cache (performance)
### TaskbarHero.Data.SkillBuffType
Define os tipos de buffs para habilidades (Normal = sem buff, Buff = com buff ativo)
- campos: value__, Normal, Buff
  - IsBuffActive(): verifica se buff está ativo (true se Buff, false se Normal)
  - ApplyBuff(): transiciona de Normal para Buff
  - RemoveBuff(): transiciona de Buff para Normal
  - GetValue(): retorna value__ (identificador numérico do tipo)
  - ToString(): retorna nome legível ('Normal' ou 'Buff')
### TaskbarHero.Data.SkillInfoData
Armazena configuração completa de uma habilidade (nome, dano, animações, ativação, buffs)
- campos: SkillKey, SkillNameKey, SkillDescriptionKey, ActivationType, ActivationValue, DamageAttribute, DamageDeliveryType, SlotType, SkillBuffType, BuffGroupKey, Param1, Param2, Param3, Param4, Param5, Range
  - mjz: Validar dados da habilidade para consistência
  - mjz: Serializar/desserializar dados (JSON/binary)
  - mjz: Obter descrição formatada para UI
### TaskbarHero.Combat.SlayerAxeSpin
Implementa habilidade de ataque especial com machado giratório em combate
- campos: bgvl, bgvm
  - mpk: Inicia e ativa a rotação do spin
  - mnj: Aplica dano aos inimigos dentro do raio de rotação
  - mml: Atualiza velocidade angular (bgvl/bgvm)
  - mpl: Verifica colisões/acertos durante o spin
  - mma: Encerra e limpa o estado da habilidade
### TaskbarHero.Combat.SlayerBaseAtk
Define o ataque base de um Slayer no sistema de combate.
  - mma: Executar o ataque base (aplicar dano, efeitos, animação)
  - mlz: Obter ou calcular informações do ataque (dano base, range, cooldown)
### TaskbarHero.Combat.SnowstormArea
Área de efeito que spawna projéteis de neve em intervalos, aplicando congelamento a inimigos
- campos: m_freezingPercentDivided, SnowProjectile, SnowSpawnOffset, ProjectileSpawnIntervalTime, bhaj, bhak, bhal
  - mtw: Update - loop principal que gerencia spawn de projéteis por intervalo
  - mui: SpawnSnowProjectile - instancia novo projétil no offset definido
  - mtq: ApplyFreezing - aplica percentual de congelamento a entidades na área
  - mvc: OnAreaTrigger - ativa/desativa a área de efeito
  - mvd: Cleanup - destrói projéteis restantes ao sair da área
### TaskbarHero.Combat.SoulSnatchEffect
Efeito visual que anima uma alma sendo puxada visualmente de um ponto a outro durante combate, com linha conectiva e timing parametrizável.
- campos: m_soulSpriteRenderer, m_lineControllerPrefab, m_lineEndOffset, m_lineStartOffset, m_pullCurve, m_pullDuration, m_pullDestinationOffset, m_pullStartOffset, m_startDelay, m_endDelay, bham, bhan, bhao, bhap, bhaq, bhar
  - PlayEffect: Inicia o efeito de puxão de alma com delays
  - AnimateSoulMovement: Interpola posição da alma ao longo da curva m_pullCurve
  - CreateVisualLine: Instancia e posiciona o prefab de linha entre pontos
  - UpdateLineGeometry: Atualiza vértices/posição da linha a cada frame
  - CleanupEffect: Destroi objetos temporários ao fim da animação
### TaskbarHero.Data.StageInfoData
Armazenar e configurar dados de um estágio/fase de combate com ondas de monstros, boss, multiplicadores e sistema de drops
- campos: StageKey, StageNameKey, STAGETYPE, STAGEDIFFICULTY, Act, StageNo, StageLevel, WaveAmount, WaveMonsterAmount, Monsters, MonsterDropItemKey, FirstClearDropKey, MonsterDropItemRate, BossDropItemRate, BossDropItemKey, BossMonsterKey
  - mjz: Calcula multiplicador final do boss (produto dos 4 multiplicadores: dano × ouro × exp × HP)
### TaskbarHero.StageManager
Gerenciador central da fase de combate que orquestra heróis, câmera, backgrounds e controla o fluxo do estágio.
- campos: ForceResurrectionEffect, MainFollowCamera, HeroList, bdml, OnHeroRemovedFromHeroList, bdmm, bdmn, bdmo, m_stageBackGroundContainerDataPath, StageBackGroundContainers, bdmp, stageState, m_corpseOutOfCameraDistance, m_rightSpawnPositionX, bdmq, SpawnPosition_Hero
  - Start/OnEnable: Inicializa estágio, carrega backgrounds, posiciona câmera
  - Update/Loop principal (ief/ieg): Tick do estágio, polling de estado, atualizações de câmera
  - SpawnHero (ieh/iei): Instancia herói na SpawnPosition_Hero com verificação de m_rightSpawnPositionX
  - RemoveHeroFromList (iel/iem): Dispara OnHeroRemovedFromHeroList e aplica ForceResurrectionEffect se necessário
  - UpdateCameraFollow (iep/ieq): Atualiza MainFollowCamera baseado em posição de herói/corpo fora de m_corpseOutOfCameraDistance
### TaskbarHero.UI.StageWaveIconSliderController
Controlador UI que exibe e anima o progresso da wave/fase atual, com transição visual especial quando um boss aparece.
- campos: m_waveProgressSlider, m_bossIconSpriteAnimator, m_backGroundFillImage, m_sliderBottomArrow, m_defaultStageSliderFillSprite, m_normalBossSliderFillSprite, m_actBossSliderFillSprite
  - SetWaveProgress: atualiza o slider (m_waveProgressSlider) e troca sprite de preenchimento (normal vs boss)
  - AnimateBossIcon: ativa animação do sprite do boss (m_bossIconSpriteAnimator) quando boss é detectado
  - UpdateSliderFillSprite: troca entre defaultStageSliderFillSprite, normalBossSliderFillSprite, actBossSliderFillSprite conforme o tipo de inimigo
  - Initialize: setup inicial dos componentes UI e listeners de eventos de wave
  - OnWaveStateChanged: callback disparado quando wave/boss muda de estado
### TaskbarHero.Data.StatModGroupInfoData
Armazena metadados de um grupo de modificadores de status (buffs/debuffs) aplicáveis a heróis em combate
- campos: bgpz, bgqa, bgqb, bgqc, bgqd
  - bgpz: ID ou tipo identificador do grupo de modificador
  - bgqa: valor numérico do modificador (multiplicador ou delta de stat)
  - bgqb: duração em turnos ou tempo de expiração do efeito
  - bgqc: tipo de statística afetada (HP, ATK, DEF, SPD)
  - bgqd: ícone, sprite ou descritor visual do modificador no UI
### TaskbarHero.StatType
Define os tipos de atributos/stats de personagem usáveis em combate (dano, resistências, defesa, etc.)
- campos: value__, NONE, AttackDamage, AttackSpeed, CriticalChance, CriticalDamage, MaxHp, Armor, MovementSpeed, AreaOfEffect, BaseAttackCountReduction, CooldownReduction, SkillRangeExpansion, FireResistance, ColdResistance, LightningResistance
  - GetDisplayName: converte enum para string legível (ex: AttackDamage → 'Ataque')
  - GetIconPath: retorna caminho da textura do ícone do stat
  - IsResistance: retorna true se o stat é resistência (Fire/Cold/Lightning/Chaos)
  - IsDefensive: retorna true se é stat defensivo (Armor, DodgeChance, BlockChance)
  - GetBaseValue: retorna valor padrão/mínimo do stat para geração de item
### TaskbarHero.Data.StatusEffectInfoData
Armazena configuração de efeitos de status (buffs/debuffs) aplicáveis em combate
- campos: bgql, bgqm, bgqn, bgqo, bgqp, bgqq, bgqr, bgqs, bgqt
  - GetDuration: retorna duração do efeito em turnos
  - ApplyToTarget: aplica efeito ao alvo com magnitude
  - GetIcon: retorna sprite/ícone para exibição UI
  - IsActive: verifica se o efeito ainda está ativo
  - RemoveEffect: remove o efeito do alvo
### TaskbarHero.Status.StatusEffect.StatusEffectOverrideType
Enum que define comportamento quando um status effect é reaplicado: resetar duração ou não sobrescrever efeito existente
- campos: value__, InitDuration, NotOverride
  - ShouldResetDuration: retorna true se tipo é InitDuration
  - CanOverride: retorna false se tipo é NotOverride
  - Parse: converte string para StatusEffectOverrideType
### TaskbarHero.Status.StatusEffect.StatusEffectType
Enumeração que define tipos de efeitos de status aplicáveis em combate (congelação, queimadura, choque, sangramento, atordoamento).
- campos: value__, Chill, Freeze, Ignite, Shock, Bleed, Stun
  - GetDuration: retorna duração base do efeito em segundos
  - IsNegative: indica se o efeito prejudica o alvo
  - GetDamagePerTick: calcula dano contínuo para efeitos destrutivos
  - CanStack: verifica se múltiplas aplicações do mesmo efeito se acumulam
  - GetVisualEffect: retorna prefab ou ID da animação do efeito
### TaskbarHero.Combat.TurretSkillData
Armazena configuração de habilidade de torre/torrente que detecta inimigos, dispara projéteis e executa ataques com animação.
- campos: Projectile, DetectRadius, Target, AttackSpeed, AttackAnimationClip
  - Initialize: setup inicial dos parâmetros de ataque
  - DetectTargetsInRadius: verifica inimigos dentro de DetectRadius
  - SelectTarget: escolhe alvo prioritário da lista detectada
  - FireProjectile: instancia Projectile e dispara contra Target
  - ApplyAttackAnimation: executa AttackAnimationClip com sincronismo de AttackSpeed
### TaskbarHero.UI.UI_Arrange
Tela de gerenciamento de lineup de heróis com desbloqueio, compra e organização tática
- campos: OnHeroUnlocked, m_arrangeState, bgem, m_heroDetailInfoPanel, button_subTabArrange, m_petArrange, m_arrangePageHeroViewSetter, m_arrangeTabHeroPortrait, PurchaseBlockDemo, m_readyArrangePanel, go_NotPurchased, TextUnlockCost, go_ArrangeCooldown, go_BossCanNot, button_TryUnLockHero, button_Arrange
  - lqe: Initialize() - inicia state e carrega dados de heróis
  - lqf: UnlockHero() ou PurchaseHero() - desbloqueia herói via custo
  - lqg: ShowHeroDetailPanel() - exibe m_heroDetailInfoPanel com info do herói
  - lqh: OnArrangeClick() - processa confirmação de arranjo e atualiza lineup
  - hsr: RefreshHeroSlots() - redesenha todos os slots com estado atual
### TaskbarHero.UI_Stage
Gerencia a interface visual do estágio/combate, incluindo saúde, notificações toast, feedbacks de início/fim e menus contextuais
- campos: HealthUI, feedback_stageStart, feedback_stageEnd, m_hoverObjectToggler, m_bottomStageQuickMenu, m_iconSliderController, m_portalRepeatSetter, m_hoverStageTextPanel, m_boxAlphaController, LogTopPosition, LogBottomPosition, m_bottomToastMessagePanelCorrectionValue, m_bottomToastMessagePanel, text_StageName, button_ShowMain, bottomPanel
  - hva: ShowStageStartFeedback — exibe feedback quando estágio inicia
  - hvb: ShowStageEndFeedback — exibe feedback quando estágio termina
  - hvc: UpdateHealthBar — atualiza exibição da barra de vida
  - hvd: SwitchToBossMode — alterna entre health bar normal e boss
  - hve: DisplayToastNotification — mostra mensagem toast no painel inferior
### TaskbarHero.Unit
Entidade jogável/inimiga com sistema de vida, habilidades, projéteis, buffs e efeitos visuais para combate tático.
- campos: OnUnitDie, View, m_viewSpriteRenderer, unitBoundsCollider, showBoundsGizmo, Transform_SpawnTurret, transform_EffectPosition, transform_CurvedProjectilePosition, transform_StraightProjectilePosition, transform_ProjectileTargetPosition_1, transform_ProjectileTargetPosition_2, bcxb, bcxc, bcxd, transform_DamageTexturePosition, transform_GoldTexturePosition
  - grj: Inicializar unidade (Awake/Start)
  - grk: Aplicar dano (TakeDamage)
  - grl: Disparar projéteis (ShootProjectile/SpawnProjectile)
  - grm: Aplicar/remover buff (ApplyBuff/RemoveBuff)
  - gro: Atualizar loop de game (Update movimento/comportamento)
### TaskbarHero.Combat.VoidMeleeTurret
Torre de combate corpo-a-corpo que persegue e ataca inimigos em tema Void
- campos: m_modelTransform
  - myq: PlayAttackAnimation — rotaciona m_modelTransform e executa animação
  - mzn: DealDamage — aplica dano ao alvo detectado
  - mzo: ResetAttackCooldown — reinicia timer e prepara próximo ataque
### TaskbarHero.Combat.WizardBaseAtk
Implementa o ataque básico de um mago no sistema de combate do jogo
  - mma: ExecuteAttack - executa o ataque mágico do mago
  - mml: CalculateDamage - calcula dano baseado em stats/spell
  - mlz: ValidateTarget - valida alvo (alcance, vivo, etc)
  - mqe: ApplyEffect - aplica efeito/status/impacto ao alvo
### TaskbarHero.Combat.WizardSnowstorm
Mago especializado em habilidade de neve/gelo que causa dano em área em combate
- campos: bgwz
  - mlz: Inicia/processa a habilidade principal de tempestade de neve (snowstorm cast)
  - mma: Gerencia movimento e animação de ataque do mago durante habilidade

## economia  (51)
### TaskbarHero.UI.AlchemyGoldText
Exibe e anima texto de ganho de ouro com fade-in/fade-out automático
- campos: m_text, m_canvasGroup, m_displayDuration, m_fadeOutDuration, bfwl
  - Show: Inicia exibição do texto, configura canvasGroup alpha=1 e inicia corrotina de timeout
  - kxe: Corrotina que aguarda displayDuration e chama o fade-out
  - kxf: Corrotina de fade-out que decrementa canvasGroup.alpha durante fadeOutDuration
### TaskbarHero.AlchemyReceipt
Gerencia receipts de alquimia com rastreamento de mudanças de ouro e validação de identidade única
- campos: o_uniqueId, o_addedGold, uniqueId, addedGold
  - GetRecipeId: retorna uniqueId da receita
  - GetGoldReward: retorna addedGold como recompensa
  - RevertToOriginal: restaura o_uniqueId e o_addedGold aos campos atuais
  - ValidateRecipe: valida integridade da receita
  - ApplyModifiers: aplica modificadores ao ouro/id da receita
### TaskbarHero.BackendPostItem
Representa um item de inventário/drop para sincronização com o servidor backend
- campos: itemID, itemName, count
  - ToJSON: serializa itemID, itemName, count para envio ao backend
  - Validate: verifica se item e count são válidos antes de postar
  - Send: envia dados do item para endpoint do servidor
  - HandleResponse: processa confirmação/erro da resposta backend
  - Clone: cria cópia do item para operações locais
### TaskbarHero.BoxData
Armazena dados de uma caixa de recompensa reclamável com sincronização entre campos legíveis e ofuscados
- campos: o_itemId, o_itemKey, o_claimableAt, o_rewardItemId, o_rewardItemKey, o_isGet, itemId, itemKey, claimableAt, rewardItemId, rewardItemKey, isGet
  - CopyToObscured: Sincroniza dados dos campos normais (itemId, etc) para os ofuscados (o_itemId, etc)
  - CopyFromObscured: Sincroniza inverso - copia dos campos o_ para normais
  - CanClaim: Verifica se claimableAt foi atingido para permitir reclamação
  - ClaimReward: Marca a recompensa como reclamada (isGet=true) e sincroniza
  - Validate: Verifica integridade entre pares normal/obfuscado
### TaskbarHero.Data.CraftingRecipeInfoDataMap
Container/dicionário que armazena e gerencia receitas de crafting do jogo
  - GetRecipe: busca receita por ID
  - AddRecipe: adiciona nova receita ao mapa
  - RemoveRecipe: remove receita existente
  - ContainsRecipe: verifica se receita existe
  - GetAllRecipes: retorna todas as receitas disponíveis
### TaskbarHero.Data.CubeLevelInfoData
Armazena requisitos de progressão de nível para cubos: nível atual e XP necessário para próximo rank
- campos: Level, ExpForLevelUp
  - GetExpRequired: retorna ExpForLevelUp do nível especificado
  - GetLevel: acessa o Level atual
  - IsAtMaxLevel: verifica se Level alcançou limite
### TaskbarHero.Log.CubeOfferingLog
Registra evento de oferenda/sacrifício de cubos com ganho de recursos
- campos: bfny, bfnz, bfoa
  - kha: Serialize — formata log de oferenda em string para exibição/arquivo
  - khb: Deserialize — reconstrói evento de oferenda a partir de dados persistidos
### TaskbarHero.CubeRecipeResource
Armazena metadados de uma receita de crafting (tipo + ícone) para o sistema de combinação de cubos de recursos.
- campos: RecipeType, SpriteIcon
  - GetRecipeType: retorna o tipo/ID da receita
  - GetIcon: retorna o sprite para exibição na UI
  - GetCraftingCost: retorna custo em recursos da receita
  - CanCraft: valida se há recursos suficientes
  - ExecuteCraft: aplica a receita ao inventário
### TaskbarHero.EasySaveData.CubeRecipeSaveData
Armazenar progresso de desbloqueio de receitas de cubo (crafting/decomposição)
- campos: CubeRecipeTypeInt, CubeKey, MaxUnlockRecipeKey
  - IsRecipeUnlocked: verifica se receita do cubo foi desbloqueada
  - UnlockNextRecipe: desbloqueia próxima receita na sequência
  - GetRecipeType: retorna tipo/categoria da receita (Int)
  - GetMaxUnlockedKey: retorna ID da receita máxima desbloqueada
  - ResetRecipeProgress: reseta progresso de desbloqueio de receitas
### TaskbarHero.UI.CubeTriggerToggleButton
Botão de compra/ativação de skill com custo em ouro; toggle entre estados ativo/inativo com feedback visual de acessibilidade
- campos: m_iconImage, m_offerringNoticeObject, m_offeringGoldText, m_offerringHoverTextColor, m_offerringDefaultTextColor, m_toggleOffTextColor, m_needMoreGoldTextColor, bfwa, bfwb, bfwc, bfwd, bfwe
  - OnClick: Processa compra da skill se tiver ouro suficiente
  - UpdateDisplay: Atualiza cores e texto do preço baseado em estado (afford/lack gold/toggled off)
  - CheckGoldRequirement: Valida se jogador tem ouro para custo e retorna bool
  - ShowOfferingNotice: Exibe notificação/aviso de oferta especial
  - SetButtonState: Alterna entre ativo/inativo e aplica cor correspondente
### TaskbarHero.CurrencyData
Armazena dados de uma moeda/recurso monetário com identificador e ícone visual para exibição na UI da taskbar.
- campos: CurrencyKey, SpriteIcon
  - Constructor(CurrencyKey, SpriteIcon): inicializa dados da moeda
  - GetCurrencyKey(): retorna o identificador único da moeda
  - GetSpriteIcon(): retorna o sprite do ícone para renderizar
  - SetCurrencyKey(key): atualiza o identificador
  - SetSpriteIcon(sprite): atualiza o ícone visual
### TaskbarHero.Data.CurrencyInfoData
Armazenar configuração de um tipo de moeda/currency (chave, nome, descrição, quantidade inicial, ícone UI).
- campos: CurrencyKey, CurrencyNameStringKey, Description, InitialAmount, IconPath
  - mjz: Inicializar instância com validação de campos obrigatórios
  - GetCurrencyDisplay: Retornar nome localizado + ícone para UI
  - ValidateCurrencyData: Verificar integridade dos dados
  - ApplyInitialAmount: Registrar no sistema de economia
  - GetLocalizedName: Resolver chave de string para idioma ativo
### TaskbarHero.EasySaveData.CurrencySaveData
Armazena tipo e quantidade de uma moeda do jogo para o sistema de salvamento.
- campos: Key, Quantity
  - GetKey: retorna o identificador da moeda (ouro, gemas, etc)
  - SetKey: define qual tipo de moeda este dado representa
  - GetQuantity: retorna a quantidade armazenada
  - SetQuantity: atualiza a quantidade de moedas
  - Constructor(key, quantity): inicializa um registro de moeda
### TaskbarHero.Data.DropInfoData
Define configuração de drops de loot com pesos probabilísticos, tipos de recompensa e condições por herói
- campos: DropKey, REWARDTYPE, RewardKey, Weight, HeroKeyCondition, DropType
  - mjz (guessed ValidateDropCondition): Valida se HeroKeyCondition é atendida para o drop ocorrer
  - GetDropChance: Retorna probabilidade normalizada usando Weight
  - IsEligibleForHero: Boolean que verifica compatibilidade herói + DropType + condição
  - GetRewardData: Retorna instância de recompensa via RewardKey e REWARDTYPE
  - CalculateWeight: Normaliza Weight para 0-1 range em loot table
### TaskbarHero.StatusSystem.EAccountStatus
Enum que define tipos de buffs/modificadores de conta (ouro, exp, drops, stats de herói) aplicáveis ao jogador.
- campos: value__, IncreaseGoldAmount, AdditionalGold, IncreaseExpAmount, AdditionalExp, DropChanceNormalChest, DropChanceStageBossChest, WaveCountReduction, WaveMonsterAmount, MaxAmountNormalChest, MaxAmountStageBossChest, MaxAmountActBossChest, CubeExpPercent, CubeAlchemyGoldPercent, AllHeroMoveSpeed, AllHeroAttackSpeed
  - GetValue: retorna valor numérico atual do buff
  - SetValue: define novo valor do modificador
  - Apply: aplica o buff ao status da conta
  - GetDescription: retorna rótulo legível (UI)
  - IsActive: verifica se o buff está ativo/desbloqueado
### TaskbarHero.EAddCubeResult
Enum de retorno para operações de adicionar itens a recipientes de crafting/síntese/alquimia, com códigos de erro específicos
- campos: value__, Success, AlreadyEquipItemIn, AlreadyEquipItemInWhenMoreMaterial, CannotInputInCubeEquipItem, NotMatchAlchemy, NotMatchSynthesis, NotMatchSynthesisItemGrade, NotMatchSynthesisNowSynthesisType, NotMatchCrafting, NotMatchDecoration, NotMatchEngraving, NotMatchInscription, NotMatchOffering, NotMatchExtraction, NotValidItem
  - TryAddItemToCube: valida e adiciona item ao recipiente
  - ValidateItemForRecipe: verifica compatibilidade com receita
  - CheckMaterialRequirements: confirma se materiais atendem requisitos
  - ValidateSynthesisType: valida tipo de síntese/crafting
  - GetErrorMessage: traduz código de erro para UI
### TaskbarHero.EBackendPostRewardResult
Enum que representa os estados de resultado ao receber/processar recompensas do backend (sucesso, falha genérica, ou falha por falta de espaço no inventário)
- campos: value__, Success, Fail, NotEnoughInventorySpace
  - IsSuccess: verifica se o resultado foi bem-sucedido
  - IsFail: verifica se houve falha genérica
  - IsInventoryFull: verifica especificamente se o inventário está cheio
  - GetErrorMessage: retorna mensagem de erro baseada no estado
  - ToString: converte enum para string legível
### TaskbarHero.EGoldCurrencySource
Enum que define as diferentes fontes de obtenção de moeda de ouro no jogo (combate, alquimia, recompensas offline).
- campos: value__, MonsterKill, CubeAlchemy, OfflineReward
  - GetMultiplier: retorna multiplicador de ganho por fonte
  - IsOfflineSource: verifica se a fonte funciona offline
  - GetSourceIcon: retorna ícone/sprite da fonte
  - GetRewardAmount: calcula valor de ouro pela fonte
### TaskbarHero.Data.EMaterialType
Define tipos de materiais coletáveis (decoração, pedras de alma, gravações, inscrições, oferendas, crafting) para sistemas de economia e artesanato
- campos: value__, DECORATION, SOULSTONE, ENGRAVING, INSCRIPTION, OFFERING, CRAFTING, NONE
  - IsValidMaterial: valida se o tipo de material é permitido no contexto
  - GetCraftingValue: retorna valor ou peso do material para receitas
  - CanBeUsedForCrafting: determina se o material pode ser artesanato
  - GetMaterialRarity: retorna nível de raridade ou drop rate
  - IsOfferingMaterial: verifica se pode ser usado em oferendas/boss
### TaskbarHero.ExchangeIngredient
Representa um ingrediente disponível para troca/comércio no sistema de economia do jogo.
- campos: UniqueId, Quantity
  - GetUniqueId: retorna o ID único do ingrediente
  - GetQuantity: retorna a quantidade disponível
  - SetQuantity: altera a quantidade após troca
  - CanTrade: valida se a troca é possível
  - ExecuteTrade: processa a troca de ingredientes
### TaskbarHero.Data.ExtractionCostInfoData
Armazena custo de extrair/desmontar materiais de equipamentos (gear) por tier
- campos: ExtractionKey, GearGroup, MaterialType, Tier, Cost
  - GetCost: Retorna custo da extração
  - GetMaterialType: Retorna tipo de material extraível
  - GetTier: Retorna nível de dificuldade/qualidade
  - GetGearGroup: Retorna grupo de equipamento-fonte
  - IsValidForGear: Valida se extração é possível neste equipamento
### TaskbarHero.Data.ExtractionCostInfoDataMap
Mapeia informações sobre custos e requisitos necessários para realizar extração de recursos/itens em localidades ou inimigos
  - GetExtractionCost: retorna o custo (moeda/recursos) necessário para uma extração
  - GetRequiredLevel: valida nível mínimo do personagem para extrair
  - GetRewardTable: mapeia recursos possíveis de serem extraídos e probabilidades
  - IsExtractionAvailable: verifica desbloqueio/disponibilidade de um tipo de extração
### TaskbarHero.Data.GearTypeScaleInfoData
Armazena escalas de recompensa de ouro/experiência por tipo de equipamento, calculando automaticamente os ratios para o sistema de economia
- campos: GearType, AlchemyGoldScale, CubeExpScale, CalculatedAlchemyGoldScaleRatio, CalculatedCubeExpScaleRatio
  - mjz: CalculateScaleRatios() — calcula os ratios derivados (CalculatedAlchemyGoldScaleRatio e CalculatedCubeExpScaleRatio) a partir das escalas brutas
### TaskbarHero.InventoryConsumeResult
Resultado de tentativa de consumir item/recipe do inventário: sucesso, erro, items removidos, IDs inválidos.
- campos: isSuccess, o_error, removed, o_removed, invalidUniqueId, o_invalidRecipeId
  - GetErrorMessage: retorna mensagem de erro se isSuccess=false
  - GetRemovedItems: retorna lista/estrutura de items removidos
  - GetInvalidRecipeId: retorna ID da recipe que falhou validação
  - GetInvalidUniqueId: retorna ID único do item que não pôde ser consumido
  - Validate: verifica se resultado tem estado consistente (removed vs invalidUniqueId)
### TaskbarHero.InventoryExchange
Gerencia trocas e permutas de itens no inventário do jogador
  - Execute: processa a troca de itens entre inventários ou NPC
  - ValidateExchange: verifica se os itens/recursos necessários estão disponíveis
  - GetRequiredItems: retorna lista de itens que devem ser dados
  - GetRewardItems: retorna lista de itens que serão recebidos
  - CanExchange: valida condições pré-requisito antes de permitir troca
### TaskbarHero.InventoryExchangeRequest
Requisição para trocar ou combinar um item do inventário por outro item resultante
- campos: o_resultItemId, o_itemKey, resultItemId, itemKey
  - Validate: verifica se o item ofertado existe e se a troca é válida
  - Execute: aplica a troca no inventário do player
  - GetRequiredItemKey: retorna qual item é necessário para a troca
  - GetRewardItemId: retorna ID do item que o player recebe
  - IsTradeComplete: confirma se a transação foi concluída
### TaskbarHero.Data.InventoryInfoData
Armazenar dados de um item desbloqueável do inventário com seu índice e custo de desbloqueio
- campos: Index, CostForUnLock
  - CanAfford: verifica se o custo de desbloqueio pode ser pago
  - IsUnlocked: retorna se o item já foi desbloqueado
  - Unlock: desbloqueia o item ao custo especificado
  - GetCostForUnlock: retorna o custo atual de desbloqueio
### TaskbarHero.InventoryUploadToSteamResult
Armazena resultado de uma tentativa de sincronização/upload do inventário do jogador com a plataforma Steam.
- campos: isSuccess, o_error, o_failedItemIds, o_failedItemKeys, removed, uploaded, o_removed
  - GetFailedItemIds: retorna lista de IDs dos itens que falharam no upload
  - GetFailedItemKeys: retorna chaves dos itens que falharam
  - GetUploadedItems: retorna lista de itens enviados com sucesso
  - GetRemovedItems: retorna itens removidos durante a sincronização
  - GetErrorMessage: retorna descrição do erro ocorrido, se houver
### TaskbarHero.Data.ItemInfoData
Define metadados completos de um item: tipo, raridade, drop-rate, tradabilidade e sincronização com servidor.
- campos: ItemKey, ITEMTYPE, GRADE, PARTS, GEARTYPE, GearGroup, ItemSynthesisType, NameKey, DescriptionKey, GearKey, DropKey, DropCooldown, Level, IsSteamItem, IconPath, IsDeletedInServer
  - mkf: ValidateItemState — verifica se item é deletado no servidor ou bloqueado de trade
  - mjz: DeserializeFromServer — desserializa item que veio do servidor
  - GetDropCooldownStatus — calcula se cooldown de drop expirou
  - IsTradeableOnMarket — boolean: pode ser vendido/trocado considerando flags temporários
### TaskbarHero.Data.ItemLevelScaleInfoData
Armazena escalas de recompensas (ouro/exp) e seus ratios calculados para um item com base em seu nível
- campos: ItemLevel, AlchemyGoldScale, CubeExpScale, CalculatedAlchemyGoldScaleRatio, CalculatedCubeExpScaleRatio
  - CalculateScaleRatios: Calcula CalculatedAlchemyGoldScaleRatio e CalculatedCubeExpScaleRatio a partir das escalas brutas
  - GetAdjustedAlchemyGold: Retorna ouro de alquimia multiplied by ratio calculado
  - GetAdjustedCubeExp: Retorna exp de cubo multiplied by ratio calculado
  - ApplyItemLevelModifier: Aplica o multiplicador de ItemLevel às escalas
  - InitializeDefaults: Popula valores padrão se escalas estão vazias
### TaskbarHero.Data.ItemTypeScaleInfoData
Armazena configuração de escalas de recursos (ouro de alquimia e cubos de XP) para cada tipo de item do jogo
- campos: ItemType, AlchemyGoldScale, CubeExpScale, CalculatedAlchemyGoldScaleRatio, CalculatedCubeExpScaleRatio
  - mjz: CalculateScaleRatios() — normaliza e calcula os ratios a partir dos scales brutos
  - Alternativa: ApplyScalesToItem() — aplica as escalas a um drop de recurso
  - Alternativa: ValidateScaleData() — valida consistência dos valores de escala
### TaskbarHero.UI.MainRecipeSlotButton
Botão de slot de receita na taskbar com indicador visual de notificações para crafts disponíveis
- campos: bfwu, bfwv, bfww, m_recipeComboBoxButtonRedDot
  - kzi: UpdateRecipeDisplay - Atualiza visual e dados da receita exibida
  - kzj: ShowRedDot - Ativa o indicador de notificação vermelho
  - kzk: HideRedDot - Desativa o indicador de notificação
  - kzl: OnButtonClick - Processa interação/clique do jogador
  - kzm: RefreshRecipeState - Valida disponibilidade de ingredientes
### TaskbarHero.MarketDataEntry
Armazena dados de item de marketplace com suporte a criptografia/obscurecimento de dados sensíveis (chaves e IDs)
- campos: o_itemKey, o_originalItemId, o_marketableAfterTimestamp, itemKey, originalItemId, marketableAfterTimestamp
  - CopyFromObscured: restaura campos claros a partir dos campos obscured (o_*)
  - IsMarketable: verifica se o timestamp atual passou de marketableAfterTimestamp
  - GetItemKey: getter para itemKey com validação/desofuscação
  - ReEncrypt: re-ofusca os campos obscured com nova chave
  - Validate: valida integridade entre pares (itemKey vs o_itemKey, etc)
### TaskbarHero.Data.MaterialInfoData
Armazena informações de um material do jogo (ID, quantidade e raridade ou preço).
- campos: bgoq, bgor, bgos
  - GetMaterialId: retorna o ID único do material
  - GetQuantity: retorna a quantidade em posse
  - GetRarity/Price: retorna o nível de raridade ou valor de venda
  - IsValid: verifica se os dados do material são válidos
  - Clone: cria uma cópia da instância
### TaskbarHero.OfflineRewardHeroResult
Armazena resultado de recompensa offline: experiência e nível ganhos por um herói durante ausência do jogador
- campos: HeroKey, StartLevel, EndLevel, StartExp, EndExp, StartMaxExp, EndMaxExp, AddedExp
  - GetTotalExpGained: calcula experiência total ganho (EndExp - StartExp)
  - GetLevelUpCount: retorna quantidade de níveis subidos (EndLevel - StartLevel)
  - IsLevelUp: verifica se houve aumento de nível
  - ApplyToHero: aplica as recompensas ao banco de dados do herói
  - GetExpToNextLevel: calcula experiência faltante para próximo nível
### TaskbarHero.Data.OfflineRewardInfoData
Armazena dados de recompensas recebidas enquanto o jogador estava offline (ouro, XP, itens).
- campos: bgot, bgou, bgov, bgow, bgox
  - bgot: Timestamp ou duração do período offline
  - bgou: Quantidade de unidades/moeda ganha
  - bgov: Valor total da recompensa
  - bgow: Ouro/wealth acumulado
  - bgox: Valor máximo ou extra bonus
### TaskbarHero.OfflineRewardResult
Armazena recompensas ganhas pelo jogador enquanto offline, incluindo ouro e resultados de heróis em um período de tempo específico.
- campos: HasReward, RewardSec, StartSteamUnixSec, EndSteamUnixSec, Gold, HeroResults
  - CalculateOfflineRewards: calcula recompensas baseado no intervalo StartSteamUnixSec até EndSteamUnixSec
  - IsRewardValid: valida se HasReward é verdadeiro e o período é válido
  - ApplyRewardToPlayer: aplica Gold e HeroResults ao estado do jogador
  - GetTotalRewardValue: retorna valor total de recompensa em moeda/recursos
  - ValidateTimestamps: garante que StartSteamUnixSec < EndSteamUnixSec
### TaskbarHero.EasySaveData.PendingEnchantSaveData
Rastrear encantamentos de equipamentos em andamento, aguardando confirmação/processamento do servidor antes de persistência final.
- campos: MaterialUniqueId, GearUniqueId, RecipeType, Enchant, TxId, TimestampTicks
  - Validate: verifica integridade dos IDs e timestamp antes de commitar
  - GetEnchantResult: obtém o resultado final do encantamento após processamento
  - RollbackOnFailure: desfaz o encantamento se a transação (TxId) falhar no servidor
  - SerializeForServer: codifica dados pendentes em formato de rede para envio
  - MatchesGearAndMaterial: valida correspondência entre GearUniqueId e MaterialUniqueId
### TaskbarHero.RecipeSlot
Interface de slot de receita alquímica com sistema de desbloqueio e seleção de craft
- campos: usm, text_LockCondition, text_Name, text_Name2, button_select, image_IconAlchemy, go_Material, BackgroundImage, UnlockImage, TipTooltipTrigger, bdgx, bdgy, bdgz, bdha, bdhb, bdhc
  - hra: Initialize — carrega os dados da receita no slot
  - hrb: DisplayRecipe — atualiza UI (nome, ícone, condição de bloqueio)
  - hrc: CheckUnlocked — verifica se receita está desbloqueada
  - hrd: OnSelectButton — handler do botão de seleção
  - hre: ShowMaterials — exibe materiais necessários do go_Material
### TaskbarHero.UI.RecipeSlotButton
Slot de UI para exibir e interagir com receitas de craft/construção, mostrando status (locked/unlocked/selecionado), custo em ouro e respondendo a eventos de mouse.
- campos: m_iconImage, m_backgroundImage, m_mouseHoverCanSelectSprite, m_defaultSprite, m_lockedSprite, m_selectedSprite, m_text, m_unlockHoverGoldText, m_noticeCubeLowLevel, m_textMouseHoverColor, m_textDefaultColor, m_textEnoughGold, m_textNeedMoreGold, m_textLockColor, m_lockIconSprite, m_canUnlockIconSprite
  - kzk: RefreshDisplay — atualiza sprites/cores baseado em estado (locked/selected/default)
  - kzr: SetLocked — bloqueia slot e desabilita interação
  - kzs: SetSelected — marca como selecionado e atualiza visual
  - kzt: UpdateGoldDisplay — mostra custo e cor baseado em ouro disponível
  - kzm: OnRecipeClick — confirma craft/construção ao clicar
### TaskbarHero.EasySaveData.RemakeTradingStashSaveData
Salva o estado de um slot no inventário/stash de comércio do jogo (item armazenado + desbloqueio).
- campos: Index, ItemUniqueId, IsUnLock
  - Index: getter do slot de armazenamento
  - ItemUniqueId: getter/setter do ID único do item neste slot
  - IsUnLock: getter do estado de desbloqueio do item
  - dwk_serialize: persiste para disco via EasySave
  - dwk_deserialize: carrega do disco e restaura estado
### TaskbarHero.ResourceManager
Gerenciador centralizado de recursos econômicos, dados de crafting (cube recipes, síntese), ativos visuais (sprites, texturas, tier images) e áudio
- campos: CurrencyData, bdcs, m_itemIconSpriteAtlas, m_runeIconSpriteAtlas, CubeRecipeTriggerButtonData, bdct, CubeRecipeData, bdcu, SynthesisItemData, bdcv, m_enchantSlotSpriteInfos, bdcw, m_tierImageContainer, SoundClipData, bdcx, go_DamageTexture
  - GetCurrencyAmount: Retorna quantidade de moeda/gold do jogador
  - GetItemIconSprite: Obtém sprite de ícone do item no atlas
  - GetCubeRecipeData: Retorna dados completos de uma receita de crafting
  - PreloadAllResources: Pré-carrega todos os dados e sprites em memória
  - GetSoundClip: Recupera clip de áudio por tipo (dano, ganho, síntese)
### TaskbarHero.Data.StashInfoData
Armazena metadados de um slot de baú/cofre desbloqueável, incluindo seu índice de posição e o custo em moeda do jogo para liberar acesso.
- campos: Index, CostForUnLock
  - IsUnlocked() - verifica se o stash já foi desbloqueado
  - GetIndex() - retorna o índice do slot no inventário
  - GetCostForUnLock() - retorna o custo em moeda para desbloquear
  - TryUnlock(playerCurrency) - tenta desbloquear se jogador tem moeda suficiente
  - GetDisplayName() - retorna nome exibível do stash (ex: 'Baú 2')
### TaskbarHero.TradingStashCache
Gerencia cache de slots de trading em um stash de inventário, disparando eventos quando itens mudam.
- campos: OnTradingStashSlotChanged, bfdq, bfdr, bfds
  - AddItem: adiciona item ao slot e notifica mudança
  - RemoveItem: remove item do slot e dispara evento
  - UpdateSlot: atualiza dados de um slot específico
  - NotifySlotChanged: dispara OnTradingStashSlotChanged com novo estado
  - GetSlotData: retorna dados em cache de um slot por índice
### TaskbarHero.Data.TradingStashInfoData
Armazenar dados de um slot de stash/cofre desbloqueável por custo em moeda do jogo
- campos: Index, CostForUnLock
  - mjz: Desbloqueia o slot pagando CostForUnLock
  - GetUnlockCost: Retorna o custo necessário para desbloquear
  - IsUnlocked: Verifica se o slot já foi desbloqueado
  - Initialize: Configura Index e CostForUnLock iniciais
  - CanAffordUnlock: Valida se o jogador tem recursos suficientes
### TaskbarHero.Data.TradingStashInfoDataMap
Mapeia e persiste dados do stash de comerciantes do jogador (itens, quantidades, preços).
  - Serialize: Converte stash de trading para formato persistível (JSON/binário)
  - Deserialize: Carrega dados do stash de disco pra estrutura C#
  - GetStashItems: Retorna lista de itens no stash com metadados
  - UpdateStashInfo: Sincroniza mudanças no stash (itens adicionados/removidos)
  - ClearStash: Limpa ou reseta dados do stash de comércio
### TaskbarHero.EasySaveData.TradingStashSaveData
Dados persistidos de um slot individual no estoque de comércio, rastreando o item armazenado e status de desbloqueio
- campos: Index, ItemUniqueId, IsUnLock
  - GetItemUniqueId(): retorna o ID único do item armazenado neste slot
  - IsSlotUnlocked(): verifica se o slot está desbloqueado para uso
  - SetItem(itemId): atribui um item ao slot por seu ID
  - ClearSlot(): remove o item do slot (ItemUniqueId = null/0)
  - GetSlotIndex(): retorna a posição do slot no stash
### TaskbarHero.UI.UI_Shop
Gerencia a interface visual da loja: exibição, fechamento e callbacks quando a loja é escondida.
- campos: OnShopHideChanged, bfzm, m_closeButton
  - hlt: ShowShop ou SetupShop — inicializa/abre a interface de loja
  - hly: HideShop ou OnCloseButtonClicked — esconde a loja e dispara OnShopHideChanged
  - hlw: OnEnable ou Update — escuta eventos/atualiza estado da UI
### TaskbarHero.UI.UI_TradingStash
Interface UI para gerenciamento de stash/inventário de trading com integração ao Steam Market
- campos: OnShowTradingStash, OnHideTradingStash, m_inventoryFixer, m_directionNpcAnimator, m_slotScrollRect, TradingStashSlot, CloseButton, m_extractionToggleButton, SteamMarketLinkButton, SteamMyMarketLinkButton, bghd, bghe, bghf, m_tradingStashWarningPanel, m_askAgainToggleButton, m_button_closeWarningPanel
  - Show: exibe o painel de trading acionando OnShowTradingStash
  - Hide: oculta o painel acionando OnHideTradingStash
  - OnSlotSelected: callback quando jogador clica em um slot de item
  - OnConfirmTrade: valida e confirma a transação de trading
  - OnOpenSteamMarket: abre link externo do Steam Market no browser
### TaskbarHero.UploadToSteamData
Estrutura para serializar e enviar dados de items do jogo (ID, quantidade) para sincronização com Steam via Steamworks API
- campos: o_gameItemId, gameItemId, o_steamItemId, steamItemId, o_quantity, qty
  - Serialize: Converte para formato binário/JSON para transmissão
  - ValidateItemIds: Valida correspondência entre game item ID e steam item ID
  - EncryptForSteam: Cifra dados antes do upload
  - GenerateChecksum: Calcula hash de integridade dos dados
  - IsValidForUpload: Verifica se estrutura está pronta antes de enviar
### TaskbarHero.VerifyDlcResult
Armazenar e processar resultado de verificação de propriedade de DLC do jogador contra servidor/storage local.
- campos: o_ownedDlc, isSuccess, ownedDlc
  - VerifyOwnership: Valida se DLC está na lista de possuídos
  - LoadFromServer: Busca status DLC do servidor remoto
  - CompareDlcLists: Compara o_ownedDlc contra ownedDlc
  - CacheResult: Persiste resultado em disco local
  - GetDlcStatus: Retorna bool se isSuccess e DLC é válido

## item  (85)
### TaskbarHero.AddItemResult
Encapsula resultado de tentativa de adicionar item ao inventário/taskbar com tipo de item e status da operação
- campos: ItemType, AddResult
  - get_ItemType: retorna tipo do item adicionado
  - get_AddResult: retorna status da adição (sucesso/falha/motivo)
  - .ctor: inicializa com ItemType e AddResult
  - IsSuccess: valida se operação foi bem-sucedida
  - GetErrorMessage: retorna descrição de falha se houve
### TaskbarHero.Manager.AllInventoryData
Gerencia o estado completo do inventário do jogador (itens, índices e persistência).
- campos: ItemInfo, InInventoryIndex
  - AddItem: insere item no inventário e atualiza InInventoryIndex
  - RemoveItem: remove item por índice e compacta a lista
  - GetItemByIndex: retorna ItemInfo na posição InInventoryIndex
  - SaveInventoryData: serializa ItemInfo para disco/banco
  - GetInventorySize: retorna contagem total de itens
### TaskbarHero.BackendPostItemSaveData
Encapsula dados de um item para sincronização/persistência com backend de servidor
- campos: itemID, itemName, count
  - ToJson: serializa para envio POST ao servidor
  - FromJson: desserializa resposta do backend
  - Validate: verifica se itemID e count são válidos
  - Clone: cria cópia profunda do objeto
  - GetHashCode: para deduplicação em dicionários de inventário
### TaskbarHero.BoxCreateItem
Armazenar e proteger dados de item (ID, quantidade, drop-key) com ofuscação contra manipulação de save
- campos: o_itemId, o_count, o_dropKey, itemId, count, dropKey
  - Initialize: Preenche campos plain com valores padrão
  - CopyToObscured: Criptografa/ofusca plain → campos o_*
  - CopyToPlain: Descriptografa o_* → plain (restore de save)
  - ValidateChecksum: Verifica integridade dos dados obscurecidos
  - Obfuscate: Aplica XOR/cipher nos valores
### TaskbarHero.UI.CraftingMaterialSlotArrow
Gerencia visualização de seta apontando para um slot de material no sistema de artesanato, indicando seleção ou fluxo de ingredientes.
- campos: m_slotIndex, m_object
  - SetSlotIndex: Define qual slot da receita a seta aponta, atualizando m_slotIndex
  - UpdateArrowPosition: Repositiona a seta visualmente para corresponder ao slot ativo
  - OnSlotSelected: Torna a seta visível quando o slot é selecionado
  - OnSlotDeselected: Esconde a seta quando o slot perde o foco
  - Initialize: Configura m_object (GameObject) e vincula aos eventos de seleção de slot
### TaskbarHero.Data.CraftingRecipeInfoData
Armazena dados de uma receita de crafting, definindo o item a ser criado, seus materiais, tier e resultado final
- campos: CraftingRecipeKey, ItemCraftingType, RecipeTier, Material, MaterialIndex, DropKey
  - Validate: Verifica se a receita é válida e acessível
  - GetMaterialCost: Retorna quantidade total de material necessária
  - CanCraft: Determina se há materiais suficientes para craftar
  - ApplyRecipe: Executa o craft e gera o item de resultado
  - GetDropResult: Retorna o ItemKey do item gerado ao completar
### TaskbarHero.UI.CubeBottomTextSetter
Gerencia a exibição de informações textuais (descrição, receita, requisitos de síntese) sobre itens Cube na seção inferior da interface
- campos: m_detailScrollRect, m_recipeDetailParent, m_cubeDefaultDescriptionText, m_dividerObject, m_recipeDetailText, m_warningColor, m_synthesisDescriptionObject, m_synthesisMinTierText, bfwn, bfwo
  - SetCubeDescription: Atualiza o texto de descrição padrão do cubo selecionado
  - ShowRecipeDetails: Exibe e atualiza os detalhes da receita/síntese
  - UpdateSynthesisInfo: Atualiza tier mínimo e requisitos de síntese com validação visual
  - RefreshScrollContent: Reposiciona o ScrollRect quando conteúdo dinâmico muda
  - SetWarningState: Aplica cor de aviso quando requisitos não são atendidos
### TaskbarHero.UI.CubeComboBoxImageCharger
Gerencia carregamento e exibição dinâmica de imagens de receitas em um combo box dropdown para seleção de Cube items.
- campos: m_comboBoxButton, m_recipeMenu
  - LoadRecipeImages: Carrega sprites/texturas das receitas ao inicializar
  - OnComboBoxSelectionChanged: Callback disparado ao selecionar uma receita no dropdown
  - UpdateImagePreview: Atualiza a imagem exibida baseada na receita selecionada
  - RefreshRecipeList: Recarrega lista de receitas disponíveis do menu
  - CacheImages: Pré-cache de imagens em memória para evitar stutter UI
### TaskbarHero.Log.CubeCraftingLog
Sistema de logging que registra histórico de crafting de cubos, rastreando receitas utilizadas, itens criados e sequência de operações
- campos: bfnf, bfng, bfnh
  - LogCraft: Adiciona registro de uma ação de crafting com receita e resultado
  - AddCraftEntry: Insere entrada no histórico com timestamp e metadados
  - GetCraftHistory: Retorna array de crafts realizados
  - SaveCraftLog: Persiste log em arquivo/banco de dados
  - ClearCraftLog: Limpa histórico (reset/destruição)
### TaskbarHero.UI.CubeCraftingSelectMaterialMainButton
Botão principal para selecionar material no sistema de crafting de cubos, gerenciando outlines de hover, slots de material e estados visuais.
- campos: m_outlineObject, m_deActiveSprite, m_subSelectButton, m_materialSlotArrowList, m_mouseHoverOutLine, bgho, bghp, bghq, bghr, bghs, bght, bghu, bghv
  - OnMouseEnter: Mostrar mouseHoverOutLine e ativar outline do botão
  - OnMouseExit: Esconder mouseHoverOutLine e desativar outline
  - OnClick: Selecionar material e trocar sprite conforme estado (ativo/inativo)
  - UpdateMaterialSlots: Renderizar materialSlotArrowList com materiais disponíveis
  - OnSubSelectButtonClicked: Expandir/colapsar subseleção de materiais
### TaskbarHero.CubeCraftingSlot
Representa um slot visual na interface de crafting de cubos, exibindo ícone do item e quantidade disponível.
- campos: image_icon, text_count
  - UpdateDisplay: Atualizar ícone e contador de quantidade no slot
  - SetItem: Definir qual item/cubo está neste slot
  - OnSlotSelected: Callback quando o slot é clicado/selecionado
  - RefreshCount: Sincronizar o text_count com quantidade atual do item
  - ClearSlot: Limpar o slot (remover item e zerar contador)
### TaskbarHero.UI.CubeEffectController
Gerencia efeitos visuais para sistemas de crafting/síntese, alquimia e aprimoramento com feedback visual para sucesso/falha.
- campos: m_levelUpSliderEffect, m_levelUpTextEffect, m_synthesis_Fail, m_synthesis_Success, m_synthesis_GreatSuccess, m_alchemyBGEffect, m_alchemyGoldTexts, m_decorationEffect, m_engravingEffect, m_inscriptionEffect, m_offeringEffect, m_BGEffect_Short, m_BGEffect_Long, m_BGEffect_Concentrate, bfwp, bfwq
  - PlaySynthesisEffect: dispara efeito baseado no resultado (falha/sucesso/grande sucesso)
  - PlayAlchemyEffect: dispara efeito visual de alquimia com textos em ouro
  - PlayLevelUpEffect: reproduz slider + texto animado do level up
  - PlayEnchantmentEffect: dispara efeito para gravação/inscrição
  - PlayBackgroundEffect: seleciona e ativa BGEffect (Short/Long/Concentrate) conforme contexto
### TaskbarHero.CubeInData
Estrutura de dados que representa um item/cubo no inventário, rastreando índice, quantidade e sincronização de mudanças
- campos: CubeDataChange, bfbk, InCubeIndex, bfbl
  - jmz: GetCubeData ou Serialize — retorna dados do cubo para rede/persistência
  - jna: SetCubeData ou Deserialize — carrega dados do cubo a partir de buffer
  - jnb: Validate — valida integridade do cubo (tipo, quantidade, índice)
  - jnc: GetQuantity ou GetStackSize — retorna quantidade/stack do item
  - jnd: SetQuantity ou AddQuantity — modifica quantidade do cubo
### TaskbarHero.CubeItemData
Armazena metadados de um item no sistema de taskbar/inventário do herói, rastreando sua origem, identidade única e posicionamento.
- campos: ItemSourceSlotType, ItemKey, ItemUniqueId, OriginalItemSlotIndex
  - GetSourceSlotType: retorna de qual tipo de slot o item provém
  - SetItemSlot: atualiza a posição/slot do item no inventário
  - IsValid: valida se ItemKey e ItemUniqueId são consistentes
  - Clone: cria uma cópia dos dados do item para movementation/cópia
  - ResetToOriginal: restaura item ao OriginalItemSlotIndex
### TaskbarHero.Data.CubeLevelInfoDataMap
Mapa de configuração que armazena níveis e requisitos de experiência para todos os cubos do jogo
  - GetLevelInfo: retorna CubeLevelInfoData para um cubo específico (lookup por key)
  - LoadFromFile: carrega dados de nível de arquivo de configuração/assets
  - GetCubeLevel: retorna nível atual de um cubo (wrapper getter)
  - GetExpForLevelUp: retorna experiência necessária para próximo nível de um cubo
  - Initialize: popula o mapa durante inicialização do jogo/cena
### TaskbarHero.Data.CubeRecipeInfoData
Armazena metadados de receita de crafting baseada em cubos: chave do cubo, tipo de receita, índice e estado de desbloqueio.
- campos: CubeKey, RecipeType, Index, IsDefaultUnlocked, TooltipStringKey
  - GetRecipeInfo: retorna instância completa da receita para exibição
  - IsUnlocked: verifica se receita está desbloqueada (padrão ou por progresso)
  - GetTooltipText: resolve TooltipStringKey para string localizada
  - CanCraft: valida pré-requisitos do cubo antes de craftar
  - Initialize: carrega dados de persistência (save game)
### TaskbarHero.Data.CubeRecipeInfoDataMap
Mapeia e armazena dados de receitas de cubos para o sistema de crafting
  - GetRecipeInfo: retorna informações de receita por ID
  - AddRecipe: registra nova receita de cubo
  - ContainsRecipe: verifica se receita existe
  - GetAllRecipes: retorna todas as receitas mapeadas
  - Initialize: carrega dados de receitas na inicialização
### TaskbarHero.CubeRecipeTriggerButtonResource
Botão UI na taskbar que aciona uma receita (craft/habilidade) com ícones que mudam conforme o estado do mouse.
- campos: RecipeType, ButtonActive, ButtonMouseHover, TriggerButtonIconDeActive, TriggerButtonIconHover, TriggerButtonIconActive
  - UpdateButtonVisuals: Aplica ícone correto (deActive/Hover/Active) baseado no estado
  - OnMouseEnter: Muda ícone para TriggerButtonIconHover quando mouse passa
  - OnMouseExit: Restaura ícone para TriggerButtonIconActive/DeActive
  - TriggerRecipe: Executa a receita quando botão é clicado
  - SetRecipeType: Atribui qual RecipeType este botão controla
### TaskbarHero.UI.CubeSlotSetter
Gerencia UI de slots de cubos para sistema de crafting/receitas, configurando backgrounds visuais e renderizando slots
- campos: m_recipeBackGroundSprite, m_cubeInventorySlots, m_recipeBackGroundSprites, bfwt
  - SetupRecipeSlots: inicializa e instancia backgrounds em m_recipeBackGroundSprites
  - LoadRecipe: carrega dados de uma receita específica nos slots
  - UpdateSlotVisuals: atualiza aparência baseado em cubos presentes
  - ClearSlots: reseta todos os slots para estado vazio
  - RefreshDisplay: sincroniza UI com estado atual do inventário
### TaskbarHero.Data.CubeSubRecipeInfoData
Armazena dados de uma sub-receita de cube (crafting/decomposição), incluindo requisitos de entrada, custos e recompensas.
- campos: bgnf, bgng, bgnh, bgni, bgnj, bgnk, bgnl, bgnm, bgnn, bgno
  - GetRequiredItems: retorna lista dos 10 itens necessários pra receita
  - GetInputCosts: calcula custo total em moeda/recursos dos ingredientes
  - GetOutputReward: retorna item/recursos obtidos ao completar
  - ValidateRecipe: verifica se receita é válida/desbloqueada
  - GetCraftDuration: retorna tempo de crafting em segundos
### TaskbarHero.Data.CubeSubRecipeInfoDataMap
Mapeamento de dados que armazena e fornece acesso a informações sobre sub-receitas de crafting/combinação de cubos
  - Get/GetCubeSubRecipeInfo: recuperar informações de uma sub-receita por ID
  - GetAll/GetRecipes: obter todas as sub-receitas cadastradas
  - Add/Register: registrar nova sub-receita no mapa
  - Contains/HasRecipe: verificar se sub-receita existe
  - LoadData/Initialize: carregar dados de sub-receitas do storage
### TaskbarHero.UI.DeleteInvalidItemSlot
Componente UI que gerencia e remove slots de itens inválidos do inventário, exibindo ícone e grade do item.
- campos: m_gradeImage, m_iconImage
  - DeleteSlot: Remove o slot de item e limpa referências
  - UpdateItemDisplay: Atualiza visual do ícone e imagem de grade
  - IsItemValid: Valida se o item pode permanecer no slot
  - ClearSlotUI: Reseta as imagens (gradeImage e iconImage) ao deletar
  - OnInvalidItemDetected: Callback acionado quando item inválido é detectado
### TaskbarHero.ECubeDataType
Enum que categoriza tipos de armazenamento/dados do sistema de inventário (itens do jogador, baú, equipamento)
- campos: value__, None, Inventory, Stash, Gear
  - ToString: retorna nome do tipo de cubo (None, Inventory, Stash, Gear)
  - GetHashCode: retorna hash do valor do enum
  - CompareTo: compara ordem dos valores de enum
  - HasValue: valida se o tipo é válido (não None)
  - GetDescription: possivelmente retorna descrição legível em UI
### TaskbarHero.ECubeSynthesisResult
Enum representando os possíveis resultados de síntese de cubos (falha, sucesso, sucesso crítico)
- campos: value__, Fail, Success, GreatSuccess
### TaskbarHero.Data.EGearGroup
Enum que categoriza tipos de equipamento em grupos (armas, armaduras, acessórios, comum, nenhum)
- campos: value__, NONE, WEAPON, ARMOR, ACCESSORY, COMMON
  - ToString: Retorna nome textual do grupo de gear
  - GetHashCode: Calcula hash do valor enum
  - Equals: Compara igualdade entre valores de grupo
  - CompareTo: Ordena grupos para classificação
  - Parse: Converte string para EGearGroup válido
### TaskbarHero.Data.EGearType
Enumeração de tipos de equipamento (armas, armaduras, acessórios) do sistema de inventário
- campos: value__, NONE, SWORD, BOW, STAFF, SCEPTER, CROSSBOW, AXE, SHIELD, ARROW, ORB, TOME, BOLT, HATCHET, HELMET, ARMOR
  - ToString(): retorna nome legível do tipo de gear
  - GetHashCode(): identifica tipo de equipamento para lookup em dicionário
  - Equals(): compara igualdade entre tipos de gear
### TaskbarHero.Data.EGradeType
Enum que define os 11 níveis de raridade/grade para itens, personagens ou recompensas, escalando de COMMON até COSMIC com variantes míticas (IMMORTAL, ARCANA, BEYOND, CELESTIAL, DIVINE)
- campos: value__, COMMON, UNCOMMON, RARE, LEGENDARY, IMMORTAL, ARCANA, BEYOND, CELESTIAL, DIVINE, COSMIC, NONE
  - ToString: retorna o nome legível da grade ("LEGENDARY", "COSMIC", etc)
  - GetRarityMultiplier: retorna multiplicador de valor/poder baseado na grade
  - IsHighTier: verifica se a grade é RARE ou superior
  - GetColorCode: retorna cor UI associada (comum=cinza, lendário=dourado, cósmico=prismático)
  - CompareTo: compara dois tipos de grade por ordem de poder
### TaskbarHero.Data.EItemCraftingType
Enum que define os tipos de equipamentos que podem ser crafados no jogo
- campos: value__, MainWeapon, SubWeapon, Helmet, Armor, Gloves, Boots, Accessory, None
  - GetValues: retorna lista de todos os tipos de crafting disponíveis
  - GetName: retorna string do nome do tipo (MainWeapon, Helmet, etc)
  - Parse: converte string para enum value
  - ToString: representa enum como string legível
### TaskbarHero.Data.EItemParts
Enum que define os slots de equipamento do personagem (armas, armaduras, acessórios)
- campos: value__, NONE, MAIN_WEAPON, SUB_WEAPON, HELMET, ARMOR, GLOVES, BOOTS, AMULET, EARING, RING, BRACER
  - ToString: converte enum para nome legível do slot
  - GetName: retorna string do nome da parte do equipamento
  - IsArmor: verifica se é peça de armadura
  - IsWeapon: verifica se é arma (main ou sub)
  - GetSlotIcon: retorna sprite/ícone do slot UI
### TaskbarHero.Data.EItemSynthesisType
Enum que define os tipos de itens que podem ser sintetizados no sistema de crafting do jogo
- campos: value__, Gear, Accessory, Material, None
### TaskbarHero.Data.EItemType
Enumeração que define os tipos de itens colecionáveis/equipáveis no jogo (caixa de estágio, material bruto, equipamento/gear, nenhum)
- campos: value__, STAGEBOX, MATERIAL, GEAR, NONE
  - ToString: serializa o tipo de item para string (ex: "GEAR")
  - GetHashCode: hash do valor do enum para collections
  - CompareTo: compara tipo de item numericamente
  - Equals: compara igualdade entre tipos de item
  - GetTypeCode: retorna tipo CLR subjacente (Int32)
### TaskbarHero.Data.EREWARDTYPE
Classificar tipos de recompensas: item único, grupo de itens, ou drop de monstro
- campos: value__, ITEM, ITEMGROUP, MONSTER
  - ToString: retorna nome legível do tipo (ITEM, ITEMGROUP, MONSTER)
  - GetValue: retorna valor inteiro subjacente do enum
  - Parse: converte string em tipo de recompensa
  - HasValue: verifica se o tipo de recompensa é válido
  - GetRewardMultiplier: retorna multiplicador de valor por tipo
### TaskbarHero.Data.ERecipeType
Enum que define os tipos de receita/crafting disponíveis no sistema de manufatura do jogo
- campos: value__, ALCHEMY, SYNTHESIS, CRAFTING, DECORATION, ENGRAVING, INSCRIPTION, OFFERING, EXTRACTION, NONE
  - ToString: retorna o nome textual do tipo de receita
  - GetValue: retorna o identificador numérico do tipo
  - GetRecipesByType: busca todas as receitas associadas a este tipo
  - IsExtractionType: verifica se é do tipo EXTRACTION (consome recursos)
  - RequiresCraftingStation: valida se precisa de estação específica
### TaskbarHero.ESlotAction
Enumeração de ações executáveis em slots de inventário (equipar, mover, trancar, descartar, negociar itens).
- campos: value__, NoOp, Equip, Lock, Unlock, MoveToStash, MoveToCube, MoveToInventory, Toast_NoEquipable, Toast_CannotMoveToCube, Toast_FullStash, Toast_FullInventory, Toast_FullCube, CubeOut, TradeWaitingListOut, TradeIn
  - Equip: Equipar item do slot no personagem
  - MoveToStash: Transferir item para baú/armazenamento seguro
  - MoveToCube: Inserir item em container mágico (Horadric Cube?)
  - TradeIn: Registrar item como oferta em negociação
  - Lock: Travar slot para evitar venda/descarte acidental
### TaskbarHero.ESlotType
Enum que categoriza tipos de slots de inventário: inventário principal, stash pessoal, stash de trading, cubo (mochila), e equipamento.
- campos: value__, NONE, INVENTORY, STASH, TRADINGSTASH, CUBEINVENTORY, GEAR
  - ToString(): Retorna nome legível do tipo de slot
  - GetValues(): Retorna array de todos os tipos disponíveis
  - Parse(string): Converte string para valor do enum
  - IsDefined(int): Verifica se valor é válido
  - CompareTo(ESlotType): Compara dois tipos de slot
### TaskbarHero.EStashGradeTypeFilter
Enum que define níveis de raridade de itens (comum até cósmico) para filtros de visualização no sistema de inventário/stash.
- campos: value__, ALL, COMMON, UNCOMMON, RARE, LEGENDARY, IMMORTAL, ARCANA, BEYOND, CELESTIAL, DIVINE, COSMIC, START, END
  - GetRarityTier: retorna posição numérica da raridade para ordenação/comparação
  - IsLegendaryOrAbove: verifica se está na faixa lendária ou superior
  - GetDisplayColor: retorna cor da UI associada à raridade
  - MatchesPlayerLevel: valida se a raridade é apropriada para o nível do jogador
  - ConvertToInventoryFilter: transforma enum em filtro aplicável ao sistema de itens
### TaskbarHero.ETryEquipGearResult
Enum que representa os possíveis resultados ao tentar equipar um item/gear em um herói.
- campos: value__, ClassTypeNotMatch, LowHeroLevel, CannotEquipableItem, InvalidItem, Success
  - ToString: retorna o nome do resultado (para debug/log)
  - HasValue: verifica se o equipamento foi equipado com sucesso
  - GetErrorMessage: traduz o enum para mensagem de erro ao jogador
### TaskbarHero.Data.GearInfoData
Armazena dados persistentes de um equipamento: stats base, 3 modificadores inerentes com tipo/valor, e um modificador único especial.
- campos: GearKey, BaseStat1_Value, BaseStat2_Value, inherentStat1StatType, InherentStat1_MODTYPE, InherentStat1_Value, inherentStat2StatType, InherentStat2_MODTYPE, InherentStat2_Value, inherentStat3StatType, InherentStat3_MODTYPE, InherentStat3_Value, UniqueModKey
  - mke: Serializer (ToProtoBuffer ou similar) para salvar no disco
  - mjz: Deserializer (FromProtoBuffer) ou Clone() para carregar dados
  - CalculateTotalStat: Soma stats base + modificadores inerentes
  - GetGearRarity: Deduz raridade do UniqueModKey
### TaskbarHero.UI.GearSlot
Slot interativo de UI para gerenciar equipamento (gear) — exibe item, permite drag-and-drop, mostra encantamentos, tooltips e estados (travado/inválido).
- campos: m_invalidItemBg, slotnum, m_lockImage, image_ItemIcon, image_GradeBg, itemTooltipTrigger, m_rectTransform, m_enchantSlotViewer, m_inCubeNoticeUI, m_executeMovePreviewImage, m_movePreviewCanMoveSprite, m_movePreviewCanNotMoveSprite, m_slotEffect, tooltipTrigger, m_levelText, bfzw
  - lhb: SetItem — atualiza ícone, grade e nível do equipamento exibido
  - lhc: OnDragBegin — ativa preview de movimento e verifica se pode mover
  - lhd: OnDragEnd — completa movimento do item ou cancela
  - lhe: UpdateLockState — ativa/desativa m_lockImage e bloqueia interação
  - lhf: ValidateSlot — marca como inválido (m_invalidItemBg) se não atende requisitos
### TaskbarHero.Data.GearTypeInfoData
Armazena configuração de tipos de equipamento com seus atributos base e modificadores aplicáveis em combate.
- campos: GEARTYPE, baseStat1StatType, BaseStat1_MODTYPE, baseStat2StatType, BaseStat2_MODTYPE
  - GetGearType: retorna o tipo de equipamento (enum ou ID)
  - GetBaseStat1Type: retorna qual atributo é a primeira stat (HP/ATK/DEF/etc)
  - GetBaseStat1ModType: retorna tipo de modificação da stat 1 (aditivo/multiplicativo/percentual)
  - GetBaseStat2Type: retorna qual atributo é a segunda stat
  - ApplyModifiers: aplica os modificadores de stats ao equipamento ou personagem
### TaskbarHero.Data.GearTypeScaleInfoDataMap
Mapa de configuração que armazena como cada tipo de gear escala em poder/raridade conforme o jogador progride.
  - GetScaleInfo(gearType): retorna dados de escala (multiplicadores, bônus) para esse tipo de gear
  - ContainsGearType(gearType): verifica se o tipo de gear existe no mapa
  - GetAllGearTypes(): lista todos os tipos de gear mapeados
  - SetScaleInfo(gearType, info): registra/atualiza escala para um tipo
  - GetScaleFactor(gearType, level): calcula multiplicador de poder para gear+nível
### TaskbarHero.Data.GradeInfoData
Define dados de classificação de itens/equipamentos com pesos de probabilidade entre grades e slots de aprimoramento (alquimia, gravura, inscrição).
- campos: GRADE, Lower2GradeWeight, Lower1GradeWeight, SameGradeWeight, Higher1GradeWeight, Higher2GradeWeight, InherentSlotAmount, ExtraSlotAmount_Decoration, ExtraSlotAmount_Engraving, ExtraSlotAmount_Inscription, BaseAlchemyGold, BaseCubeExp
  - mjz: Calcular peso de upgrade/downgrade para grade alvo
  - GetSlotCount: Somar slots totais (inerentes + extras de todas categorias)
  - GetAlchemyBaseValue: Retornar valor base para transmutação/fusão
  - GetCubeExpValue: Retornar exp base para sistema de cubo
  - IsValidGrade: Validar se grau solicitado está dentro de range permitido
### TaskbarHero.InventoryAddItem
Adds an item to the player's inventory
  - Execute: Executes the inventory add operation
  - ValidateItem: Validates if the item can be added
  - ApplyInventoryChange: Applies the inventory state change
  - GetItemData: Returns the item data being added
  - OnSuccess: Callback triggered on successful addition
### TaskbarHero.InventoryAddItemResult
Encapsula o resultado de uma tentativa de adicionar um item ao inventário, indicando sucesso e detalhes do erro.
- campos: isSuccess, o_error, added
  - Initialize: Reseta o estado (isSuccess=false, limpa o_error, added=0)
  - IsSuccess getter: Retorna se a adição foi bem-sucedida
  - GetError: Retorna a mensagem de erro se falhou
  - GetAddedCount: Retorna quantos itens foram realmente adicionados
### TaskbarHero.InventoryAddItems
Adiciona itens ao inventário do jogador, gerenciando estado e validação de quantidade/espaço disponível
  - AddItem: insere item no inventário com quantidade
  - CheckInventorySpace: valida se há espaço antes de adicionar
  - ValidateItemData: verifica se item é válido/existente
  - UpdateInventoryUI: sincroniza visão com modelo
  - GetInventoryStatus: retorna estado pós-adição
### TaskbarHero.InventoryAddItemsResult
Resultado de uma operação de adicionar múltiplos itens ao inventário, com rastreamento de sucessos, falhas e mensagens de erro.
- campos: isSuccess, o_error, o_failedItemIds, added
  - IsSuccess: retorna se todos os itens foram adicionados
  - GetError: obtém a mensagem/objeto de erro
  - GetFailedItemIds: lista IDs dos itens que não foram adicionados
  - GetAddedItems: retorna coleção de itens adicionados com sucesso
  - HasErrors: verifica se houve alguma falha parcial
### TaskbarHero.InventoryConsume
Processa o consumo de itens do inventário, obtendo parâmetros do item e retornando o resultado da ação.
  - GetParam: Extrai atributos do item a ser consumido (quantidade, tipo, cooldown)
  - GetResultData: Retorna efeitos/buffs aplicados após consumo (vida, mana, buff temporário)
  - ValidateInventory: Verifica se item existe em quantidade suficiente
  - ApplyEffect: Aplica efeito ao personagem (cura, mana, status)
  - RemoveFromInventory: Remove item consumido do inventário
### TaskbarHero.InventoryExchangeResult
Resultado de operação de troca/crafting de itens no inventário com validação de receita
- campos: isSuccess, o_error, removed, added, o_removed, invalidUniqueId, o_invalidRecipeId
  - GetError: retorna objeto de erro detalhado da falha da troca
  - GetRemoved: retorna lista complexa (Collection/Array) de itens removidos
  - GetInvalidRecipeId: retorna ID da receita que falhou na validação
  - ApplyExchange: aplica efetivamente a remoção e adição de itens
  - ValidateRecipe: verifica se receita existe e seus requisitos são atendidos
### TaskbarHero.InventoryFunctionError
Encapsula e trata erros de operações de inventário (adicionar/remover/validar itens).
- campos: bfhd, bfhe, bfhf, error
  - GetErrorCode: retorna código numérico do erro
  - GetErrorMessage: retorna descrição legível do erro
  - IsValid: valida se o erro é crítico ou recuperável
  - Handle: executa lógica de recuperação ou rollback de inventário
### TaskbarHero.Data.InventoryInfoDataMap
Mapa/dicionário que armazena e gerencia informações estruturadas de todos os itens no inventário do jogador
  - Add/Set: insere ou atualiza informação de um item pelo ID
  - Get: recupera dados de um item específico
  - Remove: deleta item do mapa
  - Contains/Has: verifica existência de item
  - GetAll ou Values: retorna todas as informações do inventário
### TaskbarHero.InventoryInitRequest
Requisição para inicializar ou sincronizar dados de inventário de um jogador, associando multiplas contas Steam (proprietário + jogador atual)
- campos: o_steamId, steamId
  - GetSteamId: Retorna o steamId do jogador atual
  - GetOwnerSteamId: Retorna o o_steamId (proprietário original da conta/inventário)
  - ValidateCredentials: Valida se os Steam IDs são compatíveis/válidos
  - LoadInventoryData: Busca dados de inventário do servidor via Steam IDs
  - IsAuthorized: Verifica se o jogador tem permissão para acessar este inventário
### TaskbarHero.InventoryInitResult
Encapsula o resultado da inicialização do inventário com itens, preços de mercado Steam e slots de DLC.
- campos: isSuccess, items, marketData, steamSlots, dlc
### TaskbarHero.InventoryItemData
Armazena dados de item do inventário com duplicação anti-cheat (versão legível + versão ofuscada protegida).
- campos: o_itemKey, o_itemId, itemKey, itemId
  - CopyFromObscured: restaura itemKey/itemId a partir dos campos o_* protegidos
  - ValidateIntegrity: verifica se dados legíveis e ofuscados estão sincronizados (anti-cheat)
  - GetHashCode/Equals: compara itens por chave+ID com checksum dos campos protegidos
  - Serialize/Deserialize: persiste apenas os campos o_* ofuscados no save
  - Constructor(itemKey, itemId): inicializa ambas as cópias
### TaskbarHero.InventoryProcessBox
Interface ou container genérico para operações de processamento de inventário (crafting, alquimia, reciclagem, transmutação).
  - GetParam: obtém os parâmetros/ingredientes necessários para a operação
  - GetResultData: retorna o item resultante e metadados da transformação
  - Process: executa a operação de processamento
  - Validate: valida se há ingredientes suficientes
  - ExecuteRecipe: aplica transformação de entrada → saída
### TaskbarHero.InventoryProcessBoxResult
Encapsula o resultado da operação de processar/abrir caixas de inventário (loot boxes/cofres), retornando sucesso/erro, items adicionados e caixas processadas.
- campos: isSuccess, o_error, added, boxes
  - GetAddedCount: retorna quantidade de items que foram adicionados ao inventário
  - GetBoxCount: retorna quantas caixas foram processadas nesta operação
  - Validate: valida se o resultado está em estado consistente
  - ApplyToInventory: aplica os items do resultado ao inventário do player
  - GetErrorMessage: retorna string legível do erro se houver
### TaskbarHero.InventoryProcessPending
Processa fila de operações pendentes do inventário (itens, efeitos, movimentações).
  - Execute: executa a operação pendente atualmente enfileirada
  - Process: processa a operação com parâmetros obtidos via GetParam
  - Complete: marca operação como concluída e retorna resultado via GetResultData
  - Queue: adiciona nova operação à fila de espera
  - Validate: valida parâmetros antes de processar
### TaskbarHero.InventoryProcessPendingResult
Encapsula o resultado de uma operação pendente no inventário (sucesso/erro, items processados, quantidade restante)
- campos: isSuccess, o_error, items, remaining, o_remaining
  - GetErrorMessage: retorna descrição textual do erro se isSuccess=false
  - IsValid: valida integridade do resultado (campos consistentes)
  - HasItems: verifica se items foram afetados
  - Clear: reseta o objeto para reuso
  - Apply: persiste as mudanças no inventário do jogador
### TaskbarHero.InventoryReceivePostItem
Handler que processa o recebimento de itens através de um sistema de correio/entrega (Post) e atualiza o inventário
  - GetParam: obtém parâmetros do item a ser recebido (ID, quantidade, rarity)
  - GetResultData: retorna estado final do inventário após recepção
  - AddToInventory: insere o item no slot disponível
  - ValidateCapacity: verifica se há espaço no inventário
  - NotifyPlayerUI: atualiza UI de notificação de item recebido
### TaskbarHero.InventoryReceivePostItemAll
Receber e sincronizar todos os itens do inventário do servidor após uma transação ou evento
  - GetParam: Obtém parâmetros da requisição (transaction ID, player ID)
  - GetResultData: Retorna lista completa de itens com quantidades atualizadas
  - ProcessAllItems: Itera e aplica cada item recebido ao inventário local
  - SyncInventoryState: Sincroniza estado com servidor para validação
  - TriggerInventoryUpdate: Dispara evento de UI para renderizar inventário
### TaskbarHero.EasySaveData.ItemEnchantSaveData
Serializar e persistir dados de encantamentos de itens, armazenando modificações de stats, tier, receita usada e materiais consumidos
- campos: StatModKey, Tier, Value, RecipeType, ModType, MaterialKey, StatType
  - GetModifierValue: retorna o valor numérico da modificação de stat
  - ApplyEnchantment: aplica o encantamento a um item target
  - GetEnchantmentDescription: gera descrição legível do encantamento
  - CalculateCraftCost: calcula custo total baseado em MaterialKey e Tier
  - CanUpgrade: verifica se o encantamento pode ser upado para o próximo Tier
### TaskbarHero.Data.ItemGroupInfoData
Armazena metadados de um grupo de itens (categoria, identificadores) para persistência ou UI de inventory
- campos: ItemGroupKey, GroupName, ItemKey
  - GetHashCode: calcula hash baseado em ItemGroupKey e ItemKey
  - ToString: retorna representação em string (GroupName ou chave)
  - Equals: compara por ItemGroupKey e ItemKey
  - Clone: retorna cópia profunda do objeto
  - Validate: verifica se chaves e nome não estão nulos
### TaskbarHero.Data.ItemLevelScaleInfoDataMap
Mapeia informações de escala (stats multiplicadores/adições) que itens aplicam baseado no nível do jogo.
  - GetScaleInfo: Retorna os multiplicadores/modificadores de stats para um item em um nível específico
  - TryGetScaleInfo: Busca escala com validação, retorna bool se existe
  - SetScaleInfo: Armazena informação de escala de um item para um nível
  - Contains: Verifica se existe escala mapeada para um nível dado
  - Initialize: Carrega o mapa de dados de escala do disco/recurso
### TaskbarHero.UI.ItemSlot
Gerencia visualmente um slot individual de item no inventário com validação, preview de drag-and-drop e efeitos de UI.
- campos: m_invalidItemBg, Image_GradeBg, bgaa, bgab, m_slotRectTransform, m_tradingStashWaitingPreview, m_slotEffect, m_levelText, m_executeMovePreviewImage, m_movePreviewCanMoveSprite, m_movePreviewCanNotMoveSprite
  - lie/lif: ShowMovePreview/HideMovePreview — exibe/oculta sprites de destino válido/inválido
  - lig/lih: ValidateSlot/InvalidateSlot — marca slot como disponível/indisponível
  - lii/lij: UpdateItemDisplay/RefreshGradeBackground — atualiza texto de level e visual de raridade
  - lik: PlaySlotEffect — reproduz animação/efeito visual do slot
  - lhx: OnItemChanged/SetItem — atualiza conteúdo quando item entra/sai do slot
### TaskbarHero.Data.ItemTypeScaleInfoDataMap
Mapeia tipos de itens para suas informações de escala (dano, stats, modificadores de atributos).
  - GetScaleInfo: Retorna dados de escala para um tipo de item
  - SetScaleInfo: Define/armazena escala para tipo de item
  - ContainsItemType: Verifica se tipo de item está mapeado
  - Initialize/Load: Carrega dados de escala do storage
  - GetAllItemTypes: Retorna lista de tipos cadastrados
### TaskbarHero.Manager.LocalInventoryManager
Gerenciador local do inventário do jogador que rastreia items e soul stones, notificando mudanças via evento
- campos: bfuc, bfud, OnSoulStoneCountChanged
  - kra: AddSoulStone - incrementa contagem e dispara OnSoulStoneCountChanged
  - krb: RemoveSoulStone - decrementa contagem e notifica mudança
  - krc: GetSoulStoneCount - retorna contagem atual de soul stones
  - krd: AddItem - adiciona item ao inventário local
  - kre: RemoveItem - remove item do inventário
### TaskbarHero.MaterialData
Armazena um material/item no inventário do herói com sua quantidade e posição de slot
- campos: ItemKey, Count, CubeIndex
  - Add: incrementa contagem do material
  - Remove: decrementa contagem do material
  - IsFull: verifica se cubo atingiu capacidade máxima
  - GetStackValue: retorna valor total da pilha (Count × valor unitário)
  - Clone: duplica a instância de MaterialData
### TaskbarHero.UI.NoticeRemoveInvalidItem
Notificação de UI que gerencia a remoção de itens inválidos do inventário/slots do jogador, com confirmação via botão.
- campos: m_deleteSlots, m_checkButton, bgdz
  - lpn: Inicializa/carrega a notificação e lista slots inválidos em m_deleteSlots
  - lpo: Executa a exclusão dos itens quando m_checkButton é acionado
  - lpp: Atualiza/renderiza a UI de fundo (bgdz) e mostra a lista de slots a serem removidos
### TaskbarHero.Data.RuneInfoData
Armazena dados e atributos de uma runa específica incluindo identificação, efeitos e propriedades
- campos: bgpd, bgpe, bgpf, bgpg, bgph, bgpi, bgpj, bgpk
  - GetRuneName: retorna identificador ou nome da runa
  - GetRuneBonus: calcula ou retorna o bônus de atributo conferido
  - GetRuneDescription: retorna descrição textual dos efeitos
  - GetRuneRarity: retorna nível de raridade ou qualidade
  - IsEquipped: verifica se runa está ativa no herói
### TaskbarHero.EasySaveData.RuneSaveData
Armazenar dados persistentes de uma runa (item de progressão) com seu identificador único e nível atual.
- campos: RuneKey, Level
  - GetRuneKey: retorna o identificador único da runa para lookup
  - GetLevel: retorna o nível de progressão/potência da runa
  - SetLevel: aumenta/modifica o nível da runa
  - OnBeforeSerialize: prepara dados antes de salvar no disco
  - OnAfterDeserialize: restaura estado da runa após carregamento
### TaskbarHero.SlotInteractionManager
Gerencia drag-and-drop e atalhos de teclado para mover itens entre slots de inventário (stash/cube).
- campos: OnDragSlotChanged, OnDirectExecuteMoveSuccess, canvas, image_MoveIcon, currentDragData, m_moveToStashKey, m_moveToCubeKey, isShiftPressed, isAltPressed, bdiz, bdja, bdjb, bdjc, bdjd, m_showDirectMoveRudderTime, bdje
  - iaq: Processar input de mouse (iniciar/atualizar drag)
  - iar: Detectar e processar atalhos (Shift/Alt + tecla)
  - ias: Atualizar UI visual do movimento (ícone, offset)
  - iat: Executar movimento direto (sem drag) via atalho
  - iau: Validar e aplicar mudança de slot, disparar OnDragSlotChanged
### TaskbarHero.UI.StageBox
Sistema UI para gerenciar abertura de caixas de recompensa/loot com animações, efeitos e modo automático
- campos: m_image_BoxIn10, m_image_BoxIn20, m_image_BoxEmpty, m_boxType, m_getItemImage, m_openBoxAnimationDelayTime, m_boxCountImageSlots, m_boxButton, m_openAllEffect, m_boxAnimator, m_boxEffectController, m_clickDetector, m_openSfxKey, bgbr, bgbs, bgbt
  - lla: OnBoxButtonClicked - manipulador de clique do botão de abertura
  - llb: OpenBox - executa lógica principal de abertura da caixa
  - llc: PlayOpenAnimation - inicia animação de abertura com delay configurável
  - lld: SpawnItemRewards - instancia/exibe os itens obtidos da caixa
  - lle: PlayOpenSFX - reproduz áudio de abertura via chave configurada
### TaskbarHero.EasySaveData.StashSaveData
Representar um slot de armazenamento (stash/inventário) com índice, item referenciado e status de desbloqueio para persistência em save data.
- campos: Index, ItemUniqueId, IsUnLock
  - GetItem: retorna o ItemUniqueId armazenado no slot
  - SetItem: atribui um ItemUniqueId ao slot
  - IsSlotUnlocked: verifica se o slot está desbloqueado (IsUnLock)
  - UnlockSlot: desbloqueia o slot
  - ClearSlot: limpa o item e reseta o estado do slot
### TaskbarHero.SteamItemDeleteResult
Armazenar resultado de operação em lote para deletar itens Steam do inventário (total, sucesso, falha).
- campos: totalCount, successCount, failCount
  - GetTotalCount: retorna totalCount
  - GetSuccessCount: retorna successCount
  - GetFailCount: retorna failCount
  - WasSuccessful: verifica se failCount == 0
  - HasFailures: retorna failCount > 0
### TaskbarHero.UI.SubRecipeComboBoxButton
Botão UI na taskbar para seleção/exibição de sub-receitas com indicador de desbloqueio e notificação.
- campos: m_subRecipeSlotButton, m_nothingShowText, m_redDotComponent, m_unlockEffect, bfxa
  - kxi: Initialize — Setup inicial do botão e listeners
  - kxk: OnButtonClicked — Handler de clique para selecionar receita
  - lae: UpdateDisplay — Refresh visual (mostra/esconde redDot conforme estado)
  - laf: PlayUnlockEffect — Anima efeito de desbloqueio ao desbloquear receita
  - lag: RefreshNotification — Atualiza indicador de receita disponível/nova
### TaskbarHero.Data.SynthesisDropInfoData
Armazena configuração de recompensas/drops para o sistema de síntese (crafting) do jogo, incluindo nível, tier de receita, tipo de item e qualidade.
- campos: SynthesisDropKey, ItemLevel, RecipeTier, ItemSynthesisType, GRADE, DropKey
  - GetSynthesisDropKey(): retorna identificador único da recompensa de síntese
  - GetItemLevel(): retorna o nível do item produzido
  - GetRecipeTier(): retorna a dificuldade/tier da receita de síntese
  - GetGrade(): retorna a qualidade/raridade do item
  - IsValidForSynthesis(): valida se os parâmetros permitem síntese
### TaskbarHero.Data.SynthesisDropInfoDataMap
Mapeia informações de síntese (crafting) a dados de drops de itens (recompensas)
  - GetDropInfo: retorna dados de drop associados a um ID de síntese
  - AddSynthesisDrop: adiciona mapeamento síntese → drop ao banco
  - ContainsSynthesis: verifica se síntese existe no mapa
  - RemoveSynthesisDropInfo: remove entrada de mapeamento
  - GetAllSynthesisIds: lista todas as sínteses com drops registrados
### TaskbarHero.UI.SynthesisItemImageChanger
Muda e renderiza a sprite visual do item de síntese na UI baseado no tipo selecionado.
- campos: m_synthesisItemType
  - loe: SetSynthesisItemType(int type) — configura qual tipo de item será exibido
  - lof: UpdateItemImage() — busca e aplica a sprite correta na imagem
  - log: GetSpriteForType(int synthesisType) — retorna a sprite asset baseada no enum do tipo
### TaskbarHero.SynthesisItemResource
Recurso de UI que representa um item sintetizável, armazenando seu tipo de síntese e os ícones sprite para diferentes estados visuais (hover, padrão, desativado).
- campos: ItemSynthesisType, HoverSpriteIcon, DefaultSpriteIcon, DeActiveSpriteIcon
  - GetIcon: retorna o sprite correto baseado no estado (hover/ativo/desativo)
  - GetItemType: retorna o ItemSynthesisType
  - IsActive: verifica se o item está ativo para síntese
  - SetActive: ativa/desativa a síntese do item
  - UpdateVisualState: atualiza qual ícone exibir na UI
### TaskbarHero.UI.SynthesisItemTypeChangerButton
Botão selecionável que permite escolher um tipo de item para síntese, com feedback visual de hover e seleção.
- campos: m_button, m_synthesisItemType, m_image, m_hoverSprite, m_selectSprite, m_defaultSprite, m_text, m_hoverColor, m_defaulColor, m_selectObject
  - OnClick: Seleciona este tipo de síntese e notifica o gerenciador
  - ApplyHoverState: Muda sprite/cor para estado de hover
  - ApplySelectState: Muda sprite/cor para estado selecionado e ativa m_selectObject
### TaskbarHero.Data.SynthesisRecipeInfoData
Armazena dados de receitas de síntese/crafting de items com requisitos de materiais, tier, e fórmula de cálculo de nível do resultado.
- campos: SynthesisRecipeKey, RecipeTier, MinMaterialTier, MinResultLevel, MaxResultLevel, ItemSynthesisType, GRADE, MaterialAmount, MinMaterialAverageLevel, LevelWeight1, LevelWeight2, LevelWeight3, LevelWeight4
  - GetMaterialRequirements: retorna MaterialAmount e MinMaterialAverageLevel
  - CalculateResultLevel: usa LevelWeight1-4 para calcular nível final entre Min/MaxResultLevel
  - ValidateMaterials: valida se materiais atendem MinMaterialTier e nível médio
  - GetRecipeGrade: retorna GRADE baseado em RecipeTier
  - CanSynthesize: verifica se receita pode ser executada com recursos dados
### TaskbarHero.Data.SynthesisRecipeInfoDataMap
Mapeia e gerencia dados de receitas de síntese/crafting do jogo
  - GetRecipe: Busca receita por ID
  - GetAllRecipes: Retorna todas as receitas disponíveis
  - ContainsRecipe: Verifica existência de receita
  - TryGetRecipe: Tenta obter receita com validação
  - LoadRecipes: Carrega dados de receitas do armazenamento
### TaskbarHero.UI.UI_Cube
Interface de UI para sistema de crafting/alquimia: jogador sintetiza itens usando um cubo com receitas, auto-preenchimento de ingredientes, e confirmação com painel de aviso.
- campos: m_mainRecipeToggleBtn, m_cubeBottomTextSetter, m_middleTextSetter, m_synthesisToggleButtonParent, m_cubeSlotSetter, m_trashToggleBtn, m_noticeRecipeTooltipPos, m_noticeAutoFillTooltipPos, text_CubeLevel, Slider_CubeExp, toggleButton_UseStorage, m_craftingMainButton, m_synthesisItemTypeButton, m_synthesisAutoFillButton, toggleButton_AutoFill, toggleButton_Trigger
  - lan: PopulateRecipeList — carrega e exibe receitas disponíveis no toggle
  - lap: OnSynthesisButtonClicked — dispara síntese quando botão é acionado
  - laq: ValidateRecipe — verifica se ingredientes/receita estão válidos
  - lar: ConfirmAlchemy — confirma crafting após aviso
  - las: RefreshCubeDisplay — atualiza nível, XP e slots visuais do cubo
### TaskbarHero.UI.UI_RemakeStash
Interface de gerenciamento de stash (cofre) com abas, slots de itens e transferência bidirecional com inventário
- campos: OnShowStash, OnHideStash, OnStashTabChange, m_stashSlotList, m_stashTabButtonList, m_dlcTabButton, bgcv, bgcw, bgcx, bgcy, SortingButton, toggleButton_StashToInventory, toggleButton_InventoryToStash, button_Close, bgcz, bgda
  - lni: ShowStash - Abre o painel do stash
  - hlt: HideStash - Fecha o painel do stash
  - lnj: OnTabButtonClicked - Muda para aba de stash selecionada
  - lnk: MoveToStash - Transfere item do inventário para stash
  - lnl: MoveToInventory - Transfere item do stash para inventário
### TaskbarHero.Skill.UniqueModData
Armazena dados de uma modificação única de equipamento/item, com tipo e até 5 parâmetros numéricos que definem o efeito.
- campos: EquipModType, Value1, Value2, Value3, Value4, Value5
  - Apply: Aplica a modificação ao equipamento base com os 5 valores
  - GetModValue: Retorna um dos 5 valores parametrizados (Value1-5)
  - ToString: Formata a modificação em texto legível (ex: '+15 ATK')
  - Clone: Cria cópia profunda da modificação
  - Validate: Verifica se os valores estão dentro dos limites válidos
### TaskbarHero.Data.UniqueModInfoData
Armazena dados de um modificador único (mod) com propriedades e metadata para o sistema de itens/equipamentos.
- campos: bgrj, bgrk, bgrl, bgrm, bgrn, bgro, bgrp, bgrq, bgrr, bgrs, bgrt, bgru, bgrv, bgrw, bgrx, bgry
  - bgrj/bgrk/bgrl: ID único, tipo de mod, raridade
  - bgrm-bgrr: Valores de stats (vida, dano, velocidade, etc)
  - bgrs-bgrt: Slotagem e compatibilidade
  - bgru-bgrw: Cooldown, custo, flags de ativo/inativo
  - bgrx-bgse: Serialização/validação de persistência

## outro  (26)
### TaskbarHero.Combat.ActBossCurvedProjectileBaseAttack
Não posso fazer engenharia reversa confiável **sem acesso ao bytecode decompilado** (IL2CPP ou dnSpy). Com apenas nome de classe e nome de campo ofuscado, seria
- campos: bgxe
### TaskbarHero.Combat.ActBossHowling
Preciso do código-fonte da classe para fazer a análise. Você forneceu a estrutura (nome, campos, nomes dos métodos ofuscados), mas não o corpo dos métodos nem o
- campos: bgxm
### TaskbarHero.ActionFuncBase
Sem código fornecido
- campos: bfgy
### TaskbarHero.Combat.ArcherSwiftSurge
Preciso do código-fonte para deduçir. Você pode:

1. **Fornecer o arquivo C#** com a classe `ArcherSwiftSurge`
2. **Ou o output descompilado** (dnSpy/dotpeek)
3
- campos: bgsy
### TaskbarHero.Log.CubeSynthesisLog
Preciso do código decompilado para fazer engenharia reversa legítima. Forneça:

1. **Assinatura dos métodos** ofuscados (parâmetros, tipo de retorno)
2. **Corpo
- campos: bfob, bfoc
### TaskbarHero.DLCManager
Gerencia carregamento, validação e status de conteúdos DLC (pacotes de expansão/personagens/itens adicionais).
- campos: m_dlcDataList, bdax, bday, bdaz, bdba
  - LoadDLCData: Carrega lista de DLCs disponíveis em memória
  - IsDLCInstalled: Verifica se um DLC específico está instalado/ativo
  - ValidateDLCIntegrity: Valida hash/assinatura dos arquivos DLC (bdba = hash?)
  - GetInstalledDLCList: Retorna array de DLCs ativos do m_dlcDataList
  - CheckDLCVersion: Verifica versão/compatibilidade de DLC (bdax = version?)
### TaskbarHero.FunctionBase
Preciso do arquivo C# real para fazer a análise. Você pode me passar o código da classe `TaskbarHero.FunctionBase`, ou me indicar onde está no repositório?

Se 
- campos: bfgw, bfgx
### TaskbarHero.Tutorial.GuidingOpenSkillUI
Preciso do código ou arquivo com a classe `GuidingOpenSkillUI` para fazer engenharia reversa legítima. Você pode:

1. Compartilhar o arquivo C# (ex: `GuidingOpe
- campos: bflu, bflv
### TaskbarHero.Combat.HunterQuickLoader
Preciso do código decompilado ou IL da classe para fazer engenharia reversa. Sem ver o corpo dos métodos (`mlz`, `mma`, `mni`, `mnj`) e como o campo `bgtf` é us
- campos: bgtf
### TaskbarHero.InventoryFunction
Preciso do arquivo decompilado (`.dll`, `.cs` obfuscado, ou IL) para fazer engenharia reversa. Você tem:

- Arquivo DLL do assembly?
- Código C# ofuscado?
- Dum
### TaskbarHero.InventoryFunctionSuccess
Não tenho o código real para analisar. Você precisa fornecer:

1. **Arquivo decompilado** (dnSpy, dotPeek, ilspy) — o MSIL/CIL dos métodos `jxy` e `jxh`
2. **OU
- campos: bfhc
### TaskbarHero.Combat.KnightBaseAtk
Não encontrei os dados de decompilação de `KnightBaseAtk` no repositório. Preciso que você forneça:

1. **O arquivo JSON/txt decompilado** com a classe (via dnS
- campos: bgth
### TaskbarHero.Combat.KnightUnyieldingWill
Não tenho o código para analisar — você mencionou os campos e métodos ofuscados, mas não colou o arquivo ou o IL decompilado. Sem ver:

- O corpo dos métodos (`
### TaskbarHero.Combat.PriestBaseAtk
Preciso do arquivo com o código da classe `TaskbarHero.Combat.PriestBaseAtk` para fazer a engenharia reversa. Você pode fornecer:

1. **O arquivo .cs compilado/
- campos: bgun
### TaskbarHero.Combat.RangeBuffSkill
Preciso do código IL ou bytecode descompilado para fazer engenharia reversa confiável dos métodos ofuscados. 

Você pode fornecer:
1. **Arquivo .cs decompilado*
- campos: bguw, bgux, bguy, bguz
### TaskbarHero.Log.ResurrectionLog
Preciso do arquivo C# real para fazer a engenharia reversa. Você não compartilhou o código da classe `TaskbarHero.Log.ResurrectionLog` — só a estrutura.

Pode p
- campos: bfpj
### TaskbarHero.Data.SoundInfoData
Armazena metadados de áudio (identificador, tipo, volume, caminho do arquivo) para reprodução de sons no jogo
- campos: SoundKey, SoundType, Volume, AudioClipPath
  - GetVolume: retorna o nível de volume configurado para o som
  - LoadAudioClip: carrega o arquivo de áudio do caminho especificado
  - Play: reproduz o som com as configurações armazenadas
  - SetVolume: altera o nível de volume do som
  - GetSoundType: retorna a classificação do tipo de som
### TaskbarHero.Log.StageFailedLog
Preciso do código real ou arquivo decompilado para fazer a análise reversa. Você tem:

1. **O código dos métodos `kha` e `khb`?** (IL decompilado, dnSpy, ILSpy,
- campos: bfpo, bfpp, bfpq, bfpr, bfps
### TaskbarHero.Data.StashInfoDataMap
Preciso do código real da classe para fazer a reversão. Você não incluiu:
- Os campos (mesmo que vazios, há getters/properties?)
- Os métodos ofuscados (dwk_XXX
### TaskbarHero.Log.TempLogData
Preciso ver o arquivo C# decompilado ou o bytecode IL para analisar os métodos ofuscados (kha, khb). Sem o código dos métodos, só tenho:

- **TempLogData** em `
### TaskbarHero.TryStageEnterTryResult
Encapsula o resultado de uma tentativa de transição para um novo estágio, armazenando o tipo de sucesso/falha e dados em cache do próximo estágio
- campos: ResultType, NextStageCache
  - IsSuccess(): bool - Verifica se a transição foi bem-sucedida
  - GetNextStage(): Stage - Retorna o estágio em cache
  - GetResultType(): ResultType - Retorna enum do resultado
  - HasCachedData(): bool - Verifica se NextStageCache foi populado
  - ClearCache(): void - Limpa o cache para próxima tentativa
### TaskbarHero.UI_HeroChild
Preciso do arquivo C# para analisar. Compartilhe o conteúdo da classe `TaskbarHero.UI_HeroChild` (ou o caminho para eu ler).

Sem o código dos 6 métodos ofuscad
- campos: go_m, bdhj
### TaskbarHero.Combat.Projectile.WizardFireball
Preciso do **código actual da classe** para fazer reversão legítima. Você tem:

1. **Arquivo C# com a classe `WizardFireball`**? (caminho exato ou conteúdo)
2. 
- campos: bhes
### TaskbarHero.Combat.WizardIceOrb
Não tenho o código C# para analisar. Preciso que você forneça:

1. O arquivo `.cs` completo (ou trecho) da classe `TaskbarHero.Combat.WizardIceOrb`
2. O tipo e 
- campos: bgwq
### TaskbarHero.Combat.WizardLightning
Preciso do código para fazer a análise. Você pode fornecer:

1. **A classe C# completa** (com o corpo dos 4 métodos ofuscados mml, mlz, mma, mqo)
2. **Ou a saíd
- campos: bgwu
### TaskbarHero.Option.sOptionData
Preciso de mais informações para deducação precisa:

1. **Quais são os métodos ofuscados reais?** (você escreveu "0 (ex: )" mas não listou nenhum)
   - Ex: `dwk
- campos: state, option

## player  (32)
### TaskbarHero.StatusSystem.AccountStatusCategoryData
Armazena categorias de status/progressão da conta do jogador (achievements, estatísticas, conquistas por tipo).
- campos: m_entries
  - kog: GetOrAddEntry — obtém ou cria entrada de categoria existente
  - GetAllEntries — retorna todas as categorias de status
  - HasEntry — verifica se categoria existe
  - ClearCategory — limpa dados de uma categoria
  - GetEntryByID — busca entrada por ID/nome
### TaskbarHero.AttributeCache
Cache de atributos do personagem (HP, MP, ATK, DEF, etc) com notificações de mudança para UI e sistema de skills
- campos: OnAttributeChange, OnAttributeChangeBySkillKey, bfal, sprite_ActiveIcon, sprite_DeActiveIcon, bfam, bfan, bfao, bfap, bfaq, bfar, bfas
  - GetAttribute: retorna valor do atributo do cache
  - SetAttribute/UpdateAttribute: atualiza valor e dispara OnAttributeChange
  - AddAttribute: incrementa atributo (usado por buffs/debuffs)
  - OnAttributeChanged: callback interno que dispara ambos os eventos
  - RefreshAllAttributes: reconstrói cache a partir de valores base + modificadores
### TaskbarHero.Data.AttributeGroupInfoData
Armazena configuração de um grupo de atributos e quantos pontos de alocação são necessários para o jogador nesse grupo
- campos: AttributeGroupKey, RequiredAllocatedPoint
  - GetAttributeGroupKey: Retorna a chave/ID do grupo de atributos
  - GetRequiredAllocatedPoint: Retorna quantidade de pontos necessária para alocar
  - SetRequiredAllocatedPoint: Define o requisito de pontos do grupo
  - IsRequirementMet: Verifica se pontos alocados atendem o requisito
  - Validate: Valida coerência entre chave e requisito
### TaskbarHero.Data.AttributeGroupInfoDataMap
Mapeamento de metadados sobre grupos de atributos (força, inteligência, etc)
  - GetGroupInfo: retorna data do grupo
  - SetGroupInfo: atualiza metadados
  - ContainsGroup: verifica existência
  - GetAllGroups: lista todos os grupos
### TaskbarHero.Data.AttributeInfoData
Armazenar dados de um atributo de herói (tipo, valor, limites de progressão).
- campos: AttributeKey, HeroKey, GroupKey, AttributeType, Value, RequiredPoint, MaxLevel
  - Getter/Setter padrão de Value e MaxLevel
  - CanLevelUp: valida se pontos suficientes para evoluir
  - ApplyPoints: consome RequiredPoint para incrementar Value ou Level
  - ResetToDefault: reseta Value para estado inicial
  - Serialize/Deserialize: persistência JSON
### TaskbarHero.Data.AttributeLoadData
Container para dados de serialização/carregamento de um atributo do jogo, mapeando a chave única do atributo para seu ícone visual.
- campos: AttributeKey, AttributeIcon
  - GetAttributeKey: Retorna a chave identificadora do atributo
  - GetAttributeIcon: Retorna a sprite/texture do ícone do atributo
  - Load: Carrega dados de atributo de arquivo/banco de dados
  - Deserialize: Desserializa JSON/binary para instância
  - ToString: Retorna representação textual debug (ex: 'Atributo: STR, Icon: sword.png')
### TaskbarHero.CameraFollower
Sistema de câmera que segue suavemente um alvo (jogador) com interpolação, offset configurável e controle selectivo por eixo.
- campos: DebuggingTest, _targetCamera, _offset, _followX, _followY, _followZ, _useSmoothFollow, _smoothSpeed, bdex, bdey
  - hkg: GetTargetPosition — obtém posição do alvo sendo seguido
  - hkh: FollowTarget — aplica lógica de follow baseada em _followX/Y/Z
  - hki: ApplySmoothFollowing — interpola câmera com _smoothSpeed entre posições
  - hkj: UpdateCameraPosition — atualiza posição final da câmera aplicando offset
  - hkk: ValidateCameraReference — valida se _targetCamera está atribuída
### TaskbarHero.CameraManager
Gerencia câmera 2D ortogonal com scroll controlado, restrições de limite de tela e correção de offset para seguimento suave.
- campos: mainCamera, bfjj, _playScreenMask, _defaultOrthographicSize, screenWidthInUnits, _scrollSpeed, m_cameraOffsetCorrectionX, m_debugCorrectionXInput
  - kdc: LateUpdate ou UpdateCameraPosition - atualiza posição da câmera por frame
  - kdd: ApplyScroll ou MoveCamera - aplica movimento suave com _scrollSpeed
  - kde: ApplyPlayScreenConstraints - limita câmera dentro dos limites de _playScreenMask
  - kdf: CalculateTargetPosition - calcula posição alvo baseada em offset
  - kdg: AdjustZoom ou SetOrthographicSize - ajusta orthographicSize dinamicamente
### TaskbarHero.StatusSystem.EAccountStatusCategory
Enum que categoriza tipos de status de conta do jogador (combate, exploração, recompensas)
- campos: value__, Battle, Exploration, Reward
  - ToString: Retorna nome da categoria como string
  - GetHashCode: Hash do valor enum para uso em dicionários
  - Equals: Compara igualdade entre categorias
  - CompareTo: Ordena categorias por valor
### TaskbarHero.EAggregateType
Enum que classifica tipos de eventos/estatísticas do jogador para rastreamento e agregação de dados no jogo
- campos: value__, MonsterKill, HeroDeath, GoldEarn, BoxObtain, ItemObtain, Synthesis, Alchemy, Crafting, Offering, Extraction, Decoration, Engraving, Inscription, StageClear, StageFail
  - GetValues(): retorna todos os tipos de eventos como array
  - Parse(string): converte string em valor do enum
  - ToString(): retorna nome do tipo de evento
  - GetName(value): retorna nome legível do tipo de evento
### TaskbarHero.Data.EEquipClassType
Enumeração que define os seis tipos de classe jogável (Knight, Ranger, Sorcerer, Priest, Hunter, Slayer) no sistema de personagem do jogo.
- campos: value__, All, Knight, Ranger, Sorcerer, Priest, Hunter, Slayer
  - ToString: retorna o nome da classe como string legível
  - GetHashCode: identificador único para a classe
  - CompareTo: compara enums para ordenação
  - HasFlag: verifica se classe está incluída em um conjunto (uso do campo 'All')
  - Parse/TryParse: converte string em enum EEquipClassType
### TaskbarHero.Data.EHeroType
Enumeração que define as 7 classes de heróis jogáveis disponíveis no jogo
- campos: value__, Knight, Archer, Wizard, Priest, Hunter, Barbarian
  - ToString: retorna o nome legível da classe do herói
  - GetHashCode: gera hash do valor do enum
  - Equals: compara igualdade entre dois tipos de heróis
  - CompareTo: ordena tipos por índice numérico
  - GetValues: retorna array com todos os 7 tipos de herói
### TaskbarHero.Log.ELogFilter
Enum de filtros para categorizar eventos do jogo no sistema de logging (combate, progresso, crafting, itens)
- campos: value__, None, StageClear, GetItemWithBoxOpen, GetBox, HeroDie, HeroResurrection, HeroLevelUp, StageFailed, SynthesisResult, AlchemyResult, DecorationResult, EngravingResult, InscriptionResult, OfferingResult, CraftingResult
### TaskbarHero.Log.ELogType
Enum que categoriza tipos de eventos/logs do jogo (combat, crafting, progressão e looting do herói).
- campos: value__, None, StageClear, GetItemWithBoxOpen, GetBox, HeroDie, HeroResurrection, HeroLevelUp, StageFailed, SynthesisResult, AlchemyResult, DecorationResult, EngravingResult, InscriptionResult, OfferingResult, CraftingResult
  - ToString: converte enum para nome em string
  - GetHashCode: valor numérico do tipo de evento
  - CompareTo: compara tipos de evento para ordenação
### TaskbarHero.EMoveType
Enum que define os três tipos de movimento possíveis para o herói: arrastável (DRAG), instantâneo (DIRECT) e bloqueado (LOCK).
- campos: value__, DRAG, DIRECT, LOCK
  - GetMovementType: retorna o tipo de movimento atual do herói
  - ApplyMovement: aplica o movimento baseado no tipo (drag, direct, lock)
  - IsMovementAllowed: verifica se o movimento é permitido naquele tipo
  - OnMovementChange: callback quando o tipo de movimento muda
  - ValidateMovementType: valida se o tipo é válido para o contexto
### TaskbarHero.Data.EPetUnlockConditionType
Define as condições para desbloquear diferentes pets do jogo (nenhuma, derrotar monstro, ou via DLC).
- campos: value__, NONE, KillMonster, DLC
  - GetConditionDescription: retorna texto legível da condição
  - IsConditionMet: verifica se o jogador atingiu a condição de unlock
  - GetUnlockRequirement: retorna o valor/requisito associado
  - IsPetUnlocked: valida se pet pode ser desbloqueado agora
  - FromConditionType: converte enum para dados de progressão
### TaskbarHero.Hero
Gerencia o herói jogável com sistema de revivimento, feedbacks visuais de ação e interface HUD.
- campos: cache, m_effectRenderer, m_effectResetSprite, _feedbackLanding, _feedbackEscaping, HeroHud, bcwb, bcwc, bcwd, bcwe, bcwf, bcwg, LevelUpEffectObject, bcwh, bcwi, bcwj
  - goi: Controlar movimento/saltar
  - goj: Registrar pouso (Landing feedback)
  - gok: Aplicar dano/receber ataque
  - gol: Atualizar barra de vida/stats
  - gon: Processar revivimento (usar ReviveTime/DelayTime)
### TaskbarHero.Data.HeroInfoData
Armazena dados de configuração de um herói (personagem) incluindo stats de combate, equipamento, recursos visuais e estado de disponibilidade.
- campos: HeroKey, HeroNameKey, DescriptionKey, ClassType, MainWeaponGearType, SubWeaponGearType, SkillKey, AttackDamage, AttackSpeed, CastSpeed, CriticalChance, CriticalDamage, CooldownReduction, MaxHp, Armor, MovementSpeed
  - mjz: Inicializa ou revalida dados do herói (p. ex., aplicando modificadores de classe ou equipamento)
### TaskbarHero.Data.HeroLoadData
Container de dados para carregar assets visuais e prefab de um herói (ícone, retrato, animador, estado de bloqueio/morte)
- campos: HeroKey, HeroIcon, HeroLockIcon, HeroDeadIcon, HeroPrefab, RuntimeAnimatorController, HeroPortrait
  - GetCurrentIcon: Retorna HeroIcon, HeroLockIcon ou HeroDeadIcon conforme estado do herói
  - IsUnlocked: Valida se herói está desbloqueado (HeroLockIcon null ou false)
  - InstantiateHero: Cria instância do HeroPrefab com RuntimeAnimatorController atribuído
  - GetHeroDisplayData: Retorna tupla (HeroIcon, HeroPortrait, HeroKey) para UI
  - ValidateAssets: Verifica se todos os campos de sprite/prefab estão preenchidos
### TaskbarHero.Data.LevelInfoData
Armazena dados de progressão por nível: nível atual e experiência acumulada/necessária para upgrade
- campos: Level, ExpForLevelUp
  - GetLevel: retorna o nível atual
  - GetExpForLevelUp: retorna XP necessário para próximo nível
  - IsLevelUp: valida se XP alcançou threshold
  - CalculateNextLevel: calcula nível baseado em XP
  - ResetProgress: reseta nível e XP para inicial
### TaskbarHero.MailSaveData
Persistência de emails/correio recebidos com itens anexados, rastreando leitura e reivindicação
- campos: index, receivedDate, itemInstanceId, isRead, isClaimedItem, ItemKey, wasMarketItem
  - MarkAsRead(): marca email como lido
  - ClaimItem(): reclama/resgata o item anexado
  - CanClaimItem(): verifica se item pode ser reivindicado
  - GetItemData(): retorna dados da instância do item
  - IsValid(): verifica se email ainda é válido/não expirado
### TaskbarHero.UI.MouseRightClickDirectMoveRudderParts
Botão de controle de UI que permite movimento direto do herói via clique direito do mouse
- campos: m_text, m_image, bfyc, bfyd, bfye
  - lda: Processa clique direito para iniciar movimento na direção
  - ldb: Define/aplica vetor de movimento baseado em posição relativa
  - ldc: Atualiza estado visual (destaque/feedback) em OnPointerEnter/Exit
### TaskbarHero.UI.MouseRightClickDirectMoveRudderParts_Old
Gerencia movimento direto do personagem via clique direito do mouse, com feedback visual de hover nos componentes do leme de direção
- campos: m_rudderDirection, bfyf, bfyg
  - ldd: Processa clique/entrada do mouse direito para iniciar movimento direto
  - lde: Finaliza/reseta movimento direto e restaura estado normal dos componentes
### TaskbarHero.MoveRequest
Enfileira uma requisição de movimento do herói através da interface de controle
- campos: 
  - Execute: Aplica o movimento ao personagem
  - Validate: Verifica limites de movimento
  - GetDirection: Extrai vetor de deslocamento
  - Serialize: Codifica para transmissão
  - Cancel: Cancela movimento em fila
### TaskbarHero.Pet.Pet
Sistema de pet companion que segue o jogador, spawna/despawna periodicamente e anima no mundo de jogo.
- campos: m_modelRenderer, m_petAnimator, m_moveSpeedRatio, m_followOffset, bfku, bfkv, bfkw, bfkx, bfky, m_spawnDelayTime, m_deSpawnDelayTime, bfkz, bfla, bflb, bflc, bfld
  - kex: AtualizarPosição (move o pet pra seguir o jogador usando followOffset e moveSpeedRatio)
  - kez: AtualizarAnimação (atualiza m_petAnimator baseado em estado de movimento)
  - kfa: AtualizarComportamento (lógica de ia do pet: parar/perseguir/reagir)
  - kfb: VerificadorDeDano (processa colisão/ataque usando campos ofuscados de hp/atk)
### TaskbarHero.Data.PetInfoData
Armazena dados configuráveis de um pet individual com identificação, localização, stats e condição de desbloqueio.
- campos: PetKey, NameKey, DescriptionKey, StatDataKey, UnlockCondition, Param1, Param2
  - GetLocalizedName: Retorna nome do pet via NameKey
  - GetUnlockInfo: Valida UnlockCondition com Param1 e Param2
  - GetStatData: Acessa stats do pet via StatDataKey
  - IsUnlocked: Verifica se o pet foi desbloqueado
  - GetDescription: Retorna descrição localizada do pet
### TaskbarHero.Data.PetLoadData
Armazena metadados e prefabs de um Pet desbloqueável para carregamento dinâmico e renderização na UI
- campos: PetKey, PetPrefab, ArrangedIcon, LockedIcon, UnlockedIcon, PetMainImage
  - GetPetPrefab(): GameObject - instancia o pet em cena
  - GetIconForState(bool locked): Sprite - retorna ícone locked/unlocked
  - GetMainImage(): Texture2D - retorna imagem principal para exibição
  - IsUnlocked(): bool - verifica status de desbloqueio
  - Initialize(PetKey key): void - inicializa dados carregados
### TaskbarHero.Manager.PetManager
Gerenciar desbloqueio, equipagem e dados de pets/criaturas do jogador em runtime.
- campos: OnArrangedPetKeyChanged, OnPetUnlocked, bful, m_arrangedPetKeyRuntimeData
  - Initialize: Carregar dados iniciais de pets e estado do manager
  - UnlockPet: Desbloquear novo pet e disparar OnPetUnlocked
  - SetArrangedPet: Equipar pet específico e disparar OnArrangedPetKeyChanged
  - GetArrangedPet: Retornar pet atualmente equipado (from m_arrangedPetKeyRuntimeData)
  - GetPetData: Recuperar stats/dados de um pet por chave
### TaskbarHero.PetModelPreview
Gerencia visualização e dados de modelos de pets no taskbar/UI do jogo
- campos: m_petEmptySlotImage, m_petModels, m_petKeys, m_petName, m_petImage, m_petArranged, bdga, bdgb
  - hns: InitializePetPreview — configura componente com dados iniciais
  - hnt: UpdatePetDisplay — atualiza imagem/nome do pet mostrado
  - hnu: BindPetData — carrega modelo e info do pet pelo índice
  - hnv: ClearPetPreview — reseta para slot vazio
### TaskbarHero.EasySaveData.PetSaveData
Armazena o estado persistente de um pet do jogador (desbloqueado e se foi visualizado).
- campos: PetKey, IsUnlock, IsViewed
  - GetPetKey: retorna a chave identificadora do pet
  - Unlock: marca o pet como desbloqueado
  - MarkAsViewed: marca o pet como visualizado pelo jogador
  - IsUnlocked: verifica se o pet foi desbloqueado
  - PetSaveData(string key): construtor que inicializa um novo pet com sua chave
### TaskbarHero.PlayerSaveData
Persistência de estado completo do jogador (inventário, heróis, economia, missões, progressão)
- campos: commonSaveData, settingSaveData, BoxData, BoxBucketUseBoxList, BoxBucketGetBoxList, AlchemyPendingIdList, AlchemyReceiptList, pendingEnchantList, backendPostList, currenySaveDatas, heroSaveDatas, mailSaveDatas, attributeSaveDatas, PetSaveData, RuneSaveData, inventorySaveDatas
  - PostLoad: deserialização e normalização de dados carregados do disco/backend
  - PreSave: validação e serialização de estado antes de persistir
  - ValidateIntegrity: verificação de corrupção de dados entre save/load
  - MigrateVersion: conversão de formato de save antigo para novo schema
  - ClearTemporaryData: limpeza de pendências (enchants, alchemy, posts)
### TaskbarHero.Data.StatModInfoData
Armazena dados de modificadores de atributos (stats) aplicados ao jogador ou inimigos.
- campos: bgqe, bgqf, bgqg, bgqh, bgqi, bgqj, bgqk
  - mjz: ApplyModifier() — aplica o modificador aos stats base
  - mko: RemoveModifier() — remove o modificador quando expira ou é cancelado
  - GetModifierValue() — retorna o valor total do modificador
  - IsActive() — verifica se o modificador está ativo
  - UpdateDuration() — decrementava duração a cada frame/turno

## rede  (20)
### TaskbarHero.StatusSystem.AccountStatus
Gerencia persistência e sincronização de dados de conta do jogador com servidor
- campos: bftg, bfth
  - GetAccountID: obtém identificador único da conta
  - SyncToServer: envia estado atual da conta para servidor
  - LoadFromServer: carrega dados de conta do servidor remoto
  - IsAccountValid: verifica se conta está autenticada/válida
  - UpdateAccountData: atualiza campos de status de conta (bftg, bfth)
### TaskbarHero.BackendPostData
Gerencia dados de postagens/eventos do servidor com expiração temporal, itens associados e sincronização com cache local
- campos: title, content, author, inDate, items, expirationTime, reservationTime
  - IsExpired: verifica se expirationTime já passou
  - CleanExpiredItems: remove itens com expiração vencida da lista
  - SyncWithServer: envia reservationTime ao servidor e atualiza estado
  - Serialize: converte estrutura para JSON/bytes para transmissão
  - CalculateRemainingTime: retorna segundos até expiração para UI
### TaskbarHero.DownloadToSteamData
Estrutura para sincronizar dados de itens Steam (ID, quantidade, origem) entre forma legível e ofuscada, provavelmente durante download/cache de inventário
- campos: o_from, o_steamItemId, o_quantity, from, steamItemId, quantity
  - CopyFromObscured: copia campos ofuscados (o_*) de volta para forma legível
  - Download: busca dados de inventário da API Steam
  - Serialize: codifica dados para armazenamento ou transmissão
  - ValidateSteamItem: verifica se steamItemId é válido na conta
  - GetQuantityFromSteam: consulta quantidade atual do item em Steam
### TaskbarHero.ERemoveSource
Enum que define a razão/fonte pela qual um item foi removido da barra de tarefas do herói (falha de sincronização server ou ação explícita do usuário).
- campos: value__, ServerNotSync, UserRequest
  - ToString: retorna nome legível do valor do enum
  - CompareTo: compara este RemoveSource com outro para ordenação
  - Equals: verifica igualdade entre duas razões de remoção
  - GetHashCode: calcula hash do RemoveSource para uso em coleções
  - HasFlag: verifica se um flag específico está presente (se implementado como flags)
### TaskbarHero.ESteamSessionStatType
Enum que tipifica eventos/estatísticas de sessão para rastreamento em Steam achievements e progresso do jogador
- campos: value__, BossKill, HeroDeath, HeroResurrection, HeroLevelUp, StageClear, StageFail, FirstDefeatStageKey, ItemObtain, ArcanaItemFound, BeyondItemFound, CelestialItemFound, DivineItemFound, CosmicItemFound, Synthesis, Crafting
  - GetDisplayName: retorna rótulo legível do tipo de stat
  - GetCategory: agrupa tipos em combate|crafting|items|looting
  - ReportToSteam: envia evento ao backend Steam
  - IsUnlockable: verifica se evento desbloqueável como achievement
### TaskbarHero.EchoFunction
Requisição de eco para validar conectividade com servidor e medir latência de rede
  - jxd: SendEchoRequest - envia mensagem de eco para o servidor com timestamp
  - jxf: ProcessEchoResponse - recebe resposta do servidor e calcula latência
### TaskbarHero.InventoryAddItemRequest
Requisição de rede para adicionar um item ao inventário do jogador
- campos: o_itemId, itemId
  - Initialize: Inicializa a requisição com o ID do item
  - ToJson: Serializa para JSON e envia ao servidor
  - GetItemId: Retorna o identificador do item
  - SetItemId: Define qual item será adicionado
  - Validate: Verifica se o ID do item é válido antes de processar
### TaskbarHero.InventoryAddItemsRequest
Serializa requisição de rede para adicionar múltiplos itens ao inventário do servidor com anti-cheat
- campos: o_itemIdList, itemIdList
  - Initialize: Aloca e inicializa as listas de IDs de itens
  - ToJson: Converte o request para string JSON para transmissão HTTP/WS
  - CopyToObscured: Copia dados sensíveis para versão ofuscada/encriptada (anti-cheat)
  - ItemIdListToJsonArray: Serializa array de IDs em formato JSON válido
### TaskbarHero.InventoryConsumeRequest
Requisição de consumo de item com validação anti-cheat através de campo ofuscado (o_itemKey)
- campos: o_itemKey, itemKey
  - CopyFromObscured: compara versão normal e ofuscada para detectar manipulação do cliente
  - Validate: verifica se item existe e pode ser consumido no estado atual
  - Deserialize: reconstrói requisição a partir de JSON recebido
  - GetItemKey: retorna ID do item com segurança
  - GenerateObfuscated: cria checksum ofuscado para transmissão
### TaskbarHero.InventoryDownloadToSteamRequest
Encapsula uma requisição de download de item de inventário do Steam, serializando IDs de conta e item para transmissão.
- campos: o_steamId, o_steamItemId, steamId, steamItemId
  - get_steamId: Propriedade de acesso ao ID da conta Steam
  - set_steamId: Propriedade que define ID da conta Steam
  - get_steamItemId: Propriedade de acesso ao ID do item Steam
  - set_steamItemId: Propriedade que define ID do item Steam
  - ValidateRequest: Verifica se IDs estão presentes antes de serializar
### TaskbarHero.InventoryDownloadToSteamResult
Encapsula o resultado de uma sincronização de inventário baixado do Steam, contendo status de sucesso, erros, e listas de itens adicionados vs. baixados.
- campos: isSuccess, o_error, added, downloaded
  - IsSuccessful: verifica se isSuccess é true
  - GetError: retorna o_error como mensagem legível
  - GetAddedItemsCount: retorna quantidade de items em 'added'
  - GetDownloadedItemsCount: retorna quantidade de items em 'downloaded'
  - ApplyToInventory: aplica mudanças (added/downloaded) ao inventário do player
### TaskbarHero.InventoryProcessBoxRequest
Requisição serializada para sincronizar mudanças de inventário (obter/usar/criar itens) com validação servidor-side via campos ofuscados.
- campos: o_getitemkey, o_useItemKeyList, createItemList, getitemkey, useItemKeyList, isReset
  - o_getitemkey: Cópia ofuscada do ID do item a obter (anti-cheat)
  - o_useItemKeyList: Versão ofuscada da lista de itens a consumir
  - createItemList: Aplica itens novos/adquiridos ao inventário local
  - isReset: Flag indicando reset de estado da caixa/recompensa
  - GetItemKeyToJsonArray: Serializa chave de item única para array JSON (transmissão rede)
### TaskbarHero.InventoryReceivePostItemAllRequest
Requisição de rede para receber/sincronizar todos os itens do inventário do servidor
- campos: o_postType, postType
  - CreateRequest: monta corpo HTTP com postType
  - Send: dispara POST para endpoint de inventário
  - GetPostType: retorna o tipo de operação (o_postType → postType)
  - Validate: verifica se campos obrigatórios existem
  - SetPostType: configura tipo de requisição (save/load/sync)
### TaskbarHero.InventoryReceivePostItemAllResult
Resposta serializada de uma chamada POST ao servidor para receber/adicionar múltiplos itens ao inventário.
- campos: isSuccess, o_error, added
  - FromJson: Desserializa resposta HTTP em instância da classe
  - Apply: Aplica os itens (campo 'added') ao inventário local do jogador
  - Validate: Verifica se isSuccess=true e 'added' é válido antes de processar
  - GetErrorMessage: Retorna mensagem legível do campo o_error
  - ClearAddedItems: Limpa a lista de itens após confirmação de recebimento
### TaskbarHero.InventoryReceivePostItemRequest
Requisição ao servidor para receber um item de correspondência/post no inventário do jogador
- campos: o_postType, o_postIndate, postType, postIndate
  - GetPostType: retorna o tipo de post
  - GetPostIndate: retorna data de recebimento
  - SetPostType: define tipo antes do envio
  - SetPostIndate: define data de recebimento
  - Validate: valida campos antes de serializar
### TaskbarHero.InventoryUploadToSteam
Sincroniza e envia dados do inventário do jogador para a plataforma Steam
  - UploadInventory: executa o upload real dos dados de inventário para Steam
  - SerializeInventory: serializa items do inventário em formato que Steam aceita
  - ValidateInventoryData: valida integridade dos dados antes de enviar
  - OnUploadComplete: callback executado após sucesso/falha do upload
  - GetInventoryState: obtém snapshot atual do inventário para sincronizar
### TaskbarHero.InventoryUploadToSteamRequest
Requisição de upload de item de inventário para Steam com validação e serialização
- campos: o_steamId, o_itemKey, o_slot, steamId, itemKey, slot
  - Initialize: Inicializa campos públicos e ofuscados com steam ID, item key e slot
  - CopyToObscured: Copia dados dos campos públicos para os campos protegidos (o_*) antes do envio
  - ToJson: Serializa a requisição para JSON para transmissão ao servidor Steam
### TaskbarHero.ServerDlcData
Gerencia e sincroniza quais DLCs (conteúdo baixável) são possuídos pelo jogador no servidor.
- campos: o_ownedDlcId, dlcId
  - SetOwnedDlc: Marca um DLC como possuído pelo jogador
  - HasDlc: Verifica se um DLC específico está na lista de possuído
  - GetOwnedDlcs: Retorna array/list de IDs de DLCs possuídos
  - RemoveOwnedDlc: Remove um DLC da lista de possuído
  - ClearOwnedDlcs: Limpa toda a lista de DLCs possuídos
### TaskbarHero.ServerState
Define os possíveis estados de conexão com o servidor de jogo
- campos: value__, None, Offline, Maintenance
  - ToString: retorna nome legível do estado (None, Offline, Maintenance)
  - GetHashCode: calcula hash do valor de estado
  - CompareTo: compara dois estados para ordenação
  - Parse: converte string para o enum correspondente
  - GetValues: retorna array com todos os valores de estado disponíveis
### TaskbarHero.StageBoxRequestAbuseGuard
Sistema anti-abuse que detecta e bloqueia requisições excessivas de stage box dentro de um intervalo de tempo
- campos: m_detectionIntervalSeconds, m_requestThreshold, m_blockDurationMinutes, bddi, bddj, bddk, bddl
  - RegisterRequest: registra timestamp e incrementa contador de requisições no intervalo atual
  - IsBlocked: retorna se o usuário está dentro de um período de bloqueio ativo
  - CheckThresholdExceeded: verifica se contador de requisições excedeu m_requestThreshold
  - ApplyBlockage: inicia período de bloqueio por m_blockDurationMinutes
  - ResetCounterIfIntervalExpired: reseta contador se m_detectionIntervalSeconds passou

## save  (28)
### TaskbarHero.EasySaveData.AggregateSaveData
Contêiner genérico para persistir dados heterogêneos (estado, config, progressão) via EasySave com chaveamento Type+SubKey
- campos: Type, SubKey, Value
  - Save: escreve Value no arquivo EasySave identificado por Type
  - Load: recupera dados salvos usando Type+SubKey como chave
  - GetValue: retorna Value tipado pela chave Type
  - SetValue: atualiza Value com validação de Type
  - Delete: remove entrada identificada por Type+SubKey
### TaskbarHero.EasySaveData.AttributeSaveData
Estrutura de dados para persistir um atributo individual do jogador (ex: força, defesa, vida) com sua chave/nome e valor numérico.
- campos: Key, Level
  - GetKey: retorna o identificador/nome do atributo
  - GetLevel: retorna o valor numérico atual do atributo
  - SetLevel: atualiza o valor do atributo
  - ToString: serializa para string (ex: 'strength:25')
  - Clone: cria uma cópia para salvamento
### TaskbarHero.Log.BoxOpenLog
Registra (loga) eventos de abertura de caixas/cofres no jogo, rastreando posição (x, y, z) e timestamp.
- campos: bfmx, bfmy, bfmz
  - kha: LogBoxOpened — escreve evento de abertura na memória/disco
  - khb: SerializePosition — formata coords (bfmx, bfmy, bfmz) pro arquivo de log
### TaskbarHero.CommonSaveData
Estrutura de dados persistente que armazena o progresso completo do jogador (heróis/pets desbloqueados, estágios/waves, tutoriais, playtime).
- campos: version, lastSavedTime, playTime, isFirstPlay, LastRollingBackUpTime, SendSteamId, TutorialCleared, ArrangedPetKey, arrangedHeroKey, firstUnlockHeroKey, maxCompletedStage, currentStageKey, currentStageWave, useStorage, isOpeningDirectionPlayed
  - LoadSaveData: carrega dados do arquivo/storage local
  - SaveToStorage: persiste estado atual em disco
  - GetPlayTime: retorna tempo total jogado
  - UpdateCurrentStage: avança para novo estágio/wave
  - IsHeroUnlocked: verifica se herói foi desbloqueado
### TaskbarHero.Log.CubeDecorationLog
Registra estado de decoração de cubo (cores, tipo, variante) para persistência no salvamento
- campos: bfni, bfnj, bfnk, bfnl
  - kha: Serializar decoração em bytes/string para arquivo
  - khb: Desserializar e restaurar decoração do arquivo
### TaskbarHero.Log.CubeEngravingLog
Sistema de logging para registrar gravações/inscrições de cubos do jogo (persistência de eventos ou dados
- campos: bfnm, bfnn, bfno, bfnp
  - kha: LogEngraving ou RecordEngraving — escreve evento de gravação no buffer/disco
  - khb: GetEngraving ou ReadLastLog — recupera última gravação registrada
### TaskbarHero.Log.CubeExtractionLog
Registra eventos de extração/coleta de cubos no jogo
- campos: bfnq, bfnr, bfns, bfnt
  - kha: Log extração (timestamp, tipo, quantidade)
  - khb: Serializar/ler do disco
  - bfnq: Timestamp da extração
  - bfnr: Tipo/ID do cubo
  - bfns: Quantidade extraída
### TaskbarHero.Log.CubeInscriptionLog
Sistema de logging que registra eventos com dados espaciais (posição 3D + valor) de inscrição em cubos
- campos: bfnu, bfnv, bfnw, bfnx
  - kha: Adiciona/registra um novo evento de inscrição no log
  - khb: Lê/recupera eventos anteriores armazenados
### TaskbarHero.EasySaveData.CubeLevelSaveData
Armazenar e persistir dados de progresso do jogador em níveis cuboides (nível atual e experiência acumulada).
- campos: Level, Exp
  - Save: Persiste Level e Exp em arquivo/cloud storage
  - Load: Recupera dados salvos anteriormente
  - AddExp: Incrementa Exp do jogador
  - LevelUp: Avança Level quando Exp atinge threshold
  - Reset: Limpa Level e Exp para estado inicial
### TaskbarHero.Data.GradeInfoDataMap
Mapeia dados de classificação/grau (grade) do jogador para estruturas persistentes de salvamento.
  - Serialize: converte GradeInfo para formato salvável
  - Deserialize: reconstrói GradeInfo do armazenamento
  - GetGradeData: recupera informações de classificação
  - SetGradeData: atualiza dados de grade
  - MapGradeFields: mapeia campos de classificação entre formatos
### TaskbarHero.Log.HeroLevelUpLog
Registra e armazena dados de progressão de level up do herói (atributos antes/depois)
- campos: bfoi, bfoj
  - GetPreviousStats: retorna atributos do herói antes do level up (bfoi)
  - GetNewStats: retorna atributos após o level up (bfoj)
  - LogLevelUp: registra o evento de progressão no sistema de save
### TaskbarHero.EasySaveData.HeroSaveData
Armazenar e gerenciar dados persistentes do herói (nível, experiência, equipamento, habilidades desbloqueadas).
- campos: heroKey, HeroLevel, IsUnLock, HeroExp, AbilityPoint, AllocatedHeroAbilityPoint, equippedItemIds, equippedSKillKey, unlockedAttributeGroupKeys
  - GetHeroLevel: retorna o nível atual do herói
  - AddExperience: incrementa HeroExp e verifica progressão de nível
  - EquipItem: adiciona ID de item a equippedItemIds
  - EquipSkill: define equippedSKillKey para a habilidade selecionada
  - UnlockAttributeGroup: adiciona chave de atributo a unlockedAttributeGroupKeys
### TaskbarHero.InventoryDownloadToSteam
Sincroniza dados de inventário do jogador com a plataforma Steam.
  - Execute: inicia a sincronização/download do inventário para Steam
  - ValidateInventory: valida os dados do inventário antes de enviar
  - MapInventoryToSteamFormat: converte formato local para formato Steam
  - HandleSyncResponse: processa resposta do servidor Steam
  - CacheInventoryData: persiste os dados baixados localmente
### TaskbarHero.EasySaveData.InventorySaveData
Serializa o estado de um item individual do inventário (desbloqueio, ID único, posição) para salvamento em disco
- campos: Index, ItemUniqueId, IsUnlock, IsUnlockedByRune
  - Constructor(int index, string itemUniqueId, bool isUnlock, bool isUnlockedByRune): inicializa a struct/classe
  - Unlock(): muda IsUnlock para true (desbloqueio normal)
  - UnlockByRune(): muda IsUnlockedByRune para true (desbloqueio via mecânica especial)
  - GetItemIndex(): retorna Index do item na tabela de referência
  - IsFullyUnlocked(): retorna (IsUnlock || IsUnlockedByRune)
### TaskbarHero.EasySaveData.ItemSaveData
Serializar dados persistentes de um item, incluindo modificações (encantamentos, decorações, gravações, inscrições) e estado de sincronização
- campos: ItemKey, UniqueId, IsChaotic, IsBlocked, IsServerPendingItem, EnchantCount, EnchantData, GetSourceType, DecorationAppliedTotalCount, EngravingAppliedTotalCount, InscriptionAppliedTotalCount
  - Serialize: Converte para formato binário/JSON salvável
  - Deserialize: Restaura a partir do formato persistente
  - GetModificationBonus: Calcula bônus total das modificações aplicadas
  - Clone: Duplica a instância para cópia segura
  - ClearServerPending: Marca sincronização como concluída
### TaskbarHero.Data.MonsterInfoDataMap
Armazena e mapeia dados demonológicos de monstros (ID → atributos de combate).
  - GetMonsterInfo: busca atributos por ID de monstro
  - LoadFromFile/JSON: carrega mapa da persistência
  - UpdateMonsterData: altera stats de combate
### TaskbarHero.Data.PetStatInfoData
Armazena e gerencia uma modificação de estatística individual de um pet (valor base + tipo de modificador).
- campos: PetStatKey, StatType, MODTYPE, Value
  - GetModifiedValue: retorna Value com MODTYPE aplicado (aditivo/multiplicativo/substituição)
  - Clone: duplica objeto para serialização ou snapshot
  - ApplyModifier: soma/multiplica Value conforme MODTYPE
  - ToString: serializa para save game ou log
  - CalculateFinal: aplica MODTYPE e retorna valor final da stat
### TaskbarHero.ResetDataOptionPanel
Painel de confirmação com múltiplas validações (checkbox + slider hold) para resetar/limpar dados do jogo com animação de progresso.
- campos: bdfr, bdfs, bdft, m_notice1Checkbox, m_closeButton, m_resetConfirmButton, m_guideKeyBeforeScroll, m_guideKeyAfterScroll, m_scrollCompleteThreshold, m_deleteHoldButton, m_cancelButton, m_reCheckPanel, m_realDeleteSlider, m_deActiveObjectList, m_slider, m_warningReseting
  - OnResetSliderValueChanged: detecta progresso do slider e valida se atingiu threshold
  - ConfirmAndExecuteReset: executa limpeza de dados após todas as validações
  - AnimateResetProgress: anima pontos/loading enquanto processa reset
  - ValidateNoticeCheckbox: valida se checkbox de aviso foi marcado antes de permitir reset
  - UpdateUIWarningState: atualiza visual dos avisos e botões conforme estado da validação
### TaskbarHero.SerializableKeyValuePair`2
Par genérico chave-valor para serializar dicionários do jogo em JSON/binário
- campos: key, value
  - kaf: Construtor(key, value) — inicializa o par
  - kag: GetHashCode() — permite uso em dicionário ou hash
### TaskbarHero.SerializedBoxData
Armazena dados serializados de uma caixa/container do inventário com seu tipo, identificador único e quantidade de itens.
- campos: BoxTypes, BoxUniqueId, BoxQuantity
  - GetBoxType: retorna o tipo da caixa
  - GetBoxUniqueId: retorna ID único para identificação
  - GetBoxQuantity: retorna quantidade de itens
  - SetBoxQuantity: altera quantidade (consumo/adição)
  - Serialize/Deserialize: persiste/carrega dados em disco
### TaskbarHero.SerializedMailData
Serialização de dados do sistema de correio/mensagens do jogador para persistência em save
- campos: Keys, MailData
  - Serialize: Converte dados de mensagens em formato binário/JSON salvável
  - Deserialize: Carrega dados persistidos de volta para estrutura de memória
  - AddMailEntry: Insere nova mensagem indexada em Keys
  - GetMailByKey: Recupera dados de mensagem específica usando identificador
  - ClearExpiredMail: Remove mensagens expiradas ou lidas
### TaskbarHero.Log.StageClearLog
Registra e persiste dados de conclusão de estágio, incluindo pontuação, tempo e resultado.
- campos: bfpk, bfpl, bfpm, bfpn
  - SaveLog: Persiste os 4 campos (stage, score, time, status) no disco ou cloud
  - LoadLog: Carrega dados salvos de stage clears anteriores
  - SerializeData: Converte campos em formato persistível (JSON/binary)
  - UploadToServer: Envia log pra sincronização online
  - GetAggregateStats: Calcula estatísticas (maior score, melhor tempo)
### TaskbarHero.Data.StageInfoDataMap
Mapa de dados que armazena e gerencia informações de estágios/fases do jogo (progressão, configurações, estado).
  - GetStageInfo: Retorna informações de um estágio específico
  - SetStageInfo: Armazena/atualiza dados de um estágio
  - ContainsStage: Verifica se um estágio existe no mapa
  - LoadFromDisk: Carrega mapa de dados do arquivo de save
  - SaveToDisk: Persiste mapa de dados em arquivo
### TaskbarHero.Data.StageLevelInfoData
Armazena parâmetros de configuração, dificuldade e balanceamento de um estágio/nível do jogo.
- campos: bgpu, bgpv, bgpw, bgpx, bgpy
  - GetDifficultyMultiplier: retorna o multiplicador de dificuldade agregado dos 5 parâmetros
  - ValidateStageData: valida se os valores de bgpu-bgpy estão dentro dos limites permitidos
  - CalculateEnemyWaveCount: usa bgpu/bgpv para determinar número de ondas de inimigos
  - GetRewardMultiplier: combina bgpw/bgpx/bgpy para calcular multiplicador de recompensa
  - ToString: serializa os 5 parâmetros para debug/logging
### TaskbarHero.Data.StatusEffectInfoDataMap
Mapa de dados que armazena e gerencia informações sobre efeitos de status (buffs/debuffs) do jogo
  - Get: obtém informações de um efeito de status por ID
  - Add: registra um novo tipo de efeito de status
  - Contains: verifica se um efeito de status existe no mapa
  - GetAll: retorna todos os efeitos de status cadastrados
  - Remove: remove um efeito de status do mapa
### TaskbarHero.SteamItemRecordData
Registra metadados de itens do jogo (aquisição, uso ou sincronização) com Steam, armazenando ID único e timestamp do evento.
- campos: UniqueId, RecordTime
  - RecordItem: registra quando um item foi adquirido ou usado
  - GetRecordTime: retorna o timestamp do registro
  - SyncWithSteam: sincroniza dados com plataforma Steam
  - IsRecorded: verifica se um item já foi registrado
  - ClearRecord: limpa o registro do item
### TaskbarHero.UI.SubRecipeSlotButton
Botão de slot para seleção/exibição de sub-receitas no UI de crafting, com indicador de desbloqueável.
- campos: m_iconParent, bfxb, bfxc, bfxd, m_unlockableRedDot
  - UpdateSlotDisplay: Atualiza ícone e texto descritivo da receita no slot
  - OnSlotClicked: Executa ação quando o botão é pressionado (confirma receita)
  - SetRecipeData: Carrega e vincula dados da sub-receita ao slot
  - UpdateUnlockIndicator: Controla visibilidade do red dot de notificação
  - SetInteractable: Ativa/desativa o botão conforme disponibilidade da receita
### TaskbarHero.VerifyDlc
Verifica se um DLC específico está instalado, desbloqueado ou disponível para acesso pelo jogador
  - GetParam: obtém o ID ou identificador do DLC a verificar
  - GetResultData: retorna booleano (está desbloqueado?) ou dados de status/metadados do DLC

## save,player  (1)
### TaskbarHero.AccountSaveData
Gerencia dados persistentes da conta do jogador, incluindo progresso, tempo de jogo e sincronização com Steam
- campos: version, isFirstRun, forceApplyServerData, lastSavedTime, playTime, lastExitSteamUnixSecEncrypted, lastExitSteamUnixSecKey, sessionCounter, ownerSteamId
  - SerializeToJson: serializa para formato JSON pra persistência em disco
  - DeserializeFromJson: desserializa from JSON/arquivo para memória
  - ValidateIntegrity: valida checksums/integridade via campos criptografados
  - SyncWithServer: sincroniza dados locais com servidor (forceApplyServerData)
  - MigrateVersion: trata upgrade de versão de save antiga pra nova

## skill  (46)
### TaskbarHero.ActiveSkillSlotDetail
Gerencia a interface de detalhe de um slot de habilidade ativa, permitindo equipar/desequipar skills, alocar pontos e visualizar cooldown
- campos: go_Check, button_Refund, button_Allocate, button_EquipSkill, button_UnEquipSkill, button_Exit, m_skillCoolTimeText, bdgc, bdgd, bdge, bdgf, bdgg
  - OnClickEquipSkill: handler do botão equipar habilidade
  - OnClickUnequipSkill: handler do botão desequipar habilidade
  - OnClickRefund: handler para reembolso de pontos investidos
  - UpdateCooldownDisplay: atualiza texto m_skillCoolTimeText com tempo restante
  - RefreshBadgeValues: renderiza bdgc-bdgg com stats (dano, mana, cooldown, etc)
### TaskbarHero.Combat.AnimaitionProbabiltyData
Armazena dados de probabilidade de animação para skills em combate, vinculando uma habilidade a um clip de animação específico com chance de execução.
- campos: skillKey, clip, probability, baseStateName
  - GetSkillKey: retorna o identificador único da skill
  - GetClip: retorna o AnimationClip associado à skill
  - GetProbability: retorna o valor de probabilidade (0-1) de executar essa animação
  - GetBaseStateName: retorna o nome do estado animator base para transição
  - IsValidForSkill: valida se dados são consistentes para a skill associada
### TaskbarHero.Combat.ArcherArrowRainAtk
Skill especial de Arqueiro que dispara chuva de flechas em área
- campos: bgsj
  - mlz: Execute/Start - Inicia o ataque e spawna flechas
  - mma: Tick/Update - Controla timing e dano contínuo da chuva
### TaskbarHero.Combat.ArcherSkewerShot
Implementa um ataque especial de arqueiro que penetra/atravessa múltiplos inimigos em linha
- campos: bgsu
  - FireSkewer: Lança o projétil que crava inimigos
  - ApplyPenetration: Aplica dano com efeito de penetração
  - CalculateArcTrajectory: Calcula trajetória do ataque
  - OnHitEnemy: Processa impacto contra cada inimigo atingido
### TaskbarHero.BackendPostSaveData
Persiste dados de missões/quests publicadas pelo servidor com validade temporal e sistema de reserva de slots.
- campos: title, content, author, inDate, expirationTicks, reservationTicks, items
  - IsExpired: verifica se expirationTicks passou
  - IsReserved: verifica se reservationTicks ainda é válido (slot ocupado)
  - GetReward: extrai items para player
  - ReserveSlot: marca tempo de reserva pro player atual
  - ToString/DebugDisplay: serializa pra UI/log
### TaskbarHero.CoolDownData
Gerenciar cooldown (tempo de espera/reutilização) para skills ou ações reutilizáveis.
- campos: o_updateTime, o_cooldownSec, updateTime, cooldownSec
  - IsReady: Verifica se o cooldown expirou e a ação pode ser executada novamente
  - Reset: Inicia/reseta o cooldown com o tempo especificado em cooldownSec
  - UpdateCooldown: Decrementa o tempo restante baseado no delta-time desde updateTime
  - GetRemainingTime: Retorna quantos segundos faltam até o cooldown expirar
  - SetCooldownDuration: Define a duração do cooldown em segundos
### TaskbarHero.CooldownPair
Gerencia pares de cooldowns diferenciados por raridade (comum e rara) para skills ou itens
- campos: common, rare
  - GetCooldown: retorna tempo de recarga baseado em raridade
  - SetCooldown: define duração do cooldown
  - ReduceCooldown: diminui tempo restante (tick)
  - IsReady: verifica se cooldown expirou
  - Reset: restaura ao valor máximo
### TaskbarHero.Log.CubeAlchemyLog
Log/registro de eventos de alquimia de cubos no sistema de crafting do jogo.
- campos: bfna, bfnb, bfnc, bfnd, bfne
  - kha: LogAlchemyEvent - registra uma transformação de cubo (entrada, saída, resultado)
  - khb: SaveLogToFile - persiste os eventos de alquimia no disco ou banco de dados
  - campo bfna: evento_id ou timestamp
  - campo bfnb: tipo_alquimia ou recipe_hash
  - campo bfnc: cubo_entrada ou material_origin
### TaskbarHero.Data.EATTRIBUTETYPE
Enumeração que define tipos de atributos/slots de habilidades (passivas, ativas, slots) para heróis.
- campos: value__, PASSIVESKILL, ACTIVESKILL, ACTIVESKILLSLOT
  - GetName: retorna string do tipo de atributo
  - GetValue: retorna valor numérico da enum
  - IsActive: verifica se é habilidade ativa ou slot ativo
  - IsPassive: verifica se é habilidade passiva
### TaskbarHero.FakeDieCreator
Gera dados falsos (simulated rolls/values) para teste de skills, cacheia skills associados
- campos: SkillCache, FakeDieCount
  - CreateFakeDie: Instancia um dado falso com valor simulado
  - GetSkillFromCache: Recupera skill do cache por ID
  - IncrementDieCount: Incrementa contador de dados criados
  - CachifySkill: Armazena skill no cache local
  - GetTotalFakeDice: Retorna contagem total de dados gerados
### TaskbarHero.Effect.FreezeEffectAnimator
Anima sequencialmente efeitos visuais de congelamento (sprites start→idle→shake→end) com variantes para boss vs inimigo normal.
- campos: m_spriteRenderer, m_normalStartSprites, m_normalIdleSprites, m_normalShakeSprites, m_normalEndSprites, m_actBossStartSprites, m_actBossIdleSprites, m_actBossShakeSprites, m_actBossEndSprites, m_frameInterval, bfqc, bfqd, bfqe, bfqf, OnEffectComplete, bfqg
  - kjr: Play - inicia a animação desde Start
  - kjs: UpdateFrame - avança para próximo sprite baseado em frameInterval
  - kjt: PlayIdleSprites - executa loop de sprites Idle
  - kju: PlayShakeSprites - toca sequência Shake
  - kjv: PlayEndSprites - reproduz sprites End e dispara OnEffectComplete
### TaskbarHero.Combat.Projectile.FrostBolt
Projétil mágico de gelo que congela inimigos ao impactar
  - Start: inicializa velocidade e orientação do projétil
  - Update: move o projétil pela cena
  - OnTriggerEnter: detecta colisão com alvo e dispara efeito
  - ApplyFrostEffect: congela inimigo e reduz velocidade
  - Destroy: remove projétil após impacto
### TaskbarHero.UI.HeroHeadUpCoolTimeSlider
Componente HUD que exibe o cooldown de uma habilidade em tempo real como slider visual com sprite animado
- campos: m_spriteRendererList, ActivateSprite, bfxp, bfxq
  - SetCoolTime: Define o tempo de cooldown e inicia a contagem
  - UpdateSlider: Atualiza a posição/tamanho da barra conforme cooldown diminui
  - OnCoolTimeComplete: Callback quando o cooldown termina, exibe ActivateSprite
  - SetActivateSprite: Configura o sprite a exibir quando habilidade fica disponível
  - Initialize: Prepara os sprite renderers da lista
### TaskbarHero.UI.HeroPreviewSkillSlot
Gerencia exibição e interação de um slot de skill equipável no hero, mostrando estado (equipado/desequipado) e nível
- campos: m_equipCheckObject, m_unEquipDimmedMaterial, m_levelText, bfzx, bfzy
  - lgp: Inicializa o slot com dados de skill (skillId, level)
  - lhq: Mostra indicador de equipado e remove material dimmed
  - lhr: Esconde indicador de equipado e aplica material dimmed
  - lhs: Atualiza m_levelText com o nível do skill
  - lgv: Atualiza visuais completos baseado no estado atual (equipado/level)
### TaskbarHero.HoldButton
Sistema de detecção de botão pressionado e mantido (hold) que dispara eventos de progresso e conclusão quando atinge limiar de tempo.
- campos: m_holdThreshold, bdez, bdfa, bdfb, OnHoldComplete, OnHoldProgress
  - ResetHold: reseta o timer e cancela o estado de hold
  - UpdateHoldDuration: incrementa o tempo de hold no Update
  - NotifyProgress: dispara OnHoldProgress com valor normalizado
  - CompleteHold: valida threshold e dispara OnHoldComplete
### TaskbarHero.Combat.KnightSacredBlade
Habilidade de ataque sagrado do Cavaleiro no sistema de combate
- campos: bguc
  - mlz: Execute / Cast — executa o ataque/ativa a skill
  - mma: GetDescription / GetInfo — retorna detalhes (dano, descrição, custo)
  - bguc (campo): damage / cost / cooldown — valor do atributo da skill
### TaskbarHero.MODTYPE
Enum que define três tipos de modificadores numéricos: fixo (FLAT), aditivo percentual (ADDITIVE) e multiplicativo (MULTIPLICATIVE).
- campos: value__, FLAT, ADDITIVE, MULTIPLICATIVE
  - GetModifierType: retorna o tipo do modificador baseado no valor enum
  - ApplyModifier: aplica o modificador ao valor base conforme o tipo
  - CalculateBonus: calcula bônus final usando a regra do modificador
  - IsPercentBased: retorna true se ADDITIVE ou MULTIPLICATIVE
  - GetDisplayFormat: formata exibição (+10 vs +10% vs ×1.5)
### TaskbarHero.Combat.MassiveVoidCallArea
Controlador de área AoE massiva de magia Void que gerencia objetos de efeito visuais para ataques/skills de combate
- campos: m_effectObjects
  - InitializeEffects: Instancia e configura os objetos de efeito armazenados em m_effectObjects
  - ActivateArea: Ativa a zona AoE e reproduz animações/efeitos dos m_effectObjects
  - UpdateEffectPosition: Atualiza posição dos efeitos conforme área se move/expande
  - DeactivateEffects: Desativa e limpa os objetos de efeito quando a habilidade termina
  - GetAffectedTargets: Detecta entidades dentro da área para aplicar dano/status
### TaskbarHero.Skill.ModStatExchangeType
Enum que define tipos de modificadores de estatísticas com diferentes escalas de divisão (bruto/1000, bruto/100, pré-dividido) para cálculos de troca ou bônus de stat.
- campos: value__, Raw_Divide1000, Raw_Divide100, Divided
  - GetDivisor: retorna divisor associado (1000, 100, 1)
  - IsRawFormat: verifica se é valor bruto
  - ConvertValue: converte valor entre tipos de escala
  - Parse: converte int/string para tipo de troca correspondente
  - ToString: serializa tipo para formato legível
### TaskbarHero.Data.PassiveSkillInfoData
Armazena dados de habilidades passivas (buffs permanentes) com propriedades como ID, efeito, nível e ícone.
- campos: bgoy, bgoz, bgpa, bgpb, bgpc
  - GetSkillEffect: Retorna o bônus ou efeito estatístico da skill
  - GetSkillLevel: Retorna o nível atual da habilidade passiva
  - GetSkillIcon: Retorna referência ou ID do ícone visual
  - IsUnlocked: Verifica se a skill está desbloqueada
  - GetSkillBonus: Retorna valor numérico do bonus (dano, defesa, etc)
### TaskbarHero.Combat.PriestBlessingOfMight
Skill de buff que aumenta poder (Might) do Sacerdote por duração limitada, aplicável a aliados em combate.
- campos: bguo, bgup
  - Apply: Aplica o buff ao alvo, armazenando duração e magnitude de aumento de poder
  - Remove: Desativa o buff e remove modificadores do alvo
  - Update: Tick de duração, decrementa tempo restante e remove se expirado
  - GetModifier: Retorna valor atual de aumento de poder (bguo/bgup interpretados)
### TaskbarHero.Combat.PriestBlessingOfWarding
Habilidade de buff que aumenta defesa/proteção (Warding) de aliados por duração limitada, reduzindo dano recebido em combate.
- campos: bguq, bgur
  - mne: Apply - Aplica o buff ao alvo com duração (bgur) e magnitude de redução de dano (bguq)
  - moy: Remove - Desativa o buff e restaura defesa original do alvo
  - mox: Update - Tick de duração, decrementa tempo restante e remove automaticamente se expirado
  - moz: GetModifier - Retorna valor atual de defesa/absorção de dano
### TaskbarHero.Combat.PriestHeal
Gerencia a habilidade de cura do sacerdote, executando restauração de HP em aliados durante combate.
- campos: bgus
  - Execute: Aplica o efeito de cura aos alvos com base no campo bgus (possivelmente base HP restoration)
  - CanCast: Valida se a habilidade pode ser usada (recursos, cooldown, status)
  - GetHealAmount: Calcula o dano/cura final baseado em stats
  - OnCastComplete: Callback pós-execução da cura
### TaskbarHero.Combat.PriestResurrection
Habilidade de combate que ressuscita aliados derrotados durante a batalha.
- campos: bgut
  - mlz: Carrega/inicializa a habilidade de ressurreição
  - mma: Executa efeito de ressurreição e restaura estado do aliado
### TaskbarHero.Combat.PriestSanctuary
Santuário defensivo - absorve dano ou nega status negativo em combate
- campos: bguu
  - mlz: Ativa proteção/barreira
  - mma: Aplica buff de defesa ou immunity
  - mnj: Desativa ou decai santuário
### TaskbarHero.Combat.PriestWrathOfHeaven
Priest divine wrath skill that strikes enemies from heaven with spell damage and possible crowd control
- campos: bguv
  - mlz: Execute/Cast - activates the wrath ability, targeting enemies
  - mma: CalculateDamage/ApplyEffect - computes divine damage and applies heaven strike effects
  - bguv: cooldown_ms or mana_cost - tracks ability cooldown or resource consumption
### TaskbarHero.Data.RuneLevelInfoData
Armazena dados de progressão de uma runa individual (nível, custos, multiplicadores de dano/efeito)
- campos: bgpl, bgpm, bgpn, bgpo, bgpp, bgpq
  - mjz: CalculateLevelUpCost() - retorna custo em ouro/mana para evoluir para próximo nível
  - GetMaxLevel() - retorna limite de progressão da runa
  - GetEffectMultiplier() - retorna modificador de dano/efeito baseado no nível
  - CanLevelUp() - verifica se há recursos suficientes para evoluir
  - GetCooldownReduction() - retorna redução de cooldown por nível
### TaskbarHero.UI.Rune.RuneNode
Nó de UI que representa uma runa individual com sistema de level up e encadeamento visual na interface.
- campos: m_rectTransform, m_runeKey, m_levelUpButton, m_runeNodeVisual, m_nextRuneKey, bgih, bgii, bgij, bgik, bgil, bgim, bgin, OnRuneLevelChanged, bgio, bgip
  - Initialize: Configura referências do RectTransform e carrega dados da runa via m_runeKey
  - LevelUpRune: Incrementa level, dispara OnRuneLevelChanged, atualiza visual
  - RefreshUIDisplay: Atualiza visual (sprite, texto, cores) baseado em estado de level
  - OnLevelUpButtonClicked: Handler do clique em m_levelUpButton com validação de recursos
  - LinkNextRuneNode: Posiciona próxima runa encadeando via m_nextRuneKey
### TaskbarHero.UI.Rune.RuneTooltip
Exibe e gerencia tooltip interativo de runas mostrando informações, preço, nível e estado de compra/upgrade com validação visual de affordances
- campos: _usm, rectTooltip, m_runeNameText, m_runeDescriptionText, m_runeCostValueText, m_runeLevelText, m_bottomImage, m_PricePanel, m_canPurchaseColor, m_cannotPurchaseColor, m_notMaxLevelBottonColor, m_maxLevelBottonColor, m_zeroLevelAndCannotPurchaseColor, m_zeroLevelCanPurchaseColor, m_maxLevelPanel, m_runeNodeVisual
  - mbr: UpdateRuneInfo — atualiza nome, descrição, custo e nível exibidos no tooltip
  - mbs: UpdatePurchaseState — valida se pode comprar e aplica cores m_canPurchaseColor vs m_cannotPurchaseColor
  - mbt: UpdateLevelVisuals — controla estado botão level-up (m_notMaxLevelBottonColor vs m_maxLevelBottonColor)
  - mbu: OnLevelUpButtonPressed — handler do m_levelUpButton com animação m_levelUp_SpriteAnimation
  - mbv: RefreshNodeConnections — atualiza m_noticePreviewNodeConnectObject com preview de conexões da runa
### TaskbarHero.SKILLTYPE
Define categorias de habilidades que o herói pode lançar na barra de skills
- campos: value__, None, Direct, Projectile, Aoe, SpawnTurret, SpawnTrap, SpawnSomething, SpawnRandomMonster
  - ToString(): Retorna nome legível do tipo
  - IsSummon(): Verifica se é habilidade de invocação (Spawn*)
  - IsProjectile(): Verifica se usa projétil
  - GetTargetType(): Retorna se é single/aoe/spawn
  - GetCooldownBase(): Retorna CD padrão do tipo
### TaskbarHero.UI.SkillDescriptionPanel
Painel de UI que exibe descrição detalhada de uma habilidade (nome, ícone, nível, atributos e instruções de interação).
- campos: m_canvasGroup, m_skillImage, m_skillLevel, m_skillName, m_attributeDescription, m_mouseActionText, m_textWidthHeightSetter
  - lpr: SetSkillInfo — carrega dados de uma skill nos campos (imagem, nome, nível, descrição)
  - lps: Show — ativa canvasGroup (alpha=1, blocksRaycasts=true) para exibir o painel
  - lpt: Hide — desativa canvasGroup (alpha=0, blocksRaycasts=false) para esconder o painel
### TaskbarHero.Data.SkillLevelInfoData
Armazena os valores base (power, speed, time) de uma skill em cada nível de progressão.
- campos: bgpr, bgps, bgpt
  - GetBasePower: retorna bgpr (dano ou poder base da skill)
  - GetBaseSpeed: retorna bgps (velocidade de execução ou custo de energia)
  - GetBaseCooldown: retorna bgpt (tempo de recarga entre usos)
  - GetLevelMultiplier: calcula multiplicador de escala para o nível atual
  - GetDescription: formata string de descrição da skill com os valores
### TaskbarHero.UI.SkillSlot
Slot de habilidade (skill) na UI com interatividade visual, exibindo nível, nome e ícone com feedback de hover/locked
- campos: slotIndex, text_SkillLevel, text_SkillName, button_Click, go_Lock, image_Back, image_Skil, image_SkillIconBG, sprite_Normal, sprite_Hover, m_hoverEdge, bgac, bgad, bgae, bgaf, bgag
  - lin: Initialize/Setup — inicializa o slot com dados de skill
  - lio: LoadSkillData — carrega e aplica dados da habilidade
  - lip: UpdateVisuals/RefreshDisplay — atualiza sprites e textos
  - liq: SetLocked — ativa go_Lock e desabilita interação quando skill indisponível
  - lgp: OnClick — seleciona a skill ao clicar no botão
### TaskbarHero.UI.SkillStatusDescriptionPanel
Painel de UI que exibe informações detalhadas de habilidades ativas/passivas do jogador com ícone, nível, cooldown e DPS.
- campos: m_canvasGroup, m_activeDefaultBG, m_activeMaxBG, m_passiveDefaultBG, m_passiveMaxBG, m_greyMaterial, m_normalPanel, m_normalDescription, m_normalTextHeightSetter, m_normalSkillLevel, m_normalSkillIcon, m_normalSkillIconBG, m_normalUsingEffect, m_normalSkillNameText, m_normalPassiveSkillNameText, m_detailSkillNameText
  - Show: exibe o painel com fade in via canvasGroup
  - Hide: esconde o painel com fade out
  - UpdateNormalSkillInfo: atualiza descrição e ícone no painel normal
  - UpdateDetailSkillInfo: atualiza nome, nível, cooldown e DPS no painel detalhado
  - RefreshSkillVisuals: aplica material cinza ou normal baseado em estado ativo/passivo
### TaskbarHero.Combat.SlayerBloodlust
Skill de frenesi de combate que aumenta ataque/velocidade do Slayer ao acumular kills ou ativar manualmente.
  - ApplyBloodlust: Ativa efeito, aplica buffs de dano/ataque speed
  - RemoveBloodlust: Desativa efeito e remove buffs
  - UpdateBloodlustDuration: Decrementa duração ou verifica condição de manutenção do efeito
### TaskbarHero.Combat.SlayerCommanderCry
Habilidade de combate que executa um grito/comando de um matador/guerreiro afetando alvo ou área
  - Execute: Ativa o grito e aplica efeitos
  - ApplyEffect: Aplica dano ou buff/debuff aos alvos
  - GetCooldown: Retorna tempo de recarga
  - CalculateDamage: Calcula dano base do grito
  - OnCast: Dispara lógica de casting da habilidade
### TaskbarHero.Combat.SlayerCrushingBlow
Skill de combate melee — Crushing Blow do Slayer: golpe especial que concentra dano contra um alvo, com gerenciamento de cooldown/recurso.
- campos: bgvs, bgvt, currentTarget
  - mlz: Inicializa/valida pré-condições (alvo válido, recurso suficiente)
  - mpo: Executa o golpe e aplica dano ao currentTarget
  - mma: Calcula magnitude do dano com base em bgvs (status do atacante)
  - mpp: Gerencia cooldown/recover ou consome recurso (energia/mana)
### TaskbarHero.Combat.SlayerGroundSlam
Habilidade de combate que causa dano em área ao bater o chão
  - Execute: Dispara a animação e lógica do ground slam
  - DealAOEDamage: Aplica dano em área ao redor do epicentro
  - ApplyCooldown: Ativa o período de recarga da habilidade
  - PlayVFX: Renderiza efeitos visuais (explosão, tremor, pó)
  - GetDamageValue: Calcula dano base + multiplicadores de stats
### TaskbarHero.Combat.SlayerSlamJump
Habilidade ofensiva de combate que permite executar um salto com golpe de área, causando dano e knockback aos inimigos próximos.
- campos: bgwd, bgwe, bgwf, bgwg, bgwh, bgwi, bgwj
  - Cast: Iniciar execução da habilidade (animação + física)
  - Update: Atualizar posição e estado durante execução
  - OnHit: Aplicar dano quando atinge inimigos
  - ApplyKnockback: Empurrar inimigos afetados
  - CheckCooldown: Verificar disponibilidade para reutilizar
### TaskbarHero.UI.StatusPanelSkillSlot
Controla exibição interativa de um slot de habilidade no painel de status, gerenciando descrição ao hover e efeito visual de equipamento.
- campos: canEquipEffect, m_isShowDescription, bgak
  - lgq: ShowDescription - exibe descrição da habilidade quando mouse entra via PointerHover
  - ApplyCanEquipEffect: ativa efeito visual em canEquipEffect indicando disponibilidade
  - HideDescription: esconde descrição em m_isShowDescription no PointerExit
  - SetSkillData: inicializa slot com dados da skill (bgak = background/ativo)
  - UpdateHighlight: atualiza estado visual de seleção/destaque do slot
### TaskbarHero.SteamSlotData
Gerencia um slot de item/habilidade com rastreamento de cooldown até seu próximo uso
- campos: o_slot, o_cooldownUntil
  - Set: Define o ID do slot e reseta o cooldown ou atualiza para um novo tempo
  - GetCooldownRemaining: Retorna quantos ms/s faltam para o cooldown expirar
  - IsReady: Verifica se cooldown expirou e slot está disponível para uso
  - ApplyCooldown: Aplica novo cooldown após ativar a habilidade/item
  - Reset: Limpa o slot e zera o cooldown
### TaskbarHero.UI.UI_Rune
UI para visualizar e gerenciar runas equipáveis com estatísticas de batalha, exploração e recompensas
- campos: Button_Close, Button_Close_Down, m_runePage, m_accountStatusCategoryData, m_battleStatParent, m_explorationStatParent, m_rewardStatParent, m_runeStatInfoPrefab, bggu, bggv, bggw, bggx
  - hlw: OnEnable - Inicializa UI e carrega página de runas
  - hlt: OnDisable - Limpa listeners e desativa componentes
  - lwj: PopulateRunePage - Instancia prefabs de runas na página
  - lwk: RefreshRuneStats - Atualiza stats exibidos ao selecionar runa
  - lwl: OnCloseButtonClicked - Handler do botão de fechar
### TaskbarHero.Skill.UniqueModStat
Representa uma modificação única de stat para uma skill, armazenando descrição, tipo de câmbio/conversão e valor do modificador
- campos: StatDescription, ExchangeType, Value
  - GetStatDescription: Retorna a descrição da stat sendo modificada
  - GetExchangeType: Retorna o tipo de troca ou conversão (e.g. flat, percent, additive)
  - GetValue: Retorna o valor numérico do modificador
  - ApplyTo: Aplica a modificação a um valor base de stat
  - GetDisplayText: Retorna string formatada para UI (ex: '+15 ATK' ou '×1.5 DEF')
### TaskbarHero.Combat.WizardFlameHydra
Habilidade de combate que lança múltiplas chamas em padrão de hidra
  - mlz: CastFlameHydra ou Execute - dispara os projéteis de chama
  - mma: TickDuration ou UpdateCooldown - gerencia duração/recarga da habilidade
### TaskbarHero.Combat.WizardMeteorStrike
Implementa a skill Meteor Strike de um mago — invoca chuva de meteoros que caem na área alvo infligindo dano
- campos: bgwy
  - mml: CastMeteor — inicia o ataque, dispara meteoros em velocidade/direção
  - mlz: UpdateMeteorFall — atualiza posição/física dos meteoros durante queda
  - mma: OnMeteorImpact — detecta colisão com inimigos e aplica dano
  - mqt: ResetCooldown — limpa estado e reinicia timer de reutilização
### TaskbarHero.ObjectPooler.WrathOfHeavenHitEffect
Efeito visual e gameplay de impacto da habilidade especial 'Ira do Céu' com dano/controle por raio em área (AOE)
- campos: LightningAOE, bfkn, bfko, bfkp, bfkq, bfkr, bfks, bfkt
  - PlayEffect/Activate: Inicializa animação, partículas e som do impacto de raio
  - DealDamage: Aplica dano e efeitos (stun/knockback) aos inimigos no raio
  - Cleanup/Disable: Reseta objeto para reutilização no pool após duração
  - CalculateAOE: Determina área/alcance do LightningAOE e entidades atingidas
  - HandleCollision: Detecta e processa acertos em inimigos/obstáculos

## ui  (166)
### TaskbarHero.UI.ActiveSkillAttributeSlot
Gerencia a visualização de um slot de skill ativa na taskbar, controlando indicadores de notificação (ponto vermelho) e efeitos visuais de brilho
- campos: RedDotComponent, usingEffect, m_glowEffect
  - ley: RefreshRedDotIndicator
  - lez: HideRedDotIndicator
  - lfa: PlayGlowEffect
  - lfb: StopGlowEffect
### TaskbarHero.UI.ActiveSkillEquipPanel
Painel de UI que permite ao jogador equipar e gerenciar skills ativos em slots de combate.
- campos: button_Exit, currentAttributeCache, currentHeroCache, allSlots, bfzp, bfzq
  - lfc: Inicializa painel com slots de skill disponíveis
  - lfd: Atualiza cache de atributos/herói e refresca display
  - lfe: Manipula clique em slot (seleciona skill para equipar)
  - lff: Remove skill de um slot (desquipa)
  - lfg: Carrega dados de skills disponíveis em bfzp/bfzq
### TaskbarHero.Editor.AlwaysVisibleOutline
Ferramenta de editor Unity que desenha contornos sempre visíveis com ponto central opcional para visualizar limites/áreas de objetos.
- campos: outlineColor, centerColor, showCenterPoint, bgjk
  - OnDrawGizmos: Renderiza o outline e centerPoint como gizmos no viewport do editor
  - bgjk: Provavelmente getter/setter ou cache de bounds/renderer do objeto
### TaskbarHero.UI.ArrangePageHeroViewSetter
Configurar e exibir a página de visualização do herói (retrato, status, habilidades, ilustração completa).
- campos: m_portraitImage, m_textSetter, m_statusButton, m_skillSlots, m_openFullIllustrationButton
  - lcd: SetupHeroData - inicializa/atualiza imagem, nome e informações do herói
  - lce: PopulateSkillSlots - popula os slots com as habilidades do herói
  - lcf: OnStatusButtonClicked - handler para exibir detalhes de status/atributos
  - lcg: OnOpenFullIllustrationClicked - handler para abrir a ilustração completa do herói em tela cheia
  - lch: RefreshHeroView - atualiza toda a visualização (chamado ao entrar na página)
### TaskbarHero.AttributeLevelSlot
Gerencia visualmente um slot de atributo do herói na barra de tarefas, mostrando nível e estado (ativo/inativo/bloqueado).
- campos: heroKey, groupKey, slotDatas, expandSlot, go_Active, go_Deactive, go_Lock, m_unlockEffect, bdgh, bdgi
  - Initialize: Configura slot com heroKey e groupKey, carrega dados iniciais
  - Refresh: Atualiza visuais (ativo/inativo/bloqueado) baseado no estado do herói
  - SetLocked: Ativa go_Lock e desativa interatividade até desbloqueio
  - TryUnlock: Valida condições, aplica m_unlockEffect, muda para go_Active
  - OnClick: Handler de clique para expandSlot ou ativar interação com atributo
### TaskbarHero.UI.AttributeSlot
Componente UI interativo para slot de atributo (herói/personagem) com seleção, estados visuais (hover/ativo/desabilitado), nível e efeitos de upgrade/refund.
- campos: image_Icon, Image_back, text_Level, button_Interrept, image_Hover, sprite_Normal, sprite_Hover, sprite_Active, go_Check, maxLevelBGSprite, greyMaterial, m_selectImage, m_levelUpEffect, m_refundEffect, _cache, bfzr
  - ley: SetLevel ou UpdateLevel — atualiza text_Level com novo valor
  - lfm: SetSelected ou Select — marca isSelect=true e aplica m_selectImage
  - lfn: SetHovered — alterna sprite entre sprite_Normal e sprite_Hover
  - lfo: SetActive — aplica sprite_Active e ativa go_Check
  - lfb: SetDisabled — aplica greyMaterial (desabilitado visualmente)
### TaskbarHero.UI.AutoBoxOnTooltipTrigger
Automaticamente mostra/posiciona uma caixa de informações de stage quando o mouse passa sobre um elemento com tooltip, e a oculta ao sair.
- campos: m_stageBox
  - OnPointerEnter: mostra e posiciona a m_stageBox
  - OnEnable: inicializa referências da caixa de tooltip
  - ShowStageBox ou UpdateBoxPosition: atualiza posição/visibilidade da caixa
### TaskbarHero.Option.BaseOption
Componente base UI para navegação entre opções com feedback de sprite animation
- campos: button_Next, button_Prev, spriteAnimation, bfuo, _isComplete, bfup
  - Show: exibe a opção e inicializa estado visual
  - Hide: oculta a opção e limpa referências
  - SelectOption: marca opção como selecionada (atualiza sprite)
  - SetComplete: marca _isComplete=true e anima transição
  - UpdateNavigation: habilita/desabilita button_Next/Prev baseado em índice
### TaskbarHero.UI.BoxAlphaController
Controlar a transparência/opacidade animada de um elemento de UI usando CanvasGroup (fade in/out).
- campos: m_canvasGroup
  - lkn: FadeIn() — aumenta alpha do CanvasGroup gradualmente
  - lko: FadeOut() — diminui alpha do CanvasGroup gradualmente
  - OnDestroy: cancela animação/corrotina ao destruir o objeto
  - .ctor: inicializa referência ao CanvasGroup
### TaskbarHero.UI.ComboBoxButton
Botão de UI que controla abertura/fechamento de uma caixa de seleção (ComboBox) com feedback visual de hover
- campos: m_mouseHoverColor, m_defaultColor, m_selectShowText, m_comboBoxOffChanger, m_comboBoxOnChanger, m_comboBoxObject, bfwm, OnComboBoxOpened, OnComboBoxClosed
  - kxg: Initialize - configura referências iniciais do ComboBox
  - kxh: OpenComboBox - ativa m_comboBoxObject e dispara OnComboBoxOpened
  - kxi: UpdateHoverColor - aplica m_mouseHoverColor no OnPointerEnter
  - kxj: CloseComboBox - desativa ComboBox e dispara OnComboBoxClosed
  - kxk: UpdateSelectedText - atualiza m_selectShowText com valor selecionado
### TaskbarHero.UI.CubeCraftingButton
Botão interativo de UI que apresenta receitas de crafting com feedback visual de hover e notificação de disponibilidade de itens
- campos: m_image, m_hoverSpriteImage, m_defaultImage, m_selectNoticeImage, m_inventorySlot, bghk, bghl, bghm, bghn
  - lyp: ShowHoverState - ativa imagem hover e exibe notice de seleção
  - lyq: ResetHoverState - restaura imagem padrão e limpa notificação
  - UpdateInventoryDisplay - sincroniza sprites com dados do inventorySlot
  - RefreshCraftingAvailability - habilita/desabilita botão baseado em recursos
  - AnimateCraftNotification - feedback visual quando crafting disponível
### TaskbarHero.UI.CubeCraftingSelectMaterialSubButton
Sub-botão interativo da UI de crafting por cubos que permite selecionar material com preview de hover e tooltip informativo
- campos: m_materialIconImage, m_buttonCraftingType, m_tooltipTrigger, m_subHoverObject
  - lza: OnPointerEnter - ativa m_subHoverObject ao passar mouse
  - lzb: OnPointerExit - desativa m_subHoverObject ao sair do hover
  - lzc: OnPointerClick - seleciona o material e notifica o sistema de crafting
  - lzd: SetMaterial - configura ícone (m_materialIconImage) e tipo de crafting
  - lze: ShowTooltip - exibe informações do material via m_tooltipTrigger
### TaskbarHero.UI.CubeInventorySlot
Interface UI para exibir e interagir com um slot de inventário contendo um cubo com suporte a encantamento, extração, preview de movimento e efeitos visuais.
- campos: bfzt, _cubeData, m_extractionOptionButton, m_enchantSlotViewer, m_blockingCubeSlotImage, m_slotBackGround, image_ItemIcon, image_GradeBg, button_Click, itemTooltipTrigger, m_rectTransform, m_levelText, m_executeMovePreviewImage, m_movePreviewCanMoveSprite, m_movePreviewCanNotMoveSprite, m_blockingSpriteList
  - SetCubeData: Carrega dados do cubo e atualiza ícone, fundo de gradação e nível
  - ShowMovePreview: Exibe sprite indicando se o cubo pode ou não ser movido
  - PlayWhiteEffect: Reproduz animação de 6 ou 11 frames branca (feedback visual)
  - CheckIfCanMove: Valida se cubo pode ser movido conforme regras e estado
  - OnClickSlot: Dispara callback de clique e mostra opções (extração/encantamento)
### TaskbarHero.UI.CubeMiddleTextSetter
Atualiza e exibe texto com avisos no cubo central da taskbar de UI
- campos: m_text, m_warningColor, bfwr, bfws
  - kyt: Inicializa componentes de Text e warning color do cubo
  - kyu: SetText - atualiza o texto exibido no campo m_text
  - kyv: ShowWarning - aplica m_warningColor e ativa indicadores visuais (bfwr/bfws)
  - kyw: HideWarning ou ResetColor - volta à cor normal, desativa avisos
  - kyx: Update ou RefreshUI - sincroniza estado visual a cada frame
### TaskbarHero.UI.CubeSlotResetButton
Botão da UI que reseta o estado/conteúdo de um slot de item/cubo no inventário ou bar de ações
- campos: m_button
  - kwh: Registra listener do botão ou chama handler de clique
  - kwi: Atualiza estado visual do botão (habilitado/desabilitado)
  - kwj: Executa a lógica de reset (limpa dados do slot)
  - kwk: Notifica sistema de inventário que o slot foi resetado ou broadcasts evento
### TaskbarHero.UI.CubeSlotSettingType
Enum que define tipos de aparência/background para slots de item na UI do inventário
- campos: value__, AllBackGroundOff, DefaultItemSlotBackGround
  - GetHashCode: retorna hash do valor do enum
  - Equals: compara se dois valores de enum são iguais
  - ToString: converte enum para string legível
  - CompareTo: compara dois valores de enum
### TaskbarHero.UI.CubeSynthesisAutoFillButton
Botão de UI que controla síntese automática de cubos, atualizando ícones visuais da síntese principal e secundária.
- campos: m_mainSynthesisIconChanger, m_subSynthesisIconChanger
  - kwl: OnClick — ativa síntese automática ou aplica preset de cubos
  - kwm: UpdateSynthesisIcons — atualiza ícones dos sintetizadores via m_mainSynthesisIconChanger e m_subSynthesisIconChanger
### TaskbarHero.UI.DlcBundleSlot
Gerenciar um slot de UI para exibição e acesso a bundles de DLC, permitindo compra/download via plataforma de distribuição digital.
- campos: m_dlcAppIds, m_bundleStoreUrl
  - Initialize/Setup: carrega m_dlcAppIds e m_bundleStoreUrl, configura o slot com dados do bundle
  - OpenStore/Redirect: abre m_bundleStoreUrl no navegador ou loja nativa para compra do DLC usando os appIds
### TaskbarHero.UI.DlcSlot
Gerenciar e exibir um slot de DLC na interface do usuário
- campos: m_dlcAppId
  - Initialize: Carrega o ID do DLC (m_dlcAppId) e configura o slot na UI
  - RefreshDisplay: Atualiza a visualização do estado do DLC (disponível/instalado/bloqueado)
  - OnSlotClicked: Manipula interação do jogador com o slot
### TaskbarHero.UI.DownloadContentSlotBase
UI base para exibir slots de conteúdo baixável com integração Steam e indicador de compra.
- campos: m_useSteamOverlay, m_purchasedIndicator, bfzj, bfzk, bfzl
  - ler: Inicializar/configurar slot com dados de conteúdo
  - les: Atualizar visual do indicador de compra e estado
  - let: Abrir overlay Steam para compra de conteúdo
  - leu: Limpar/resetar referências de UI
### TaskbarHero.EBlockingReason
Enumeração de razões que bloqueiam ações do jogador na interface (caixa de correio, desconexão Steam, reset de dados, limpeza de inventário, problemas de servidor).
- campos: value__, BecauseMailBox, BecauseSteamDisconnect, BecauseResetData, BecauseRunningFixingInventory, BecauseServerIssuePopUp
  - IsBlocked: verifica se existe alguma razão de bloqueio ativa
  - GetBlockingMessage: retorna mensagem de erro/aviso ao jogador
  - ClearBlockingReason: remove razão específica de bloqueio
  - HasReason: verifica se uma razão específica está presente
  - CanPlayerAct: determina se jogador pode executar ação baseado no bloqueio
### TaskbarHero.UI.ELogWindowType
Enum que define tamanhos/tipos de janelas de log da UI (Small, Medium, Large)
- campos: value__, Small, Medium, Large
  - ToString: retorna nome legível do tipo (Small/Medium/Large)
  - Parse: converte string para valor enum
  - GetValues: lista todos os tipos disponíveis
  - Equals: compara dois tipos de janela
### TaskbarHero.EMainTab
Enum que representa as abas principais da interface do usuário (Taskbar) do sistema de herói
- campos: value__, None, Hero, NewStash, Portal, Cube
  - ToString: converte o valor da aba em string legível
  - GetHashCode: retorna hash do enum para uso em dicionários
  - CompareTo: compara duas abas para ordenação
  - Equals: verifica igualdade entre abas
  - Parse: converte string em valor de aba
### TaskbarHero.EMaxFPS
Enumeração de limites de FPS (60, 45, 30) para configurações de vídeo/performance do jogo.
- campos: value__, FPS_60, FPS_45, FPS_30
  - GetName: retorna string do valor FPS (ex: 'FPS_60')
  - GetValue: retorna int do valor de FPS
  - Parse: converte string para enum EMaxFPS
  - TryParse: tenta converter string para enum com fallback
  - IsValid: valida se valor FPS está suportado
### TaskbarHero.UI.EMovePivotTarget
Enum que define os pontos de pivot/ancoragem para animações de movimento de elementos na UI da taskbar
- campos: value__, Left, Middle, Right, None
  - ToString: retorna nome legível do valor (Left/Middle/Right/None)
  - GetValues: retorna array com todos os valores do enum
  - IsDefined: valida se um inteiro corresponde a um valor válido
  - Parse: converte string para enum value
  - Equals: compara dois valores do enum
### TaskbarHero.EPageTabType
Enumera as posições das abas na taskbar da interface, organizando painéis de controle do herói
- campos: value__, Left, Middle, Right, Bottom, LeftMiddleRight
  - ToString: converte enum para string (Left/Middle/Right/etc)
  - GetPosition: retorna coordenadas baseadas no valor do enum
  - IsValidTab: valida se a posição da aba é válida
  - GetLayoutIndex: calcula índice de layout para renderização
  - CreateTabPanel: instancia painel visual na posição especificada
### TaskbarHero.UI.EPopupType
Enum que define tipos de popups exibidos na UI (saída, resolução, desconexão, anti-cheat, inventário)
- campos: value__, ExitGame, ResolutionChanged, SteamDisconnect, LocalizeExitGame, DetectAntiCheat, TradingStashBlock, ToManyItems, None
  - ToString: retorna nome da enum em string
  - GetHashCode: gera hash do valor do enum
  - CompareTo: compara ordem dos valores
  - GetLocalizedString: retorna texto localizado do popup
  - IsValid: verifica se enum é um tipo de popup válido
### TaskbarHero.UI.RedDot.ERedDotEvent
Enum de eventos que ativam notificações (red dot) na taskbar de UI
- campos: value__, None, NewItem, NewMail, SkillPoint, SkillEquipable, UnlockableSynthesisRecipe, UnlockStatus, UnlockCube, UnlockRune, UnlockedPet, NewBackendReward
  - GetEventType: retorna o tipo de evento para determinar qual ícone de notificação mostrar
  - HasNotification: verifica se o evento requer exibição do ponto vermelho
  - GetPriority: ordena eventos por urgência (NewMail, SkillPoint, Reward antes de cosmético)
  - IsUnlockEvent: agrupa eventos de desbloqueio (Unlock*, New*) para tratamento em massa
  - ClearNotification: limpa o red dot após o jogador ler a notificação
### TaskbarHero.UI.ERudderDirection
Enumeração de direções de controle/navegação (cima, esquerda, direita e variante direita)
- campos: value__, Up, Left, Right, SubRight
  - GetDescription(): tradução legível da direção
  - IsHorizontal(): verifica se Left/Right
  - IsVertical(): verifica se Up
  - Normalize(): retorna direção principal de SubRight
  - GetOpposite(): retorna direção inversa
### TaskbarHero.UI.EScaleType
Enumeração que define os eixos ou dimensões de escala para elementos de UI (X, Y, ou Z)
- campos: value__, X, Y, Z
  - ToString: retorna nome legível do tipo (X, Y, ou Z)
  - GetHashCode: hash do valor enum
  - CompareTo: compara dois tipos de escala
  - Parse: converte string para enum
  - GetName: retorna nome da constante
### TaskbarHero.EShortcutTriggerMode
Enumeração que define o modo de ativação de atalhos na barra de tarefas (ao pressionar ou ao manter).
- campos: value__, OnPress, OnHold
  - OnPress: dispara ação ao pressionar a tecla
  - OnHold: dispara ação ao manter a tecla pressionada
### TaskbarHero.EStashItemTypeFilter
Enum de filtros para o sistema de estoque (Stash), permitindo ao jogador filtrar itens por tipo (arma, armadura, acessório, material) ou compatibilidade de classe.
- campos: value__, ALL, WEAPON, ARMOR, ACCESSORY, MATERIAL, CLASS_KNIGHT, CLASS_ARCHER, CLASS_SORCERER, START, END
  - ToString: retorna nome legível do filtro
  - HasFlag: verifica se tipo específico está no filtro
  - GetItemsForFilter: retorna lista de itens que correspondem ao filtro
  - IsCompatibleWithClass: valida se item/classe passa no filtro
  - GetFilterRange: retorna intervalo START-END para iteração
### TaskbarHero.EToastType
Enum que define posições de alinhamento para notificações Toast (alertas/mensagens temporárias)
- campos: value__, LEFT, CENTER, RIGHT, BOTTOM
### TaskbarHero.UI.ETooltipSourceType
Enum que identifica a origem de um tooltip na interface — de qual container (inventário, gear, stash, trading stash ou Horadric Cube) o item está sendo exibido.
- campos: value__, None, Inventory, Gear, Stash, TradingStash, Cube
  - ToString: retorna nome legível do tipo de origem
  - GetValues: retorna todos os valores possíveis do enum
  - Parse: converte string para ETooltipSourceType
  - IsDefined: valida se um valor existe no enum
  - GetHashCode: retorna identificador numérico do valor
### TaskbarHero.Tutorial.ETutorialType
Enum que define as etapas sequenciais do tutorial onboarding, cobrindo UI, combate, equipamento e crafting.
- campos: value__, WindowDrag, MonsterWave, GetFirstBox, EquipItem, OpenSkillUI, UnlockCube, Synthesis, End
### TaskbarHero.EWindowLayoutPreset
Enum que define posições predefinidas para o layout de janelas/elementos da interface no HUD ou taskbar do jogo.
- campos: value__, None, LeftBottom, CenterBottom, RightBottom, LeftTop, CenterTop, RightTop
  - GetHashCode: retorna hash do valor do enum
  - ToString: retorna nome do preset (ex: 'LeftBottom')
  - CompareTo: compara dois presets
  - HasFlag: verifica se um flag está ativo (se usado como Flags)
  - Parse: converte string em preset válido
### TaskbarHero.EWindowZoomRate
Enum que define os níveis de zoom disponíveis para a janela da UI do jogo (100%, 125%, 150%, 200%, 300%).
- campos: value__, Zoom1, Zoom1_25, Zoom1_5, Zoom2, Zoom3
  - GetZoomMultiplier: retorna o valor numérico do zoom (1.0, 1.25, 1.5, 2.0, 3.0)
  - GetNextZoom: avança para o próximo nível de zoom disponível
  - GetPreviousZoom: volta para o nível anterior de zoom
  - IsMaxZoom: verifica se está no maior zoom (3x)
  - ApplyZoom: aplica o nível de zoom selecionado à janela
### TaskbarHero.EnchantSlotSpriteResource
Gerencia sprites visuais de slots de encantamento, mapeando tipos de grau para suas representações gráficas na UI.
- campos: GradeType, EmptySlotSprite
  - GetSpriteForGrade: retorna o sprite correspondente ao tipo de grau do encantamento
  - GetEmptySlot: retorna o sprite do slot vazio quando sem encantamento
  - Initialize: carrega e cache os sprites de cada grau
  - IsValidGrade: valida se o grau existe neste recurso
  - GetAllSprites: retorna dicionário/lista de todos os sprites por grau
### TaskbarHero.UI.EnchantSlotViewer
Visualizar e gerenciar 3 slots de encantamento (Decoração, Gravação, Inscrição) com seus efeitos visuais na UI
- campos: m_decorationBackGround, m_decorationFillSlot, m_engravingBackGround, m_engravingFillSlot, m_inscriptionBackGround, m_inscriptionFillSlot, m_decorationSlotEffects, m_engravingSlotEffects, m_inscriptionSlotEffects
  - Initialize: Configura os 3 slots com backgrounds, fills e efeitos
  - UpdateDisplay: Atualiza visualização quando encantamento muda
  - PlaySlotEffect: Ativa animação/efeito visual do slot selecionado
  - ApplyEnchantment: Aplica encantamento ao slot específico
  - RefreshUI: Recarrega elementos visuais de todos os 3 slots
### TaskbarHero_client.UI.EntryQueue
Gerencia interface de controle de janela do jogo (sempre no topo, quit, tempo restante) e fila de eventos de entrada.
- campos: m_toggleAlwaysOnTopOn, m_toggleAlwaysOnTopOff, m_nextToggleButton, m_gameQuitButton, m_remainTimeText, bcsy, bcsz
  - OnToggleAlwaysOnTopOn: ativa flag de janela sempre no topo
  - OnToggleAlwaysOnTopOff: desativa flag de janela sempre no topo
  - OnNextToggleButtonClick: avança estado do toggle ou próxima entrada na fila
  - OnGameQuitButtonClick: dispara sinal de encerramento do jogo
  - UpdateRemainTimeText: atualiza display de tempo restante no UI
### TaskbarHero.UI.EquipPanelSkillSlot
Gerenciador de UI para um slot de habilidade em painel de equipamento, exibindo status, cooldown, efeitos visuais e notificações.
- campos: RedDotComponent, CoolTimeSlider, bfzv, m_skillTooltipShowTrigger, m_skillEdgeHoverTrigger, usingEffect, m_glowEffect, m_selectImage, m_emptySlotTooltip
  - SetSkill: Definir dados da habilidade (ícone, ID, cooldown) no slot
  - UpdateCooldown: Atualizar o CoolTimeSlider conforme o tempo de recarga passa
  - ShowSkillTooltip: Ativar tooltip ao passar mouse (PointerHover)
  - PlayUsingEffect: Reproduzir efeito visual (usingEffect, glowEffect) quando a skill é usada
  - UpdateRedDot: Mostrar/esconder RedDotComponent se a skill está disponível ou em cooldown
### TaskbarHero.UI.ExplainLinux
Painel de UI educativo que apresenta explicação sobre Linux com opções de concordância e navegação entre steps
- campos: m_deAgreeButton, m_nextButton, bfyh, bfyi
  - ksz: Awake - Inicializa painel e registra callbacks dos botões
  - ldf: OnAgreeButtonClicked - Processa confirmação do usuário
  - ldg: OnNextButtonClicked - Avança para próximo passo do tutorial
  - Show: Torna painel visível
  - Hide: Oculta painel da tela
### TaskbarHero.UI.ExtraStatDetail
Painel UI que exibe descrição e tier visual de uma estatística extra do jogador.
- campos: m_decriptionText, m_tierImage, bgdm, bgdn
  - lop: Initialize() — configura referências iniciais dos componentes
  - loq: SetDescription(string) — atualiza o texto de descrição na UI
  - lor: SetTierImage(Sprite) — renderiza a imagem de tier/rarity
  - los: Show() — ativa o painel visual
  - lot: UpdateBackground() — alterna estilos bgdm/bgdn conforme tema/estado
### TaskbarHero.UI.ExtractionPreviewTooltip
Tooltip que exibe preview visual de extração com imagem de tier/raridade ajustando layout dinâmico
- campos: m_tierImage, m_layoutElementSetter
  - Initialize: configura componentes Image e LayoutElement no cache
  - SetTierImage: atualiza m_tierImage com sprite da raridade extraída
  - AdjustLayout: chama m_layoutElementSetter para recalcular posicionamento do tooltip
  - Show: ativa o tooltip posicionando-o próximo ao alvo
  - UpdatePreview: atualiza todos os dados visuais antes de exibir
### TaskbarHero.UI.ExtractionPreviewTooltipTrigger
Componente UI que exibe um tooltip de preview com informações de extração ao interagir com um elemento, acionado por gatilho (hover/click).
- campos: bgdo, bgdp
  - lov: Inicializa ou carrega os dados do preview de extração
  - low: Mostra o tooltip ao disparar o gatilho (OnPointerEnter)
  - gcz: Esconde o tooltip ao sair do gatilho (OnPointerExit)
### TaskbarHero.UI.FullIllustrationPanel
Painel UI que exibe ilustração completa de um herói, habilidade ou item com animação, título e descrição interativa.
- campos: m_animator, m_title, m_simpleDescription, button, bfxr, bfxs, bfxt, bfxu, bfxv, bfxw, bfxx, bfxy, bfxz
  - Show: Exibe o painel com transição animada via Animator
  - Hide: Oculta o painel
  - Initialize: Configura componentes Text/Image a partir dos dados
  - OnButtonClick: Callback do botão interativo (confirma/fecha)
  - PlayAnimation: Dispara sequência de animação nos 9 elementos visuais (bfxr-bfxz)
### TaskbarHero.UI.GameScreenBoundsCorrector
Corrige a posição de elementos UI para mantê-los dentro dos limites visíveis da tela.
- campos: m_targetRectTransform, m_correctingWhenOutOfScreen, m_ignoreCorrectingX, m_correctingOnEnable, bfxe, bfxf, bfxg
  - lbc: Update — verifica continuamente se o elemento está fora dos limites
  - lbd: ClampRectTransformToScreen — reposiciona o RectTransform quando necessário
### TaskbarHero.Data.GradeSO
ScriptableObject que armazena dados visuais e de apresentação (cores, sprites, formatação) para diferentes graus/raridades de itens no UI
- campos: eGRADE, color, color_Back, sprite_logBg, gradeBg_itemSlot, IsBold, TagOpen, TagClose
  - GetGradeColor: retorna a cor associada ao grau
  - GetGradeSprite: retorna o sprite de fundo do grau
  - ApplyGradeFormatting: aplica as tags open/close ao texto do item
  - InitializeGradeDisplay: configura todos os elementos visuais
  - IsGradeBold: retorna se o grau deve ser exibido em negrito
### TaskbarHero.Tutorial.GuidingGetBox
Caixa visual interativa de tutorial que destaca e guia o jogador através de elementos da interface
- campos: bflh, bfli, bflj, bflk
  - kfc: Inicializa a caixa de guia com posição e conteúdo
  - kff: Exibe a caixa e executa animação de entrada
  - kfo: Oculta a caixa gradualmente
  - kfp: Anima transição da caixa para novo alvo
  - kfj: Define posição/tamanho ao redor do elemento a destacar
### TaskbarHero.Tutorial.GuidingOpenCubeUI
Componente de UI que guia o jogador durante o tutorial para abrir um cubo, exibindo instruções localizadas em múltiplas etapas.
- campos: bflq, bflr, bfls, bflt, m_secondGuideTextLocalizeKey
  - Show: Exibe a UI de guia e ativa o primeiro texto de instrução
  - UpdateGuideText: Atualiza o texto localizado do guia (m_secondGuideTextLocalizeKey)
  - OnCubeOpened: Callback disparado quando o cubo é aberto, avança a guia
  - AnimateUI: Anima a entrada/saída dos elementos de UI do guia
  - Hide: Oculta a UI de guia ao completar o tutorial
### TaskbarHero.Tutorial.GuidingSynthesis
Sistema de exibição de guias sintetizados que orienta o jogador durante tutoriais com textos localizáveis
- campos: m_secondGuideTextLocalizeKey, bflw, bflx, bfly, bflz, bfma, bfmb
  - Show: exibe o guia na tela
  - Hide: esconde o guia do jogador
  - SetGuideText: define o texto do segundo guia via chave de localização
  - UpdateGuide: atualiza estado/posição dos elementos de guia
  - Initialize: inicializa componentes UI e referências de campos
### TaskbarHero.Tutorial.GuidingWindowMove
Elemento visual de tutorial que guia o jogador destacando e acompanhando uma janela/UI que deve ser movida.
- campos: bfmc, bfmd, bfme
  - Start/Awake: Inicializa referências aos campos (posições/transforms) e ativa o guia
  - Update: Atualiza posição do elemento visual para seguir o alvo sendo guiado
  - SetTarget: Define qual janela/elemento deve ser destacado e guiado
  - PlayAnimation: Anima entrada/destaque visual do guia (fade, escala, etc)
  - OnComplete: Desativa ou limpa o guia quando tutorial termina
### TaskbarHero.HeroArrowSwitchPanel
Painel UI da taskbar que permite navegar entre heróis usando botões de seta anterior/próxima
- campos: m_buttonPrevHero, m_buttonNextHero, bdfc
  - OnPrevHeroClicked: Trata clique no botão de herói anterior e muda seleção
  - OnNextHeroClicked: Trata clique no botão de herói seguinte e muda seleção
  - SelectHero: Seleciona e destaca o herói na taskbar
  - UpdateDisplay: Atualiza visual do painel com dados do herói atual
  - InitializeButtons: Registra listeners nos botões prev/next
### TaskbarHero.UI.HeroHUD
Gerencia exibição de vida do herói, cooltime de habilidades e layout dinâmico de sliders de cooldown no HUD
- campos: m_spriteSlider, m_healthBarSprite, HeadUpCoolTime, TwoInterval, ThreeInterval, bfxh, bfxi, bfxj, bfxk, m_nonOverlapCooltimeSliderPosY, m_overlapCooltimeSliderStartPosX, m_overlapCooltimeSliderIntervalX, bfxl
  - UpdateHealthBar: atualizar preenchimento/sprite da barra de vida
  - UpdateSkillCooldown: atualizar visual de cooltime das 4 habilidades (bfxh-bfxk)
  - PositionCooltimeSliders: posicionar sliders de cooldown (modo overlap vs não-overlap)
  - OnSkillActivated: callback quando habilidade é lançada (inicia cooldown)
  - RefreshCooldownDisplay: sincronizar valores de cooldown com intervalos de tempo
### TaskbarHero.Data.HeroIdleSpriteData
Armazena dados de sprites para animações em estado idle do herói
- campos: Entries
  - GetRandomIdleSprite: Retorna um sprite aleatório da coleção de idle
  - GetIdleSpriteAt: Obtém sprite idle por índice
  - LoadIdleSprites: Carrega a coleção de dados de sprites idle
  - AddIdleSprite: Adiciona novo sprite idle à coleção Entries
  - GetIdleSpriteCount: Retorna número de sprites armazenados
### TaskbarHero.Data.HeroIdleSpriteEntry
Entrada de dados mapeando herói para sua sprite de animação ociosa na taskbar
- campos: HeroKey, IdleSprite
  - GetHeroKey: retorna identificador do herói
  - SetHeroKey: define identificador do herói
  - GetIdleSprite: retorna sprite de inatividade
  - SetIdleSprite: define sprite de inatividade
### TaskbarHero.UI.HeroPortrait
Gerencia exibição visual do retrato do herói na UI, com painel de ilustração completa e animações.
- campos: m_viewTabType, m_fullIllustPanelButton, m_notArrangedColor, m_fullIllustPanel, m_animator, bfya, m_portraitImage, bfyb
  - lco: InitializePortrait — carrega e configura a imagem do herói
  - lcp: OnFullIllustrationButtonClicked — exibe/fecha painel com arte completa
  - lcq: UpdatePortraitState — atualiza cores e estados visuais (arranged/not arranged)
  - lcr: PlayHeroAnimation — dispara animação via m_animator
  - lcs: SetupViewTab — configura qual aba/view está ativa (m_viewTabType)
### TaskbarHero.HeroRightTopButtonPanel
Painel de controle rápido na taskbar com toggles de configuração (sempre no topo, mute) e atalhos para sistemas principais (correio, loja, sair).
- campos: m_toggleAlwaysOnTopOn, m_toggleAlwaysOnTopOff, m_toggleMuteOn, m_toggleMuteOff, m_buttonMailBox, m_buttonShop, m_buttonQuitGame
  - OnAlwaysOnTopToggled: Ativa/desativa janela sempre no topo do desktop
  - OnMuteToggled: Ativa/desativa mute de áudio do jogo
  - OnMailBoxClicked: Abre interface de correio/mensagens
  - OnShopClicked: Abre interface de loja de itens
  - OnQuitGameClicked: Exibe diálogo ou sai do jogo imediatamente
### TaskbarHero.HeroSlot
Componente de UI que gerencia um slot de herói na barra de tarefas com suporte a seleção, hover visual e exibição de estado.
- campos: bdgm, usm, button_Click, image_Icon, go_Hover, go_Check
  - hpm: Initialize — configura slot com dados padrão
  - hpn: SetHeroData — carrega dados do herói nos campos (bdgm, usm)
  - hpo: ShowCheckmark — ativa go_Check para indicar seleção
  - hpp: HideCheckmark — desativa go_Check
  - hpq: PlayHoverEffect — ativa animação em go_Hover ao entrar
### TaskbarHero.HeroSmallSlot
Exibir e gerenciar um slot de herói na barra de tarefas, mostrando ícone, estado de bloqueio e notificações de skill points.
- campos: _num, _equipHeronKey, image_heroIcon, button_action, usm, tooltipTrigger, m_skillPointRedDot, IsLocked, LockedUI, bdgn, bdgo, bdgp, bdgq, bdgr, bdgs
  - hpu: Inicializar componentes visuais (ícone, botão, tooltip)
  - hpv: Carregar dados do herói e atualizar exibição
  - hpw: Handle clique no botão (Select ou detalhes)
  - hpx: Atualizar visual de bloqueio (IsLocked/LockedUI)
  - hpy: Mostrar/ocultar notificação red dot de skill point
### TaskbarHero.UI.HoverSfxPlayer
Toca som de hover (efeito sonoro) quando o mouse passa sobre elementos da UI da barra de tarefas
- campos: m_hoverSoundKey
  - PlayHoverSound: reproduz o áudio armazenado em m_hoverSoundKey
  - OnPointerExit: para o som ou remove listener ao sair do elemento
  - Awake/Start: inicializa a chave de som e o audio source
  - GetHoverSoundKey: getter para m_hoverSoundKey
### TaskbarHero.UI.HoverStageTextPanel
Painel de UI que exibe informações de estágio ao passar o mouse, alternando visuais entre estágio padrão e estágio de boss com posicionamento dinâmico
- campos: m_actBossStageBackSprite, m_defaultStageBackSprite, m_backGrondImage, m_demonIconObject, m_defaultStagePosX, m_actBossStagePosX, bgbk
  - lkp: UpdateStageBackground — troca sprite de fundo entre padrão e boss
  - lkq: RepositionPanel — reposiciona o painel horizontalmente conforme tipo de estágio
### TaskbarHero.HpSlider
Gerencia a visualização da barra de vida (HP) do herói na interface
- campos: Slider, image_Slider
  - gqp: SetHpValue ou UpdateSlider - atualiza o valor da barra de vida
  - gqq: RefreshImage ou SyncVisuals - sincroniza a imagem com o estado atual do HP
### TaskbarHero.InputManager
Gerenciador de entrada que captura cliques de mouse e atalho de teclado para disparar eventos de interação com UI/caixas.
- campos: m_openAllBoxKeyCode, OnOpenAllBoxKeyPressed, OnRightMouseClicked, OnLeftMouseClicked
  - HandleLeftMouseClick: Detecta clique esquerdo e dispara OnLeftMouseClicked
  - HandleRightMouseClick: Detecta clique direito e dispara OnRightMouseClicked
  - CheckOpenAllBoxesKey: Verifica pressionamento de m_openAllBoxKeyCode e dispara OnOpenAllBoxKeyPressed
### TaskbarHero.UI.InventorySlot
Componente UI de slot individual em inventário com suporte a encantamentos, bloqueio manual, notificações e validação de classe de equipamento.
- campos: usm, image_ItemIcon, Go_ManualLock, itemTooltipTrigger, m_enchantSlotViewer, m_equipClassNotMatchingIcon, m_reservedIcon, index, RedDot, bfzz
  - Initialize/SetItem: Configura qual item está no slot e atualiza visual
  - UpdateDisplay/RefreshIcon: Recarrega ícone, encantamento e ícones de estado (lock/reserva/classe)
  - OnPointerEnter/ShowTooltip: Gatilho do itemTooltipTrigger, exibe tooltip do item
  - SetLocked/ToggleLock: Alterna Go_ManualLock (prende item contra venda/drop)
  - ValidateEquipClass: Verifica se classe do item bate com personagem, ativa m_equipClassNotMatchingIcon
### TaskbarHero.ItemTooltip
Exibe tooltip interativo de itens com grau, stats, requisitos e modificadores únicos
- campos: _usm, rectTooltip, img_itemGradeBg, img_itemIcon, text_itemName, m_specialGradeTextName, m_specialGradePanel, m_celestialGradeBackPanel, m_divineGradeBackPanel, m_cosmicGradeBackPanel, text_gold, text_RequireEquipClass, text_gradeAndItemType, text_class, text_level, m_uniqueModStat
  - Show: Ativa o tooltip e posiciona na tela
  - Hide: Desativa o tooltip
  - SetItemData: Carrega dados do item (nome, ícone, grau, stats)
  - UpdateGradePanel: Ativa painel visual correto (special/celestial/divine/cosmic)
  - UpdateStats: Preenche stats base e inerentes
### TaskbarHero.Option.LanguageOption
Gerencia seleção e exibição de opções de idioma na interface de usuário
- campos: _tabGroupLanguage, selectedLanguage
  - Show: Torna visível o painel de seleção de idiomas
  - ksz: Carrega e aplica o idioma salvo anteriormente
  - ktp: Muda o idioma selecionado e persiste a escolha
  - ktq: Inicializa o grupo de abas com idiomas disponíveis
### TaskbarHero.UI.LayoutPresetButton
Botão de UI que permite ao jogador selecionar e aplicar presets de layout de interface
- campos: m_button, m_selectedView, m_preset
  - lkr: ApplyPreset - aplica o preset de layout quando botão é clicado
  - lks: UpdateVisualState - atualiza highlight/seleção visual do botão
  - lkt: OnPresetSelected - callback quando este preset é escolhido, atualiza m_selectedView
### TaskbarHero.Option.LoadingOption
Gerencia delay ou animação de carregamento para opções da interface
- campos: loadingTime
  - Métodos provavelmente relacionados a timing/animação (não posso deduzir sem código)
### TaskbarHero.Log.LogCellView
Célula individual de um painel de log que exibe mensagens (padrão ou sistema) com animação de entrada/saída e resposta ao hover do mouse
- campos: m_isSpacer, m_rectTransform, m_logText, m_defaultMessageColor, m_systemMessageColor, m_textTransform, m_rectMaskTransform, m_scrollSpeed, m_waitTime, bfok, bfol, bfom
  - SetMessage: configura o texto, cor baseado em tipo (padrão vs sistema), e dispara animação
  - AnimateIn: executa fade/slide usando m_scrollSpeed em coroutine com m_waitTime
  - SetAsSystemMessage: aplica m_systemMessageColor e marca como mensagem de sistema
  - OnPointerEnter: ativa highlight, tooltip ou mantém célula visível (bfok/bfol flags)
  - FadeOut: anima desaparecimento após tempo ou quando novo log chega (bfom state)
### TaskbarHero.Data.LogColorSettings
Armazena e gerencia configurações de cores e dados de exibição para o sistema de log/eventos de combate (monstros normais, bosses, tempos, nomes).
- campos: m_normalMonsterOrBox, m_stageBossMonsterOrBox, m_actBossMonsterOrBox, m_stageName, m_actBossStageName, m_timeText, m_decorationColor, m_inscriptionColor, m_engravingColor, m_gold, m_heroName
  - GetNormalMonsterColor: Retorna cor configurada para monstros normais
  - GetStageBossColor: Retorna cor configurada para chefes de stage
  - GetActBossColor: Retorna cor configurada para chefes de ato
  - ApplyColorsToUI: Aplica cores configuradas aos elementos de exibição
  - LoadDefaultSettings: Carrega ou reseta configurações padrão de cores
### TaskbarHero.Log.LogData
Container de dados para armazenar entradas de log com formatação visual (tamanho de célula, cores e estilos).
- campos: m_cellSize, bfon, bfoo, bfop, bfoq, bfor, bfos
  - kha: GetCellSize — retorna o tamanho da célula do grid de log
  - khj: SetCellSize — define o tamanho da célula
  - khk: GetLogColor — retorna cor de um campo de log (um dos 6 bfo*)
  - khb: SetLogFormat — configura formatação (cor/estilo) dos 6 campos
  - khl: Serialize — serializa dados do log para persistência
### TaskbarHero.Log.LogManager
Gerencia exibição de logs/histórico em painel UI do jogo com scroll
- campos: bfot, bfou, bfov, bfow, m_logScroll, bfox
  - khn: AddLog — adiciona mensagem ao histórico
  - kho: ClearLogs — limpa o buffer de logs
  - khp: RefreshDisplay — atualiza renderização da UI
  - khq: OnScroll — callback quando scroll muda posição
### TaskbarHero.Log.LogScroll
Componente de UI que exibe um log scrollável e filtrável de eventos/mensagens com reciclagem de células
- campos: m_spacerSizeCorrectionValue, m_logFilter, m_scroller, m_logCellSize, m_cellView, m_spacerCellViewPrefab, m_scrollRect, OnLogDataChanged, bfoy, bfoz, bfpa, bfpb, bfpc, bfpd
  - OnScrollValueChanged: Detecta mudança no scroll e atualiza range de células visíveis
  - UpdateCellsVisibility: Mostra/esconde células conforme viewport e posição de scroll
  - ApplyLogFilter: Refiltra logs e reconstrói lista visual
  - RecycleCell: Reutiliza célula de log com novo dado (pattern de scroll virtualization)
  - OnLogDataChanged: Callback que atualiza display quando dados de log são adicionados
### TaskbarHero.Log.LogSetting
Painel de configurações para o sistema de logging do jogo, permitindo ativar/desativar diferentes tipos de logs e exibir timestamps.
- campos: logToggleOptions, m_showLogCreatedTime, m_resetOptionButton, bfpg, bfph, bfpi
  - OnEnable: Inicializa toggles e carrega configurações salvas
  - ResetOptions: Reseta opções de log aos valores padrão (invoca m_resetOptionButton)
  - OnLogToggleChanged: Callback para mudança de estado dos toggles (logToggleOptions)
  - SaveSettings: Persiste as configurações de logging
  - LoadSettings: Carrega configurações salvas anteriormente
### TaskbarHero.Log.LogToggleButton
Componente UI que alterna a visibilidade/exibição de um tipo de log no HUD, com suporte a localização multi-idioma.
- campos: Type, ToggleButton, OptionLocalize
  - Initialize: configura o tipo de log e callbacks do toggle
  - OnToggleChanged: callback quando o botão é clicado para mostrar/esconder log
  - UpdateLocalization: atualiza o texto do botão para o idioma atual
  - SetLogType: define qual tipo de log este toggle controla
  - IsLogVisible: retorna se o log deste tipo está visível na tela
### TaskbarHero.UI.LogTopBottomChanger
Gerencia dinamicamente a posição (topo/fundo) e tamanho dos painéis de log e utilitários, ajustando offsets e altura conforme o estado da UI.
- campos: m_logRectTransform, m_logScrollRectTransform, m_topPanelRectTransform, m_optionPanelTransform, m_logUtilPanel, m_logUtilPanelBackPanel, m_utilPanelHeight, m_topPanelOffsetCorrectionValue, bgic, bgid, bgie, bgif, bgig
  - OnEnable: inicializa cache das referências RectTransform dos painéis
  - SetLogPositionTop: repositiona log para topo ajustando m_topPanelOffsetCorrectionValue
  - SetLogPositionBottom: repositiona log para fundo com offset negativo
  - UpdateUtilPanelHeight: recalcula altura de m_logUtilPanel e aplica aos bg* (provavelmente Images)
  - RecalculatePanelLayout: aplica nova geometria a m_logRectTransform, m_logScrollRectTransform e m_topPanelRectTransform
### TaskbarHero.Manager.MailBoxPoolManager
Gerencia pool de objetos visuais (slots/abas) da caixa de correio com sincronização de backend
- campos: mailSlotPrefab, mailParent, m_backendPostParent, bfue, bfuf
  - GetOrCreateMailSlot: obtém ou instancia novo slot de correio do pool
  - ReturnMailSlotToPool: devolve slot usado ao pool para reutilização
  - SyncMailBoxWithBackend: sincroniza estado visual com dados do servidor
  - ClearMailBoxPool: limpa todos os slots ativos do pool
  - UpdateMailSlotUI: atualiza visuais de um slot específico com dados
### TaskbarHero.MailSlot
Componente de UI que exibe um item individual no correio/inbox, mostrando ícone, nome, tempo de recebimento e botão de coleta.
- campos: usm, image_ItemIcon, image_ItemGradeBg, text_ItemName, text_ReceivedTime, button_Claim, bdgv, bdgw
  - SetupMailItem: Inicializa o slot com dados do item (ícone, nome, badges, timestamp)
  - OnClaimButtonClicked: Handler do botão de reivindicar/coletar o item
  - RefreshDisplay: Atualiza visual dos componentes (ícone, raridade, badges)
  - UpdateReceivedTime: Atualiza o texto de tempo decorrido desde recebimento
### TaskbarHero.UI.MainRecipeComboBoxButton
Componente de seleção de receitas de síntese/crafting na barra de tarefas do herói, permitindo escolher receita principal, sub-receitas e tipo de item com indicadores visuais.
- campos: m_selectRecipeIcon, m_subRecipeComboBoxButton, m_synthesisItemTypeComboBoxButton, m_unlockEffect, m_redDot
  - Initialize: Configurar referências aos componentes filhos (ícone, combo boxes, efeitos)
  - SelectRecipe: Processar seleção de receita principal e atualizar ícone exibido
  - UpdateSubRecipeOptions: Popular opções de sub-receitas baseado na receita selecionada
  - ShowUnlockEffect: Reproduzir animação m_unlockEffect quando receita é desbloqueada
  - UpdateNotificationBadge: Mostrar/esconder m_redDot quando há novo item ou atividade
### TaskbarHero.UI.RedDot.MainRecipeComboBoxButtonRedDot
Gerencia o indicador visual (ponto vermelho) no botão de seleção de receita principal da taskbar.
  - Show: Exibe o ponto vermelho quando há receita principal nova
  - Hide: Esconde o indicador após a receita ser visualizada
  - UpdateRedDot: Verifica e atualiza o estado do indicador baseado na receita ativa
  - OnRecipeChanged: Callback disparado quando a receita principal muda
  - IsActive: Retorna se o indicador deve estar visível
### TaskbarHero.UI.MainToggleButton
Botão toggle para exibir/ocultar a Hero UI com feedback visual de hover
- campos: Ui_Hero, SpriteWhenShowHeroUIExitPointer, SpriteWhenShowHeroUIOnPointer, bfwg, bfwh, bfwi, bfwj, bfwk
  - kxc: Alterna visibilidade da Hero UI (ToggleHeroUI)
  - gay: Atualiza sprite para estado OnPointer (ShowOnPointerSprite)
  - kxd: Restaura sprite para estado ExitPointer (ShowExitPointerSprite)
### TaskbarHero.Tutorial.MainUserInterfaceTutorialStep
Gerenciar um passo individual do tutorial que guia o jogador através dos elementos da interface principal do jogo
- campos: m_guideTextLocalizeKey, bfmf, bfmg, bfmh, bfmi, bfmj, bfmk
  - Initialize: Configura o tutorial com texto localizado e ativa/desativa elementos UI
  - Execute: Exibe o passo do tutorial, destacando componentes (bfmf-bfmk são provavelmente UI elements)
  - Validate: Verifica se o jogador completou a ação esperada neste passo
  - Cleanup: Remove destaques, restaura estado da UI e transiciona para próximo passo
### TaskbarHero.Option.MonitorOption
Painel de UI para selecionar qual monitor exibirá o jogo em resolução completa.
- campos: _transformMonitorSelectButtonParent, _prefabMonitorSelectButton, bfuz, _selectedMonitorIndex, text_next
  - Show: Ativa e exibe o painel de seleção de monitores
  - ksz: Popula dinamicamente os botões de seleção baseado nos monitores detectados
  - ktw: Handler de clique que atualiza _selectedMonitorIndex e marca a seleção
  - ktj: Navega para o próximo monitor ou confirma e fecha o painel
### TaskbarHero.UI.MouseRightClickDirectMoveRudder
Renderiza indicador visual de movimento (rudder/leme) quando jogador clica-direito para mover direto no mapa
- campos: m_rectTransform, m_rudderParts
  - lcy: Inicializa componentes do rudder (rectTransform, partes visuais)
  - lcz: Atualiza posição/rotação do rudder conforme movimento do mouse
  - OnRightClick: Detecta clique-direito e ativa rudder
  - UpdateRudderDirection: Calcula direção do vetor mouse→personagem
  - DeactivateRudder: Esconde rudder após movimento confirmado
### TaskbarHero.MoveResult
Encapsula o resultado de uma ação de mover/arrastar um item na barra de atalhos (taskbar), rastreando qual item foi afetado.
- campos: MoveItemUniqueId
  - hzm: bool IsSuccessful() — valida se movimento foi válido/completado
  - hzn: string GetErrorMessage() — retorna motivo da falha (slot ocupado, item inválido, etc)
  - ExecuteMove(int targetIndex) — aplica o movimento para o slot destino
  - ValidateItemPlacement() — verifica regras de colocação (peso, tipo, restrições)
  - GetPreviousPosition() — retorna slot anterior para undo/rollback
### TaskbarHero.UI.Rune.NodeConnectionLine
Renderiza a linha de conexão visual entre dois nós de um sistema de runas, com cores dinâmicas de ativo/inativo
- campos: m_lineImage, m_lineWidth, m_nodeInActiveColor, m_nodeActiveColor, m_parentRuneNodeKey, m_myRuneNodeKey
  - mac: Atualiza a cor da linha baseado no estado ativo/inativo do nó
  - mad: Inicializa a conexão entre o nó pai e o nó filho da runa
### TaskbarHero.UI.NoticeBackendReward
Interface popup para o jogador reivindicar recompensas do servidor com gerenciamento de cooldown, expiração e slots.
- campos: m_deleteSlots, m_claimRewardButton, m_closePopUpButton, m_rewardDescription, m_rewardTitle, m_rewardExpirationText, m_coolTimeButton, m_coolTimeText, bgdu, bgdv, bgdw, bgdx, bgdy
  - lpe: Inicializa popup e registra listeners de botões (OnEnable já visível)
  - lpf: ConfiguraReward — popula m_rewardTitle, m_rewardDescription, m_rewardExpirationText
  - lpg: AtualizaCooldown — tick do timer em m_coolTimeText/m_coolTimeButton, controla disponibilidade de claim
  - lpi: ReivindicaRecompensa — serializa request ao backend quando m_claimRewardButton clicado
  - lpj: FechaPopup — invoca m_closePopUpButton, limpa estado (provavelmente chamado por lpj ou OnDisable)
### TaskbarHero.Option.NoticeOption
Sistema de notificações/avisos sequenciados da UI que exibe mensagens com suporte a múltiplos textos encadeados.
- campos: text_next
  - Show: exibe a notificação na tela
  - Next: avança para o texto seguinte armazenado em text_next
  - SetText: define o conteúdo da mensagem
  - Close: remove a notificação da UI
  - Initialize: configura a opção com parâmetros iniciais
### TaskbarHero.UI.OpenBoxStats
Gerencia temporizadores e lógica de abertura automática de caixas/baús com tempos configuráveis (máximo e redução por upgrades).
- campos: AutoOpenBoxMaxTime, AutoOpenBoxReduceTime
  - Initialize/Start: Configura timers iniciais com AutoOpenBoxMaxTime
  - UpdateAutoOpenTimer: Decrementa contador e verifica se atingiu tempo de abertura
  - GetEffectiveOpenTime: Retorna MaxTime - ReduceTime (tempo final ajustado)
  - ApplyReduceTimeUpgrade: Modifica AutoOpenBoxReduceTime (progressão)
  - TryAutoOpenBox: Dispara abertura se timer expirou, reseta contador
### TaskbarHero.UI.OptimizedSpriteAnimator
Anima sprites sequencialmente em UI com controle de velocidade e callback de conclusão
- campos: m_image, m_clips, m_stopOriginalImage, m_fps, m_unscaledTime, bgbb, bgbc, bgbd, bgbe, bgbf, OnAnimationComplete
  - Play: Inicia reprodução de um clip de animação específico
  - lkh: Para a animação em progresso e restaura imagem original
  - lki: Avança frame-by-frame ou aplica pausa
### TaskbarHero.UI.PetAbilityDescriptionSlot
Slot de UI que exibe nome, ícone e descrição de uma habilidade de pet (combat ability descriptor panel).
- campos: GameObject, Text
  - SetAbility: configura qual habilidade do pet será exibida no slot
  - DisplayDescription: atualiza o campo Text com a descrição da habilidade
  - Show: ativa o GameObject para visualizar o slot na tela
  - Hide: desativa o GameObject quando a habilidade não está selecionada
  - UpdateVisuals: sincroniza ícone, nome e efeitos especiais do slot
### TaskbarHero.UI.PetSlot
Gerencia slot visual de um pet com interação (hover, seleção, deleção, notificações)
- campos: m_hoverObject, m_selectedObject, m_arrangedObject, m_iconImage, m_checkImageObject, m_deleteImageObject, m_redDotObject, m_button, bfym, bfyn, bfyo, bfyp, bfyq, bfyr
  - SetPetData: atualiza ícone e dados visuais do pet
  - OnPointerEnter: mostra m_hoverObject ao passar mouse
  - OnPointerExit: esconde m_hoverObject ao sair mouse
  - OnClick: seleciona o pet (ativa m_selectedObject e callback)
  - UpdateNotification: mostra/esconde m_redDotObject de novo item/atualização
### TaskbarHero.UI.PetSlotParent
Gerenciador UI que instancia e organiza slots de seleção/exibição de pets do jogador na taskbar
- campos: bfys, m_slotPrefab
  - ldr: Instantia os slots filhos usando m_slotPrefab e popula bfys com referências
  - bfys (campo): Array/List de PetSlot instanciados armazenando os componentes filhos
### TaskbarHero.UI.Portal
Portal interativo na UI que permite ao jogador sair/navegar para outro local com animação e tooltip localizado
- campos: m_exitLocalizeKey, m_exitTooltipLocalizeKey, PortalAnimator, text_Portal, button_Portal, m_tooltipTrigger, bfyv, bfyw, bfyx, bfyy
  - Initialize: Configura referências ao animator, botão e localização das strings exit/tooltip
  - OnPortalClick: Handler do botão que ativa a animação e executa a transição
  - UpdateLocalization: Atualiza textos e tooltip com chaves localizadas (m_exitLocalizeKey)
  - PlayPortalAnimation: Dispara sequência de animação via PortalAnimator
  - SetTooltipTrigger: Configura trigger para exibir tooltip ao hover
### TaskbarHero.UI.PortalTooltipTrigger
Exibe tooltips (dicas de interface) quando o jogador interage com portais na barra de tarefas.
  - OnPointerEnter: Mostra o tooltip quando mouse passa sobre o portal
  - OnPointerExit: Esconde o tooltip quando mouse sai da área
  - SetPortalData: Atualiza o conteúdo da dica com informações do portal (nome, nível, descrição)
  - ShowTooltip: Torna o tooltip visível na tela
  - HideTooltip: Remove o tooltip da exibição
### TaskbarHero.UI.RandomSpriteSetter
Seleciona e aplica um sprite aleatório de uma lista quando o objeto UI é ativado.
- campos: m_spriteList, bfyz
  - SetRandomSprite: escolhe um índice aleatório e atribui o sprite correspondente
  - GetRandomSprite: retorna um sprite aleatório da m_spriteList sem aplicar
  - Start: inicializa referência ao SpriteRenderer ou Image component
  - OnEnable: dispara a seleção aleatória do sprite ao ativar
  - bfyz: provavelmente referência ao componente visual (SpriteRenderer ou Image) que recebe o sprite
### TaskbarHero.UI.RecipeBackGroundSprite
Mapeia tipos de receita para sprites de fundo visualmente distintos na interface de crafting/receitas.
- campos: RecipeType, BackGroundSprite
  - GetBackgroundSprite: Retorna o sprite de fundo correspondente ao RecipeType atual
  - SetRecipeType: Define qual tipo de receita exibir e atualiza o sprite
  - UpdateDisplay: Recarrega visualmente o fundo baseado no tipo de receita
  - InitializeMapping: Configura o dicionário tipo→sprite na inicialização
  - GetRecipeType: Retorna o tipo de receita ativo
### TaskbarHero.UI.RedDot.RecipeComboBoxButtonRedDot
Exibe notificação visual (ponto vermelho) quando há novas receitas disponíveis no botão de seleção de receitas da barra de tarefas.
- campos: m_comboBoxButton, m_spriteRenderer
  - OnEnable: Inicializa referências ao comboBoxButton e spriteRenderer
  - CheckNewRecipes: Verifica se há receitas desbloqueadas não visualizadas
  - ShowRedDot: Ativa sprite renderer do ponto de notificação
  - HideRedDot: Desativa sprite renderer da notificação
  - OnRecipeUnlock: Callback disparado ao desbloquear nova receita
### TaskbarHero.UI.RecipeDetailText
Exibe e formata o texto de detalhes de uma receita de crafting/combinação na UI
- campos: m_text, m_layoutElement
  - kzq: DisplayRecipe — atualiza o TextMesh com informações da receita e ajusta o LayoutElement
  - SetRecipeData — define dados da receita a exibir
  - AdjustLayoutHeight — recalcula altura do layout baseado no tamanho do texto
### TaskbarHero.UI.RectTransformTextWidthHeightSetter
Ajusta automaticamente largura e/ou altura de um RectTransform de texto UI baseado em dimensões do elemento pai e restrições mínimas.
- campos: m_adjustWidth, m_adjustHeight, m_parentElementDefaultHeight, m_parentElementDefaultWidth, m_minHeight, m_parentLayoutElement, m_parentLayoutRectTransform, bfzb, bfzc
  - lej: CalculateTextDimensions — computa largura/altura do texto usando LayoutElement pai e m_minHeight
  - lek: ApplyAdjustments — aplica os valores calculados ao RectTransform do elemento, respeitando m_adjustWidth/m_adjustHeight
  - OnEnable: Cache referências do LayoutElement pai e inicializa dimensões padrão
### TaskbarHero.UI.Rune.RuneNodeVisual
Renderiza visualmente um nó de runa com diferentes estados de nível e indicador de modo preview
- campos: m_previewModeAlpha, m_canvasGroup, m_runeIconImage, m_zeroLevelEdge, m_normalLevelEdge, m_maxLevelEdge, m_dimOverlay
  - UpdateLevelVisual: muda a borda exibida (m_zeroLevelEdge/m_normalLevelEdge/m_maxLevelEdge) conforme o nível da runa
  - SetPreviewMode: ativa/desativa o overlay escuro (m_dimOverlay) e ajusta alpha (m_previewModeAlpha) para indicar seleção/preview
### TaskbarHero.UI.Rune.RunePage
Renderizar árvore interativa de runes/habilidades com nós conectados e atualizar custos em ouro.
- campos: m_startNodes, m_scrollViewRectTransform, m_goldText, m_listRuneNode, m_linePrefab, m_lineParent, m_lineWidth, m_connectionLine, bgir, bgis
  - Initialize: setup inicial da página, carrega startNodes e configura scroll
  - UpdateRuneTree: atualiza posições de todos os RuneNodes na árvore
  - DrawConnections: desenha/atualiza linhas de conexão entre nós usando linePrefab
  - OnNodeClicked: callback quando jogador clica em um nó da árvore
  - UpdateGoldCost: atualiza goldText com custo total de ativação selecionada
### TaskbarHero.UI.Rune.RunePageZoom
Gerencia zoom interativo de uma página de visualização de runas, permitindo scroll e ampliação/redução dentro de limites definidos
- campos: m_scrollRect, m_content, m_minZoom, m_maxZoom, m_zoomSpeed, bgit
  - ApplyZoom: aplica fator de escala ao conteúdo via transform.scale
  - ClampZoom: limita valor de zoom entre m_minZoom e m_maxZoom
  - AnimateZoom: transição suave do zoom atual até target via Coroutine/Time.deltaTime
  - ResetZoom: volta zoom a 1.0 (tamanho original)
  - UpdateContentScale: recalcula posição/tamanho do m_content conforme zoom muda
### TaskbarHero.UI.Rune.RuneStatInfo
Exibe e gerencia informações visuais de uma Rune (stat, valor, ícone) na UI do Taskbar
- campos: m_text, bgiu, bgiv
  - SetupRune: inicializa a runa com seu valor/tipo
  - UpdateStatText: atualiza m_text com novo valor de stat
  - SetBackgroundImage: configura bgiu (background image)
  - SetBackgroundColor: configura bgiv (background color/tint)
### TaskbarHero.UI.Rune.RuneTooltipTrigger
Exibe tooltip com informações da runa ao passar o mouse sobre ela
- campos: m_runeNode, m_hoverIamge
  - mbq: Mostra ou esconde a imagem de tooltip conforme o estado do hover
  - OnPointerEnter: Dispara exibição do tooltip quando mouse entra
  - OnPointerExit: Esconde o tooltip quando mouse sai
  - OnDisable: Limpa estado visual e oculta tooltip ao desativar
  - GetRuneData: Recupera informações da runa para preencher o tooltip
### TaskbarHero.SDModelPreview
Exibe prévia visual de heróis (Knight, Ranger, Sorcerer, Priest, Hunter, Slayer) com animações sprite correspondentes em UI de seleção ou composição de equipe.
- campos: m_notArrangedColor, m_spriteAnimationKnight, m_spriteAnimationRanger, m_spriteAnimationSorcerer, m_spriteAnimationPriest, m_spriteAnimationHunter, m_spriteAnimationSlayer, m_previewObject, bdfd, bdfe, bdff
  - SetHeroClass: define qual classe de herói será mostrado na prévia
  - UpdateAnimation: atualiza a animação sprite do herói atual
  - ApplyNotArrangedState: aplica cor visual indicando herói não selecionado/não incluído
  - ShowPreview: torna o objeto de prévia visível na tela
  - GetSpriteAnimationForClass: retorna a animação sprite correspondente à classe do herói
### TaskbarHero.UI.ScreenBoundsCorrector
Corrige a posição de um elemento UI (RectTransform) para mantê-lo dentro dos limites visíveis da tela, com suporte para múltiplos monitores.
- campos: m_targetRectTransform, m_correctingWhenHideMonitor, m_ignoreCorrectingX, bfzd, bfze, bfzf
  - lel: Aplica a correção dos limites do RectTransform baseado na resolução da tela
  - lem: Obtém e atualiza os limites seguros da tela (viewport) considerando monitor oculto
### TaskbarHero.UI.ScrollRectEdgeIndicator
Gerencia indicadores visuais (up/down) que aparecem nas extremidades de um ScrollRect para mostrar quando há conteúdo adicional para rolar.
- campos: m_upObject, m_downObject, m_edgeThreshold, bfzg
  - len: Responde ao evento OnScroll do ScrollRect, atualiza posição de scroll armazenada
  - leo: Calcula se o indicador superior deve ser visível baseado na posição e threshold
  - lep: Calcula se o indicador inferior deve ser visível baseado na posição e threshold
### TaskbarHero.Option.SelectHeroOption
Tela de seleção e visualização de heróis com efeitos visuais, bloqueio de personagens e navegação entre opções.
- campos: selectUIIdx, heroUIList, text_HeroName, text_HeroDesc, fireAnimaition, fireLightAnimaition, toggleButton_Next, lockedImage, text_chooseCharacter, summonEffect, effectAnimation, animator, m_parentRT, duration, intensity, bfve
  - Show: exibe a UI de seleção de herói
  - ksz/Hide: fecha/oculta a tela de seleção
  - kti/UpdateHeroInfo: atualiza nome, descrição e visual do herói selecionado
  - kue/OnHeroSelected: processa seleção de herói (ativa efeitos e animações)
  - kuh/SetLockedState: aplica visual de bloqueado (imagem overlay, cor desativada)
### TaskbarHero.SenderType
Enum que categoriza tipos de emissores de notificações/mensagens exibidas na taskbar da UI
- campos: value__, General, News, RegularReward, UpdateReward, IncidentCompensation
  - ToString: retorna nome do tipo de sender como string
  - GetValue: retorna o valor inteiro associado ao enum
  - Parse: converte string para o tipo SenderType correspondente
  - CompareTo: compara dois tipos de sender pela ordem
  - HasFlag: verifica se um tipo contém um flag específico
### TaskbarHero.Animation.SetBoolOnEnter
Define um parâmetro booleano do Animator quando um estado de animação é ativado
- campos: m_parameterName, m_willSetValue
  - OnStateEnter: Callback acionado ao entrar no estado; chama animator.SetBool(m_parameterName, m_willSetValue) para atualizar estado de animação
### TaskbarHero.Animation.SetBoolOnExit
Ao sair de um estado de animação, define um parâmetro booleano do Animator para controlar transições
- campos: m_parameterName, m_willSetValue
  - OnStateExit: Invoca Animator.SetBool(m_parameterName, m_willSetValue) para sinalizar saída do estado
### TaskbarHero.Option.SettingOption
Tela de configurações (Settings) do jogo que permite ajustar volume, zoom, comportamento da interface e opções de inicialização.
- campos: text_next, slider_masterVolume, toggle_alwaysOnTop, toggle_autoStart, dropdown_zoom, bfvj
  - Show: Exibir/abrir a tela de configurações
  - ksz: Carregar (Load) configurações salvas do disco
  - kup: Atualizar (Update) valores dos controles na UI
  - kuq: Aplicar (Apply) configurações e fechar tela
  - kur: Resetar (Reset) configurações para padrão do jogo
### TaskbarHero.SettingSaveData
Classe de persistência para salvar todas as preferências e configurações do jogador (UI, gameplay, sistema).
- campos: language, isAlwaysOnTop, maxFps, windowZoomRate, windowLayoutPreset, FixingLog, LogFilter, LogShowTime, isAutoStart, StageRepeatStageFailed, RepeatActBossDuringConsumeAllStone, AskWarningAlchemy, AskWarningTradingStash, BoxAlphaValue, BoxShowWhenHover, ShowQuickMenu
  - Save: serializar para disco os campos em JSON/binário
  - Load: desserializar do arquivo de configuração
  - Reset: restaurar valores padrão de fábrica
  - Validate: verificar integridade dos dados salvos
  - GetWindowState: retornar geometria da janela (zoom + preset)
### TaskbarHero.ShortcutBinding
Armazena e gerencia um atalho de teclado com modificadores (Ctrl, Shift, Alt) para ativar ações na UI
- campos: m_mainKey, m_requiresCtrl, m_requiresShift, m_requiresAlt
  - IsPressed: verifica se a combinação de tecla+modificadores foi pressionada agora
  - SetBinding: configura nova tecla principal e seus modificadores
  - MatchesInput: valida se o input atual corresponde exatamente a este binding
  - GetDisplayName: retorna string legível do atalho (ex: 'Ctrl+Shift+S')
  - Reset: limpa o binding, deixando sem atalho atribuído
### TaskbarHero.UI.SimpleCloseButton
Botão de fechar UI que destruir alvo referenciado ao ser clicado/destruído, com suporte a múltiplos targets
- campos: m_closeTargets, bgia
  - lzt: Registra listeners de clique ou evento de destruição para invocar Close nos targets
  - Close: Destrói ou desativa os GameObjects em m_closeTargets
  - Awake: Inicializa referências e setup do componente
  - OnDestroy: Limpeza de listeners e callbacks registrados
### TaskbarHero.UI.SimpleHoverImageColorChanger
Componente de UI que muda a cor de uma imagem em resposta ao hover do mouse, fornecendo feedback visual interativo.
- campos: m_image, m_hoverColor, m_defaultColor
  - OnEnable: Restaura o estado inicial e aplica a cor padrão à imagem
  - OnPointerEnter: Muda a cor da imagem para m_hoverColor
  - OnPointerExit: Restaura a cor da imagem para m_defaultColor
### TaskbarHero.UI.SimpleImageChanger
Componente UI que troca sprites de um elemento de imagem ao interagir com o mouse (hover/normal/desativado).
- campos: m_image, m_controlImageObjectActive, m_hoverSprite, m_normalSprite, m_deActiveSprite, bfzn
  - lev: SetSprite — define o sprite da imagem para o valor passado
  - lew: RefreshImageState — atualiza o sprite baseado no estado ativo/hover/desativado
  - OnPointerEnter: chama lev/lew com m_hoverSprite
  - OnPointerExit: chama lev/lew com m_normalSprite
  - OnEnable: inicializa com m_normalSprite ou carrega bfzn (estado persistido)
### TaskbarHero.UI.SimplePointerHoverObjectToggler
Componente UI que ativa/desativa objetos alvo quando o mouse passa por cima (hover).
- campos: m_targetObjects, bfzo
  - lex: ToggleObjects() — ativa/desativa m_targetObjects baseado no estado bfzo
  - bfzo: booleano que armazena se está em hover ou estado ativo
### TaskbarHero.UI.SimpleTextColorChanger
Componente UI que muda a cor de um texto quando o mouse passa por cima (hover effect).
- campos: m_text, m_hoverColor, m_defaultColor
  - SetTextColor: aplica uma cor específica ao campo m_text
  - UpdateHoverState: wrapper que chama SetTextColor com m_hoverColor
  - ResetToDefaultColor: restaura m_defaultColor (chamado em OnEnable e OnPointerExit)
  - GetTextComponent: lazy-load do m_text via GetComponent se nulo
  - ValidateFields: debug/editor method que verifica se m_text está atribuído
### TaskbarHero.UI.SimpleToggleButton
Botão UI que alterna ativação/desativação de múltiplos GameObjects alvo com estado visual
- campos: m_toggleTargets, m_disableWhenAwake, bgib
  - lzu: Executa toggle nos m_toggleTargets (ativa/desativa objetos)
  - OnClick: Handler registrado no Button component que chama lzu
  - UpdateVisuals: Atualiza cor/estado visual de bgib baseado em IsOn
  - ApplyInitialState: Desabilita m_toggleTargets se m_disableWhenAwake for true
### TaskbarHero.SlotActionContext
Captura contexto de interação do usuário com slots (skills/itens) na UI de ação rápida, incluindo cliques, modificadores de teclado e hover.
- campos: SourceType, Item, SlotType, SlotIndex, IsRightClick, IsShift, IsCtrl, IsAlt, IsJustHover
  - GetSourceType: retorna tipo de origem do slot (inventário/equip/skillbar)
  - IsModified: verifica se há algum modificador (Shift/Ctrl/Alt pressionado)
  - IsContextMenu: retorna true se clique direito (menu de contexto)
  - GetActionType: deduz tipo de ação (move/use/drop) baseado em modificadores
  - CanExecute: valida se contexto permite executar ação no slot
### TaskbarHero.SlotActionResult
Estrutura de resultado que encapsula a ação executada em um slot de habilidade/item da barra de tarefas, incluindo validação de tipo de destino e permissão de troca de abas.
- campos: DestinationSlotType, Action, AllowTabSwitch
  - IsValidDestination(): verifica se o tipo de slot de destino é válido para a ação
  - CanExecuteAction(): valida se a ação pode ser executada no slot
  - GetActionEffect(): retorna o efeito/modificador da ação executada
  - ShouldSwitchTab(): determina se a UI deve mudar de aba após a ação
  - SerializeResult(): serializa o resultado para sincronização ou persistência
### TaskbarHero.SoundClipResource
Mapeia uma chave de som para um AudioClip do Unity, servindo como container de recurso de áudio para o sistema de som do jogo.
- campos: SoundKey, Clip
  - GetClip: retorna o AudioClip armazenado para reprodução
  - Play: toca o clip de áudio no contexto do jogo
  - SetClip: atribui um novo AudioClip ao recurso
  - GetSoundKey: retorna a chave identificadora do som
  - Initialize: carrega e inicializa o clip de áudio a partir de um recurso
### TaskbarHero.UI.SpriteSlider
Componente UI que renderiza uma barra de progresso visual usando sprites com suporte a escala e correção de sobreposição.
- campos: fillBar, m_scaleInverseOperation, m_overlapCorrectionValue, ScaleType, FullScale, bgbg, bgbh, bgbi, bgbj
  - SetValue(float): Atualiza o valor/progresso da barra de preenchimento
  - UpdateDisplay(): Recalcula a posição e escala visual do fillBar
  - OnEnable(): Inicializa o slider quando o componente é ativado
  - OnDisable(): Limpa referências e estado ao desabilitar
  - LateUpdate(): Sincroniza mudanças de escala após updates do frame
### TaskbarHero.StageNode
Componente de UI que renderiza um nó de estágio selecionável em menu, com estados (trancado/disponível) e interação hover.
- campos: m_lockIcon, m_hoverIcon, m_occupiedIcon, m_stageText, button_Enter, m_stageTextStyle, m_occupiedTextStyle, bdhe, bdhf
  - hrj: OnButtonClick — handler de clique no button_Enter para entrar no estágio
  - hrk: RefreshVisuals — atualiza ícones (lock/hover/occupied) e estilos de texto conforme estado
  - hrl: SetLocked — marca estágio como trancado, exibe m_lockIcon
  - hrm: SetOccupied — marca estágio como disponível, exibe m_occupiedIcon e m_occupiedTextStyle
  - hrn: UpdateDisplay — sincroniza todos os visuais (ícone, texto, estilo) com estado interno (bdhe/bdhf)
### TaskbarHero.UI.StagePortal
Portal UI para seleção/entrada de estágios mostrando custo em Soul Stones com feedback visual animado
- campos: m_soulStoneTextParent, text_Soulstone, m_arrowObject, bgcg, bgch
  - lmr: Inicializar portal com dados do estágio (custo, nível, etc)
  - lms: Atualizar e exibir quantidade de Soul Stones necessários no UI
  - lmt: Animar entrada/aparecer do portal (fade/scale)
  - lee: Controlar visibilidade/animação da seta de indicação
  - lmu: Animar/transicionar fundo com canvas groups (fade entre bgcg e bgch)
### TaskbarHero.UI.StagePortalRepeatSetter
Gerencia UI de repetição de estágios no portal, alternando visuais e textos baseado no tipo de estágio (boss ativo, padrão, inativo)
- campos: m_animator, m_settingButton, m_hoverImage, m_defaultStageRepeatHoverSprite, m_actBossStageRepeatHoverSprite, m_deActiveRepeatHoverSprite, m_actBossRepeatCountText, m_actBossRepeatHoverViewObject, m_stageRepeatHoverText, bgci, bgcj, bgck, bgcl, bgcm, bgcn
  - UpdateRepeatStateVisual: Troca sprite do hover entre m_actBossStageRepeatHoverSprite, m_defaultStageRepeatHoverSprite ou m_deActiveRepeatHoverSprite
  - ShowRepeatCountText: Exibe/atualiza m_actBossRepeatCountText com número de repetições disponíveis
  - ToggleHoverViewObject: Ativa m_actBossRepeatHoverViewObject e m_stageRepeatHoverText durante hover
  - ApplyBackgroundColor: Aplica uma das 6 cores (bgci-bgcn) ao estado visual baseado em condição
  - PlayHoverAnimation: Dispara m_animator com estado de entrada/saída de hover
### TaskbarHero.UI.StageRepeatButtonTooltipTrigger
Gerencia o tooltip (dica visual) que aparece ao interagir com o botão de repetição de estágio
  - OnMouseEnter/OnPointerEnter: ativa/mostra o tooltip quando cursor entra no botão
  - OnMouseExit/OnPointerExit: desativa/esconde o tooltip quando cursor sai do botão
  - ShowTooltip: renderiza visualmente a dica com informações do estágio repetível
  - UpdateTooltipPosition: posiciona o tooltip próximo ao cursor/botão
  - GetStageRepeatInfo: busca dados (cooldown, recompensa, tentativas) pra exibir na dica
### TaskbarHero.UI.StageRepeatHoverText
Tooltip/hover text que aparece ao passar mouse sobre um estágio repetível, com posicionamento auto-corrigido para não sair da tela.
- campos: m_text, m_iconObject, m_screenBoundsCorrector
  - lnc: AdjustScreenBounds ou UpdateLayout — ajusta posição do hover text pra caber dentro dos limites de tela usando m_screenBoundsCorrector
### TaskbarHero.UI.StageUserInterfaceHoverObjectToggler
Alterna visibilidade de elementos UI quando o jogador passa o mouse sobre objetos no estágio.
- campos: m_boxCanvasGroup, bgcf
  - OnPointerEnter: Mostra o canvas group (opacidade 1) ao entrar com o mouse
  - OnPointerExit: Esconde o canvas group (opacidade 0) ao sair com o mouse
  - Awake ou Start: Inicializa referência ao CanvasGroup e define estado inicial
  - SetCanvasGroupAlpha: Utilitário para animar/definir a opacidade do grupo
  - ValidateReferences: Valida que m_boxCanvasGroup está atribuído no editor
### TaskbarHero.Option.StartOption
Wizard de inicialização (startup screen) que gerencia termos de serviço, detecção de compatibilidade do sistema (Linux), notificações e progresso através de opções de setup sequenciais até completar.
- campos: panel_m, m_noticeText, m_noticeCheckButton, m_somethingWrongPanel, m_explainLinux, m_linuxFirstOptions, firstOptions, alreadyOptions, m_entryQueue, m_termsOfServiceReAgree, m_termsOfServiceAgreeVersion, bfvs, bfvt, bfvu, currentOption, currentOptionIndex
  - Initialize: configura painéis, carrega firstOptions e alreadyOptions conforme estado
  - ShowNextOption: renderiza opção atual (currentOptionIndex) e incrementa progresso
  - ValidateTermsOfService: verifica concordância e versão antes de avançar
  - CheckSystemCompatibility: detecta Linux e mostra m_explainLinux se necessário
  - ProcessNotification: exibe m_noticeText e aguarda m_noticeCheckButton antes de continuar
### TaskbarHero.UI.StashSlot
Gerencia um slot individual de armazenamento (stash) exibindo item, estado de bloqueio, encantamentos e validação de classe de equipamento.
- campos: m_enchantSlotViewer, usm, Index, image_Image, button_TryUnLock, Go_ManualLock, itemTooltipTrigger, m_equipClassNotMatchingIcon, m_reservedIcon, bgai, bgaj
  - UpdateDisplay: renderiza imagem do item, ícones de status e informações visuais
  - OnClickSlot: trata clique para seleção/interação com o item
  - ToggleLock: alterna bloqueio manual do slot
  - ValidateEquipClass: verifica compatibilidade de classe e exibe ícone de aviso
  - ShowItemTooltip: dispara exibição de tooltip com detalhes do item
### TaskbarHero.UI.StashTabButton
Botão de aba de armazenamento/inventário que gerencia estado visual (selecionado/deselecionado), suporte a conteúdo DLC travado e feedback visual/audio.
- campos: m_button, m_buttonImage, m_dlcTabSprite, m_dlcTabHoverSprite, m_selectedTabSprite, m_unSelectedTabSprite, m_unSelectedTabHoverSprite, m_stashTabText, m_selectedCorrectionY, m_unSelectedCorrectionY, m_isLockedDlcTab, m_selectedColor, m_unSelectedColor, m_onClickRandSoundKey, bgco, bgcp
  - lnd: OnClick — Registra clique, seleciona aba e dispara som
  - lne: SelectTab — Marca aba como selecionada, aplica sprites/cores do estado selected
  - lnf: DeselectTab — Marca aba como deselecionada, volta a sprites/cores padrão
  - lng: UpdateVisuals — Sincroniza aparência (sprite, cor, texto, posição Y) com estado (selected/unlocked/dlc)
### TaskbarHero.UI.SteamDisconnectPopUp
Pop-up de desconexão do Steam que oferece opção de sair do jogo ou aguardar reconexão automática com timer.
- campos: m_gameExitToggleButton, m_reconnectNoticeText, m_reconnectNoticeTime, bfyu
  - leb: Atualiza texto e timer de reconexão a cada frame ou intervalo
  - Possível: OnGameExitToggleChanged - Handler do botão de sair (m_gameExitToggleButton)
  - Possível: UpdateReconnectCountdown - Decrementa m_reconnectNoticeTime
  - Campo bfyu: Provavelmente timer/coroutine ativo ou estado booleano de reconexão em progresso
### TaskbarHero.UI.SteamInventoryFixer
UI para sincronizar/corrigir inventário Steam com estado local via botão e ajuste de parâmetro por slider
- campos: m_fixingTriggerButton, m_slider
  - OnFixButtonClicked: handler do botão que inicia a correção
  - OnSliderValueChanged: callback quando o slider é ajustado (ajusta quantidade/força da correção)
  - FixInventory: lógica principal que aplica a correção ao inventário
  - SyncWithSteam: sincroniza estado local com API/backend Steam
  - ValidateInventoryState: valida/registra antes e depois da correção
### TaskbarHero.UI.SynthesisItemTypeComboBoxButton
ComboBox UI para seleção de tipo de item durante síntese (crafting), gerenciando múltiplos botões de filtro/categoria.
- campos: m_buttons
  - PopulateButtons: inicializa array m_buttons com tipos de item disponíveis
  - OnItemTypeSelected: callback quando usuário seleciona tipo na combobox
  - UpdateButtonStates: ativa/desativa botões baseado em seleção ou disponibilidade
  - ShowComboBoxDropdown: expande a combobox mostrando todas as categorias
  - RefreshUI: reconstrói visual dos botões após mudança de estado
### TaskbarHero.TBHSavedLanguageSelector
Gerencia seleção e persistência do idioma do jogo via PlayerPrefs.
- campos: PREF_KEY_LANGUAGE_INDEX
  - SetLanguageIndex: define idioma ativo e salva no PREF_KEY_LANGUAGE_INDEX
  - LoadLanguagePreference: restaura idioma salvo ao iniciar jogo
  - GetCurrentLanguageIndex: retorna índice do idioma atual em memória
  - SaveLanguagePreference: persiste seleção via PlayerPrefs.SetInt
  - ApplyLocaleToUI: aplica idioma carregado aos elementos de interface
### TaskbarHero.UI.TMP_WindowScaleDropdown
Dropdown UI para seleção de escala de janela (TMP - TextMesh Pro) em TaskbarHero
- campos: bfzh, m_tooltipStringKey, bfzi
  - leq: OnValueChanged - aplica nova escala de janela quando item é selecionado
  - bfzh: provavelmente escala mínima ou índice padrão
  - bfzi: valor numérico de escala atual ou percentual (100%, 125%, 150%, etc)
### TaskbarHero.UI.TermsOfService
Gerenciar UI para exibição e concordância obrigatória com termos de serviço antes de usar o jogo
- campos: m_deAgreeButton, m_nextButton, m_scrollRect, m_textWidthHeightSetter
  - ksz: OnAgreeButtonClicked - Detecta e processa clique no botão de concordar
  - kti: OnNextButtonClicked - Avança para próxima tela após validação
  - OnEnable: Inicializa componentes e listeners da UI
  - SetupButtonInteractivity: Habilita/desabilita botões conforme estado
  - ScrollToEnd: Força scroll até fim antes de permitir 'Agree'
### TaskbarHero.UI.TermsOfServiceReAgree
Tela de re-concordância com Termos de Serviço, permitindo aceitar/rejeitar antes de prosseguir
- campos: m_deAgreeButton, m_nextButton, m_scrollRect, m_buttons, m_textWidthHeightSetter, bfyl
  - ksz: Inicializa handlers de clique dos botões (concordar/discordar/próximo)
  - ldi: Valida se usuário scrollou até o final do ScrollRect antes de permitir prosseguir
  - ldj: Handler do botão m_deAgreeButton (rejeitar — volta ou fecha)
  - kti: Handler do botão m_nextButton (avança para próxima tela após concordar)
### TaskbarHero.UI.TestFaddedUi
Componente UI de teste que anima fade in/out de uma imagem com efeito de desvanecimento
- campos: targetImage
  - Awake/Start: inicializa canvasGroup ou alfa da targetImage em valor inicial
  - FadeIn: corrotina que incrementa alpha da targetImage durante fadeInDuration
  - FadeOut: corrotina que decrementa alpha da targetImage até se tornar transparente
  - UpdateAlpha: atualiza canvasGroup.alpha por frame baseado em tempo decorrido
### TaskbarHero.Data.TierImageContainer
Armazena e gerencia a imagem visual associada a um tier (nível de raridade ou progressão) do sistema de heróis.
- campos: TierImage
  - GetTierImage: retorna a Texture2D/Sprite do tier armazenado
  - SetTierImage: define/carrega a imagem para um tier específico
  - InitializeTierImage: carrega asset da imagem desde disco/resources
  - LoadTierImageByTier: busca a imagem correta baseado no ID do tier
### TaskbarHero.Option.TitleOption
Exibe e anima o título do herói na taskbar com efeito de piscar.
- campos: blinkAnimation
  - kwf: Initialize ou PlayBlinkAnimation (ativa/configura a animação de piscar)
  - Show: Torna o título visível
  - Hide: Oculta o título
### TaskbarHero.ToastManager
Gerenciador de notificações (toasts) que exibe mensagens temporárias em múltiplas posições da tela (esquerda, centro, direita, inferior).
- campos: m_toast_left_transform, m_toast_center_transform, m_toast_right_transform, m_toast_bottom_transform, toast_left, toast_center, toast_right, toast_bottom, bfke, bfkf, bfkg, Toast_Cannot_DeArrange, padding, showDuration, bfkh
  - kec: Enfileira e inicia a exibição de um toast em uma posição específica
  - ked: Remove ou oculta um toast após expiração de showDuration
  - kee: Calcula e aplica a posição correta do toast usando as transforms de referência
  - kef: Atualiza a visibilidade e estado dos toasts ativos baseado em Timer
  - keg: Limpa todos os toasts expirados da fila e reorganiza posições restantes
### TaskbarHero.ToastMessage
Sistema de notificações toast (pop-up) que exibe mensagens temporárias com animação de fade in/out
- campos: text_toast, toastRectTransform, canvasGroup, padding, m_fadeTime, _isShowing, m_fadeCurve, m_maxWidth, bdhg
  - hro: StartFadeInAnimation - inicia animação de entrada
  - hrp: StartFadeOutAnimation - inicia animação de saída
  - hrq: SetMessageText - atualiza o texto do toast
  - hrr: UpdateLayoutSize - recalcula tamanho baseado em m_maxWidth e padding
  - hrs: EvaluateFadeCurve - aplica a curva de animação ao canvasGroup.alpha
### TaskbarHero.UI.TradingStashSlot
UI slot de trading/inventário que exibe itens, gerencia desbloqueios, cooldowns e integração com Steam.
- campos: TestCooldownSeconds, usm, Index, image_Image, text_UnlockCost, button_TryUnLock, Go_ManualLock, itemTooltipTrigger, m_uploadButtonBackGround, m_uploadToSteamButton, m_undoWaitingListButton, m_downloadButtonBackGround, m_downloadToSteamButton, m_coolTimeViewBox, m_coolTimeText, bgat
  - SetItemData: configura ícone, custo de desbloqueio e estado do item no slot
  - UpdateCooldownDisplay: atualiza texto e visual do tempo de cooldown restante
  - OnUnlockButtonClicked: handler para desbloquear item via custo de moeda
  - OnSteamUploadClicked: submete item para workshop do Steam
  - RefreshSlotVisuals: recarrega cores de background e estado visual baseado em disponibilidade
### TaskbarHero.Tutorial.TutorialObject
Gerencia um objeto individual de tutorial, controlando ativação, exibição e progressão dentro da sequência de tutoriais do jogo.
- campos: TutorialObjectKey, bfms, bfmt, m_childTutorialObject, bfmu, bfmv
  - Show: Exibe o tutorial visual na tela
  - Initialize: Inicializa estado e registra callbacks de conclusão
  - Hide: Oculta o tutorial do jogador
  - ActivateChild: Ativa o próximo tutorial na hierarquia
### TaskbarHero.Tutorial.TutorialObjectBlockingButton
Botão do tutorial que bloqueia/desbloqueia sub-elementos de UI até o jogador completar objetivos didáticos
- campos: bfmw, m_subButtons, OnButtonClicked
  - kgw: BlockSubButtons - desativa interação dos sub-botões
  - kgx: UnblockSubButtons - reativa os sub-botões quando tutorial avança
### TaskbarHero.Tutorial.TutorialObjectPopUp
Exibe popup tutorial com texto configurável, múltiplos tamanhos de fundo e botões de fechamento com callback de interação.
- campos: m_popUpText, m_closeButtons, m_smallBackGround, m_largeBackGround, OnButtonClicked
  - Show/Open: exibe popup com transição/animação de fundo apropriado
  - Hide/Close: fecha popup e notifica callback de clique
  - SetupButtons: registra listeners OnClick dos botões de fechar
  - SetText: atualiza conteúdo de m_popUpText em runtime
  - SelectBackground: ativa fundo pequeno ou grande conforme tamanho do conteúdo
### TaskbarHero.Tutorial.TutorialStep
Classe base que define o ciclo de vida de um passo do tutorial onboarding, com inicialização, execução, validação de progresso e limpeza
  - kfc: Initialize — configura o passo do tutorial (UI, textos localizados, callbacks)
  - kff: Execute — exibe o tutorial, destaca componentes na tela
  - kfj: CheckProgress — valida se o jogador completou a ação esperada neste passo
  - kfn: OnComplete — limpa destaques, restaura UI e transiciona para próximo passo
### TaskbarHero.UIManager
Gerenciador central de UI para jogo RPG mobile tipo dungeon-crawler com sistema de heróis, múltiplos painéis e transições entre telas.
- campos: NeedInitCanvasGroup, bddp, bddq, _squareHoverCheck, _squarePlayScreenMask, bddr, bdds, bddt, canvas_Stage, canvas_Main, canvas_FaddedUi, canvas_Qa, Ui_TestFadded, Ui_Stage, ui_log, Ui_Shop
  - InitializeCanvases: Inicializa todos os CanvasGroup e vincula referências de panels UI no startup
  - ShowPanel / OpenScreen: Abre painéis específicos (Shop, Hero, Setting, Mailbox) e gerencia visibilidade de canvas
  - CheckSquareHover: Verifica estado de hover sobre quadrados/tiles do mapa (usa _squareHoverCheck e _squarePlayScreenMask)
  - UpdateHeroDisplay: Atualiza informações e animações do herói exibidas na UI principal
  - OnUITabCloseClicked: Handler para botão de fechar abas, gerencia transição e fade-out de painéis
### TaskbarHero.UI.UI_Base
Gerenciador base de UI com sistema de abas, bloqueio de navegação e states de buttons
- campos: TabType, BlockingPageButton, _usm, bges, bget
  - lrm: Carrega recursos/templates de UI (LoadResourceManager ou similar)
  - hlt: Manipula evento de toggle em lista/aba (HandleListToggle)
  - hlw: Gerencia visibilidade/comportamento de janela de lista (HideListWindow)
  - hly: Aplica transições/yields em elementos da UI (HandleListYield/coroutine wrapper)
### TaskbarHero.UI_Hero
Gerencia a interface do herói: inventário, equipamento, runas, stats (nome/nível/exp) e preview visual
- campos: text_gold, m_heroInventoryPagePanel, m_inventoryScrollRect, ui_Inventory, transform_InventoryParent, Prefab_Inventoryslot, inventorySlots, bdfi, m_activeSkillSlots, m_inventoryRunePageButton, EquipPanel, text_HeroName, text_level, bdfj, slider_HeroExp, gearSlots
  - hlk: Initialize/Setup - configura painéis, slots e listeners iniciais
  - hll: UpdateHeroInfo - atualiza texts de nome, nível, exp slider e descrição
  - hlm: RefreshInventorySlotsDisplay - carrega items do inventário nos prefabs de slot
  - hln: OnInventorySlotClicked - lida com clique em item, mostra detalhe ou equipa
  - hlo: EquipItemToGearSlot - move item do inventário para gear/rune slot
### TaskbarHero.UI.UI_Log
Sistema de UI para exibição e controle de logs/console do jogo com scroll customizado, botões de ação (fixar, expandir, limpar) e auto-close.
- campos: m_logSetting, m_topBottomChanger, m_logSettingObject, m_logScroll, go_BackGroundPanel, m_backPanelImage, button_logClearButton, button_fixButton, button_unfixButton, button_OptionCloseButton, button_OptionButton, button_expandButton, button_shrinkButton, _autoCloseLogScrollViewTime, m_logBackPanel, m_scrollBarObject
  - lrs: ClearLog - limpa todo o conteúdo de logs exibido
  - lrt: ToggleFixed - alterna entre estado fixado/flutuante do painel
  - lru: UpdateScrollHeight - recalcula e redimensiona altura do scroll view
  - lrv: OnAutoCloseTimer - executa fechamento automático após inatividade
  - hlt: HandleOptionToggle - abre/fecha o painel de opções/configurações
### TaskbarHero.UI.UI_Mailbox
Sistema de interface de correio (mailbox) que exibe mensagens, gerencia recompensas e permite ao jogador coletar itens com cooldown de refresh.
- campos: FadeOutTime, NoticeOutBorder, RefreshCoolTimeObject, RefreshCoolTimeText, RefreshCoolTime, OnMailSlotCountChanges, bgfr, transform_MailParent, Prefab_MailSlotParent, bgfs, bgft, m_backendRewardIcon, m_backendRewardGradeBg, m_mailCountDescriptionText, bgfu, toggleButton_ClaimAll
  - RefreshMailList: carrega/atualiza a lista de correios
  - ClaimAllRewards: coleta todas as recompensas pendentes
  - UpdateRefreshCooldown: atualiza timer do cooldown de refresh
  - CreateMailSlot: instancia um slot visual para exibir um mail
  - ShowRewardPopup: exibe popup com detalhes da recompensa coletada
### TaskbarHero.UI.UI_PetArrange
Tela de seleção e visualização de pets com détalhes de habilidades e progresso de aquisição para arranjar na formação.
- campos: m_petParentSlot, m_petModelPreview, m_petAbilityDescriptionObject, m_acquisitionSourceText, m_acquisitionProgressText, m_acquisitionCompleteText, m_leftDescriptionTitle, m_petSlotTooltipTrigger, m_arrangePetButton, m_petDetailInfoPanel, bfyt, OnPetSelected
  - lds: LoadPetPreview — carrega modelo 3D/visualização do pet selecionado
  - ldt: UpdateAbilityDescription — atualiza painel de descrição de habilidades
  - ldu: UpdateAcquisitionInfo — atualiza textos de progresso/conclusão de aquisição
  - ldw: OnArrangePetButtonClicked — handler do botão para arranjar pet na formação
  - ldx: RefreshPetDetailsPanel — atualiza dados completos (slot, tooltips, info geral)
### TaskbarHero.UI.UI_Portal
Gerenciar a UI do portal de seleção de atos e fases com múltiplos níveis de dificuldade (Normal, Hell, Nightmare, Torment).
- campos: transform_ActParent, prefab_ActSlot, currentAct, slotViewCount, scrollRect, bggh, transform_StageParent, m_prefab_NormalStageNode, m_prefab_BossStageNode, m_actMapPanels, m_normalIcon, m_hellIcon, m_nightmareIcon, m_tormentIcon, m_difficultyIconImage, m_currentStageDifficulty
  - SetupActSlots: Instancia e configura os slots de atos no transform parent
  - UpdateDifficultyUI: Atualiza ícone e sprites do dropdown conforme dificuldade selecionada
  - SelectAct: Carrega stages do ato selecionado e atualiza m_actMapPanels
  - SelectStage: Marca stage como selecionado e prepara para entrar em combate
  - OnDifficultyDropdownChanged: Handler que muda m_currentStageDifficulty e atualiza visuais
### TaskbarHero.UI_Setting
Tela de configurações do jogo TaskbarHero com controles para janela, áudio, idioma, features toggles e log/restauração.
- campos: button_Close, dropdown_language, bdhk, text_language, m_logFixSettingButton, toggle_alwaysOnTop, toggle_autoStart, text_version, toggle_discord, toggle_miniCozyRoom, toggle_panoptyca, dropdown_maxFps, dropdown_zoom, button_restoreTransparentWindow, button_restoreWindowPosition, button_restoreWindowScale
  - Start: Inicializa UI com valores salvos de configurações (idioma, volume, flags de toggle)
  - OnLanguageDropdownChanged: Aplica idioma selecionado e salva preferência
  - OnVolumeSliderChanged: Atualiza volume mestre/BGM/SFX conforme slider (aplica em tempo real)
  - OnToggleChanged: Aplica toggle de Discord/AlwaysOnTop/AutoStart/etc e persiste
  - OnRestoreButtonClicked: Restaura posição/escala/transparência da janela e salva estado
### TaskbarHero.UI_Stash
Gerenciador de UI para Stash (baú de armazenagem) com transferência bidirecional de itens entre inventário e stash via slots scrolláveis.
- campos: OnShowStash, OnHideStash, transform_StashParent, Prefab_StashSlotParent, StashSlot, m_scrollRect, bdhy, SortingButton, toggleButton_StashToInventory, toggleButton_InventoryToStash, button_Close, bdhz, bdia
  - hlt: Instancia/popula os slots de stash a partir do prefab
  - hwh: Handler transferência de itens Stash→Inventário
  - hwi: Handler transferência de itens Inventário→Stash
  - hwj: Refresh visual dos slots com estado do stash
  - hwk: Handler botão ordenação/sort dos itens
### TaskbarHero.UI.UI_Status
Exibe e atualiza em tempo real todas as estatísticas, habilidades e informações do herói na interface do jogo
- campos: m_skillPointAnimationController, text_HeroName, text_HeroLevel, slider_Exp, text_HeroAbilityPoint, text_DefaultStat_Level, text_DefaultStat_Exp, text_DefaultStat_DPS, text_DefaultStat_AttackDmg, m_hpStatText, text_DefaultStat_MaxHp, text_DefaultStat_AttackSpeed, text_DefaultStat_CriChance, text_DefaultStat_CriDmg, text_DefaultStat_CoolTimeReduction, text_DefaultStat_MoveSpeed
  - UpdateHeroInfo: atualiza nome, nível e pontos de habilidade do herói
  - UpdateCombatStats: sincroniza todas as stats de combate (dano, crítico, armadura, velocidade)
  - UpdateExperienceBar: atualiza slider de exp e progresso de levelup
  - PlaySkillPointAnimation: reproduz animação quando pontos de skill mudam
  - RefreshAllStats: limpa e reconstrói toda a UI com dados atualizados
### TaskbarHero.ValidationResult
Encapsula resultado de validação com estado booleano de sucesso e mensagem de falha
- campos: 
  - hzk: Factory method estático que retorna ValidationResult com sucesso (IsValid=true)
  - hzl: Factory method estático que retorna ValidationResult com falha (IsValid=false, FailureReason preenchido)
### TaskbarHero.WindowManager
Gerenciar layout, posicionamento e redimensionamento de painéis de UI (MainUI, LogUI, WorldUI, InfoPanel) respondendo a eventos de resolução e monitor
- campos: bddu, bddv, OnWindowResolutionChanged, OnMonitorInfoChanged, OnWorldUIPositionChanged, OnWindowLayoutPresetChanged, OnWindowPositionChanged, OnBlockMoveWindowChanged, m_resolutionWindowX, m_resolutionWindowY, m_rectTransformWorldUI, m_rectTransformMainUI, m_rectTransformLogUI, m_rectTransformWorldUIInfoPanel, m_rectTransformTopLogRegion, m_rectTransformBottomLogRegion
  - UpdateWindowResolution: Atualiza m_resolutionWindowX/Y e dispara OnWindowResolutionChanged
  - RepositionUIElements: Reposiciona RectTransforms dos painéis baseado em nova resolução
  - OnMonitorInfoChanged: Detecta mudança de monitor e ajusta layout/escala
  - ApplyLayoutPreset: Restaura posições salvas dos painéis (OnWindowLayoutPresetChanged)
  - ValidateWindowBounds: Garante que painéis não saem da tela (OnBlockMoveWindowChanged)
### TaskbarHero.Option.WindowPositionOption
Gerencia opções de posicionamento de botões na barra de tarefas (TaskBar) em coordenadas de tela
- campos: _buttonScreenLeft, _buttonScreenRight, bfvy
  - ksz: Aplica/valida a posição dos botões esquerdo e direito na tela
  - GetLeftButtonPosition: Retorna posição X do botão esquerdo
  - GetRightButtonPosition: Retorna posição X do botão direito
  - SetWindowBounds: Define limites de tela para os botões
  - SavePositionPreference: Persiste opção de posicionamento em disco
### TaskbarHero.Option.eOptionState
Enum que define os diferentes estados/páginas do menu de opções (idioma, termos, personagem, configurações, avisos)
- campos: value__, FirstTitle, Language, TermsOfService, Position, Setting, Warning, Character, Loading, Count, LinuxWarning
  - GetLocalizedName: retorna string localizada do nome do estado
  - IsWarningState: bool que verifica se é estado de aviso/warning
  - RequiresValidation: bool que valida se estado necessita confirmação
  - GetNextState: retorna próximo estado da navegação
  - CanNavigateTo: verifica se pode navegar para este estado

## ui,skill  (1)
### TaskbarHero.DragData
Encapsula dados de arraste (drag-and-drop) de slots na UI taskbar, transferindo tipo, índice e ícone do item/skill sendo movido
- campos: SlotType
  - hpe: ValidateDragSource — valida se o slot-origem está válido e pode ser arrastado
  - hpf: CloneDragData — cria cópia dos dados de drag para transferência segura
  - hpg: ApplyDropTarget — aplica/registra o alvo (slot destino) e executa a troca
  - hph: ResetDragState — limpa/reseta dados de arraste após conclusão ou cancelamento
