# IARA — Floresta Viva de Entidades (plano)

> Aposentar a árvore-de-buckets (palavra-chave). O conhecimento da IARA vira UM grafo de
> **entidades** que ela **navega, cresce E mantém sozinha** — um organismo que cuida da própria floresta.
> Data: 2026-07-16.

## 1. Modelo — a floresta
- **Nó = entidade** (nome canônico). "brasil" citado por um fato **é o mesmo nó** que a raiz da árvore Brasil. Entidade nunca duplica.
- **Aresta = relação real** do fato: `sujeito —rel→ objeto`. Já existe nos dados (todo fato é uma tripla).
- **Cada nó é raiz da própria árvore**: clico em Brasil → estados → cada estado é raiz → cidades → distâncias → folhas.
- **Galhos entre árvores (automático)**: um objeto de uma árvore é o sujeito/raiz de outra (São Paulo é citado por muitos E tem a própria árvore) → as árvores se costuram por entidades compartilhadas. Mais os galhos por **correlação** (entidades que dividem vizinhos).
- **Entradas = hubs**: as entidades mais conectadas são os pontos de partida.
- **Mesma floresta pra IARA raciocinar**: ela responde andando pelas relações (músculo→sangue→coração).

## 2. O jardineiro — organismo que cuida (movido a hormônio)
- **Imunidade / validação**: acha contradição (mesmo sujeito+rel → objetos conflitantes) → quarentena → professor desempata. (cortisol)
- **Confiança que respira**: saúde por fato; corroboração↑, desuso↓ (Ebbinghaus); abaixo do limiar → esquece.
- **Poda / fusão**: funde entidade duplicada (brasil↔Brasil), remove órfão/duplicata.
- **Cicatrização**: objeto citado sem definição = lacuna → curiosidade → professor preenche. (dopamina ao fechar)
- **Religação**: correlação nova entre folhas de árvores diferentes → cria galho (choque+contraste).
- **Sono**: replay + consolida nos pesos (LoRA) o que se provou.
- **4 fontes de validação**: consistência interna · corroboração · professor (oráculo) · auto-dedução dela.

## 3. Fases (cada uma verificável)
- **F1 — Núcleo da floresta** *(agora)*: índice de entidades a partir das triplas + fusão de duplicatas + `graph(focus=entidade)` (ego-grafo, expande na vizinhança) + front clique-expande + breadcrumb de trilha. **Valida:** clico "brasil" → vejo estados/geografia/tudo, sem beco; desço a São Paulo → árvore própria; salto entre árvores.
- **F2 — Raciocínio pela floresta**: `_gather` anda pelas relações (multi-hop entidade→entidade). **Valida:** "de que o coração precisa?" → atravessa coração→sangue→oxigênio.
- **F3 — Jardineiro (imunidade+poda+confiança)**, hormonal, observável no painel. **Valida:** planto contradição → ela acha e põe em quarentena; entidade duplicada → ela funde.
- **F4 — Cicatrização + religação + sono**: cresce/religa/consolida sozinha. **Valida:** observo a floresta se curar ao longo do tempo.

## 4. Riscos honestos
- Contradição com falso-positivo; fusão de entidade ambígua (São Paulo cidade × estado) → começar por fusões seguras.
- Escala: ego-grafo é local (ok em 100k+); passes globais de manutenção têm de ser incrementais/amostrados.

## Estágio
Provado em pedaços: verificador (sei/blefo), auto-crescer, esquecimento, choque+contraste, consolidação nos pesos, dopamina/cortisol. **Novo** = costurar tudo num loop vivo sobre a floresta-entidade.
