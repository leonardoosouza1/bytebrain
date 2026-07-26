import torch, torch.nn as nn

class WaterMemory(nn.Module):
    """Grafo de fatos + difusão água÷grau (PPR), diferenciável.
    A água entra pelos fatos parecidos com a pergunta, escorre pelas arestas, e ÷grau mata o hub."""
    def __init__(self, graph, encoder, d=128, alpha=0.85, iters=6, cache_emb=True, refresh_every=200):
        super().__init__()
        self.graph, self.encoder, self.d = graph, encoder, d
        self.alpha, self.iters = alpha, iters
        self.cache_emb = cache_emb
        self.refresh_every = refresh_every
        self.stmts = [f.stmt for f in graph.facts]
        ents = sorted({f.entity for f in graph.facts})
        eidx = {e:i for i,e in enumerate(ents)}
        N = len(graph.facts)
        rows, cols = [], []
        by_e = {}
        for i,f in enumerate(graph.facts): by_e.setdefault(eidx[f.entity], []).append(i)
        import random as _r; _rng = _r.Random(0)
        MAX_DEG = 64                                      # gap G13: hub não explode
        for _, ids in by_e.items():                       # fatos da mesma entidade se ligam
            for a in ids:
                viz = ids if len(ids) <= MAX_DEG+1 else _rng.sample(ids, MAX_DEG+1)
                for b in viz:
                    if a != b: rows.append(a); cols.append(b)
        A = torch.sparse_coo_tensor(torch.tensor([rows,cols]), torch.ones(len(rows)), (N,N)).coalesce()
        self.register_buffer("A", A)
        self.register_buffer("degree", torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0))
        self.register_buffer("prior", torch.tensor([f.strength for f in graph.facts], dtype=torch.float32))
        self._cache = None
        self._cache_step = 0   # how many calls have been served from the current cache
        self._step_count = 0   # total fact_embeddings() calls (for refresh scheduling)

    def invalidate_cache(self):
        """Force a cache rebuild on the next fact_embeddings() call."""
        self._cache = None
        self._cache_step = 0

    def fact_embeddings(self):
        # Eval mode: use cache indefinitely (original behaviour)
        if self.cache_emb and self._cache is not None and not self.training:
            return self._cache

        # Training mode: use cached index unless it is missing or refresh window elapsed
        if self.cache_emb and self.training:
            if self._cache is not None and self._cache_step < self.refresh_every:
                self._cache_step += 1
                self._step_count += 1
                return self._cache
            # Cache is stale or missing — rebuild under no_grad (ANCE-style periodic refresh)
            with torch.no_grad():
                outs = []
                for i in range(0, len(self.stmts), 512):
                    outs.append(self.encoder(self.stmts[i:i+512]))
                E = torch.cat(outs).detach()
            self._cache = E
            self._cache_step = 1
            self._step_count += 1
            return self._cache

        # cache_emb=False: always recompute (unchanged behaviour)
        outs = []
        for i in range(0, len(self.stmts), 512): outs.append(self.encoder(self.stmts[i:i+512]))
        E = torch.cat(outs)
        if not self.training: self._cache = E.detach()
        return E

    def soft_scores(self, q_emb, by_degree=True):
        E = self.fact_embeddings()
        p = torch.softmax(q_emb @ E.t() / (self.d**0.5), dim=-1)   # a gota entra aqui
        r = p
        for _ in range(self.iters):                                # a água escorre
            r = (1-self.alpha)*p + self.alpha*torch.sparse.mm(self.A, (r/self.degree).t()).t()
        r = r * self.prior                                         # sinapse forte puxa mais
        return r/self.degree if by_degree else r                   # ÷grau = a correção validada

    def retrieve(self, q_emb, k=8):
        s = self.soft_scores(q_emb)
        v, i = torch.topk(s, min(k, s.shape[-1]), dim=-1)
        return i, v

    def facts_text(self, idx):
        return [[self.stmts[j] for j in row.tolist()] for row in idx]
