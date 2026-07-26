import json, random, time
from dataclasses import dataclass, field

@dataclass
class Fact:
    key: str; stmt: str; rel: str; obj: list; kind: str
    ts: float; strength: float; used: int
    @property
    def entity(self) -> str: return self.key.split()[0] if self.key else ""

@dataclass
class Graph:
    facts: list
    def by_entity(self):
        d = {}
        for f in self.facts: d.setdefault(f.entity, []).append(f)
        return d

def load_graph(path, min_strength=0.0, drop_bulk=False):
    raw = json.load(open(path))
    seeds = raw.get("seeds", raw)
    out = []
    for k, v in seeds.items():
        if not isinstance(v, dict) or not v.get("stmt"): continue
        kind = v.get("kind", "bulk")
        if drop_bulk and kind == "bulk": continue
        st = float(v.get("str", 0.5))
        if st < min_strength: continue
        t = v.get("t", "00:00")
        ts = float(v.get("tu", 0)) or _hhmm(t)
        out.append(Fact(k, v["stmt"], v.get("rel",""), v.get("obj",[]), kind, ts, st, int(v.get("used",0))))
    return Graph(out)

def _hhmm(t):
    try:
        h, m = t.split(":")[:2]; return float(h)*3600 + float(m)*60
    except Exception: return 0.0

def split_by_entity(g, frac=0.15, seed=0):
    ents = sorted({f.entity for f in g.facts}); random.Random(seed).shuffle(ents)
    n = int(len(ents)*frac); val = set(ents[:n])
    return Graph([f for f in g.facts if f.entity not in val]), Graph([f for f in g.facts if f.entity in val])

@dataclass
class Step:
    query: str; gold_answer: str; gold_fact_key: str; dt_seconds: float; kind: str

@dataclass
class Session:
    steps: list

def make_sessions(g, seed=0, n_sessions=4000, len_range=(3, 12)):
    """Sequências com dt REAL variado: dentro da sessão segundos-minutos; entre sessões horas-dias.
    Inclui REPETIÇÃO ESPAÇADA (o mesmo fato volta depois) — é o sinal que só o tempo resolve."""
    rng = random.Random(seed); byent = g.by_entity(); ents = list(byent)
    out = []
    for _ in range(n_sessions):
        e = rng.choice(ents); fs = byent[e]
        L = rng.randint(*len_range); steps = []
        for i in range(L):
            f = rng.choice(fs)
            dt = rng.uniform(2, 90) if i else rng.choice([300., 3600., 86400., 604800.])
            steps.append(Step(_q(f), f.stmt, f.key, dt, f.kind))
        if rng.random() < 0.5 and steps:            # repetição espaçada
            r = rng.choice(steps)
            steps.append(Step(r.query, r.gold_answer, r.gold_fact_key, rng.choice([3600., 86400.]), "repeticao"))
        out.append(Session(steps))
    return out

def _q(f):
    return f"o que é {f.entity}?" if not f.rel else f"{f.entity} {f.rel}?"
