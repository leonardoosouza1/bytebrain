#!/usr/bin/env python3
"""CÉREBRO VIRTUAL POR ÓRGÃOS — "ser" que APRENDE UM DOMÍNIO (Leonardo 2026-07-09).
Segue PLANO_CEREBRO_VIRTUAL.md (precisão por propósito):
  spikes/ativação = BINÁRIO {0,1} (mensagem) · sinapse = float no UPDATE / TERNÁRIO no REPOUSO (memória)
  neuromoduladores = FLOAT (dopamina=RPE, ACh=foco, cortisol=medo) · córtex = float.
Aprende por PLASTICIDADE DE TRÊS FATORES: Δw ∝ pré(h) × pós(dopamina) × modulador(ACh). Sinapse começa em ZERO.
DOMÍNIO: pistas (padrões binários) valem comida(+)/veneno(-) por REGRA ESCONDIDA esparsa; pistas NOVAS a cada
trial -> tem que GENERALIZAR (medido em HELD-OUT). REVERSÃO no meio (regra inverte) -> tem que REAPRENDER.
Valida: ablação (tira dopamina/ACh/hipocampo), precisão (ternário no repouso ~= float?), e nicho do hipocampo
(mundo com MARCOS que repetem). Numpy puro, honesto, multi-seed.
"""
import numpy as np

class Being:
    def __init__(self, D, H, seed, ternary_store=False, dopamine=True, ach=True, hippo=True, plastic=True):
        rng = np.random.default_rng(seed)
        self.W1 = rng.choice([-1, 0, 1], size=(H, D), p=[0.25, 0.5, 0.25]).astype(np.float32)  # fiação fixa ternária
        self.b1 = -(rng.random(H).astype(np.float32) * 0.4)
        self.w2 = np.zeros(H, np.float32)                 # SINAPSE plástica (float no aprender), começa em ZERO
        self.H = H
        self.ternary_store, self.dopamine_on = ternary_store, dopamine
        self.ach_on, self.hippo_on, self.plastic = ach, hippo, plastic
        self.cortisol = 0.0
        self.hippo_mem = {}
        self.base_lr = 0.08

    def spikes(self, cue):
        return (self.W1 @ cue + self.b1 > 0).astype(np.float32)     # BINÁRIO

    def _q(self, w):
        s = np.abs(w).max() + 1e-9
        return np.round(w / s).clip(-1, 1) * s                      # ternário {-1,0,1}×escala

    def w2eff(self):
        return self._q(self.w2) if self.ternary_store else self.w2  # REPOUSO/inferência ternário; aprende float

    def value(self, cue, h):
        v = float(self.w2eff() @ h)
        if self.hippo_on:
            k = cue.tobytes()
            if k in self.hippo_mem:
                v = 0.5 * v + 0.5 * self.hippo_mem[k]              # HIPOCAMPO: recall exato rápido
        return v

    def learn(self, cue, h, r):
        if not self.plastic:
            return
        v = float(self.w2eff() @ h)
        rpe = r - v                                                # DOPAMINA = erro de predição
        dopa = rpe if self.dopamine_on else 0.0
        ach = (1.0 + 0.4 * min(1.0, abs(rpe))) if self.ach_on else 1.0   # ACh = foco (suave)
        self.w2 += self.base_lr * ach * dopa * h / (h @ h + 1.0)   # 3 FATORES, normalizado, update em FLOAT
        np.clip(self.w2, -3.0, 3.0, out=self.w2)
        if self.hippo_on:
            k = cue.tobytes()
            self.hippo_mem[k] = 0.6 * self.hippo_mem.get(k, 0.0) + 0.4 * r
        self.cortisol = min(1.0, self.cortisol + 0.1) if r < 0 else self.cortisol * 0.9

def rule_value(cue, rule):
    return 1.0 if float((cue - 0.5) @ rule) > 0 else -1.0

def eval_heldout(being, rule, D, seed):
    rng = np.random.default_rng(90000 + seed)
    test = (rng.random((300, D)) < 0.5).astype(np.float32)
    w = being.w2eff(); ok = 0
    for cue in test:
        v = float(w @ being.spikes(cue))                           # só sinapse (sem hipocampo) = generalização pura
        approach = (rng.random() < 0.5) if abs(v) < 1e-6 else (v > 0.0)
        val = rule_value(cue, rule)
        ok += 1 if ((approach and val > 0) or (not approach and val < 0)) else 0
    return ok / len(test)

def run_life(D=24, H=128, trials=3000, reversal=True, seed=0, p_repeat=0.0, **kw):
    rng = np.random.default_rng(1000 + seed)
    rule = np.zeros(D, np.float32)                                  # REGRA ESCONDIDA ESPARSA (6 features importam)
    idx = rng.choice(D, size=6, replace=False)
    rule[idx] = rng.choice([-1.0, 1.0], size=6).astype(np.float32)
    landmarks = (rng.random((6, D)) < 0.5).astype(np.float32)       # pistas que REPETEM (nicho do hipocampo)
    being = Being(D, H, seed, **kw)
    correct = np.zeros(trials); ho_pre = 0.0
    for t in range(trials):
        if reversal and t == trials // 2:
            ho_pre = eval_heldout(being, rule, D, seed); rule = -rule
        cue = landmarks[rng.integers(6)].copy() if rng.random() < p_repeat else (rng.random(D) < 0.5).astype(np.float32)
        val = rule_value(cue, rule)
        h = being.spikes(cue)
        v = being.value(cue, h)
        greedy = (rng.random() < 0.5) if abs(v) < 1e-6 else (v > 0.0)
        approach = (rng.random() < 0.5) if rng.random() < 0.1 else greedy
        correct[t] = 1.0 if ((approach and val > 0) or (not approach and val < 0)) else 0.0
        if approach:
            being.learn(cue, h, val)
    ho_post = eval_heldout(being, rule, D, seed)
    return correct, ho_pre, ho_post

def sparse_rule(D, rng, k=6):
    r = np.zeros(D, np.float32); idx = rng.choice(D, size=k, replace=False)
    r[idx] = rng.choice([-1.0, 1.0], size=k).astype(np.float32); return r

def eval_md(being, rule, D, ctx, context, seed):
    rng = np.random.default_rng(90000 + seed)
    test = (rng.random((300, D)) < 0.5).astype(np.float32); w = being.w2eff(); ok = 0
    for cue in test:
        x = np.concatenate([cue, ctx]) if context else cue
        v = float(w @ being.spikes(x))
        approach = (rng.random() < 0.5) if abs(v) < 1e-6 else (v > 0.0)
        val = rule_value(cue, rule)
        ok += 1 if ((approach and val > 0) or (not approach and val < 0)) else 0
    return ok / 300

def run_multidomain(D=24, H=128, phase=1600, seed=0, context=False, **kw):
    """Aprende domínio A, depois domínio B. Mede se lembra de A depois (esquecimento). Context = órgão 'onde estou'."""
    rng = np.random.default_rng(1000 + seed)
    Dc = D + (2 if context else 0)
    ruleA, ruleB = sparse_rule(D, rng), sparse_rule(D, rng)
    being = Being(Dc, H, seed, **kw)
    ctxA, ctxB = np.array([1, 0], np.float32), np.array([0, 1], np.float32)
    def phase_run(rule, ctx):
        for _ in range(phase):
            cue = (rng.random(D) < 0.5).astype(np.float32)
            x = np.concatenate([cue, ctx]) if context else cue
            val = rule_value(cue, rule); h = being.spikes(x); v = being.value(x, h)
            greedy = (rng.random() < 0.5) if abs(v) < 1e-6 else (v > 0.0)
            approach = (rng.random() < 0.5) if rng.random() < 0.1 else greedy
            if approach: being.learn(x, h, val)
    phase_run(ruleA, ctxA)
    a_before = eval_md(being, ruleA, D, ctxA, context, seed)
    phase_run(ruleB, ctxB)
    a_after = eval_md(being, ruleA, D, ctxA, context, seed)
    b = eval_md(being, ruleB, D, ctxB, context, seed)
    return a_before, a_after, b

def bench_md(label, seeds=8, **kw):
    r = np.array([run_multidomain(seed=s, **kw) for s in range(seeds)]).mean(0)
    print(f"{label:<30} A antes {r[0]:.2f} → A depois de aprender B {r[1]:.2f}  |  B {r[2]:.2f}")

def run_multidomain_gated(D=24, H=128, phase=1600, seed=0):
    """Sinapse GATEADA por contexto: banco W[contexto]. O córtex/hipocampo escolhe o sub-conjunto -> sem interferência."""
    rng = np.random.default_rng(1000 + seed)
    ruleA, ruleB = sparse_rule(D, rng), sparse_rule(D, rng)
    being = Being(D, H, seed)                       # usado só p/ spikes (fiação binária)
    W = np.zeros((2, H), np.float32); lr = 0.08
    def phase_run(rule, g):
        for _ in range(phase):
            cue = (rng.random(D) < 0.5).astype(np.float32); val = rule_value(cue, rule)
            h = being.spikes(cue); v = float(W[g] @ h)
            approach = (rng.random() < 0.5) if (abs(v) < 1e-6 or rng.random() < 0.1) else (v > 0.0)
            if approach:
                rpe = val - v; ach = 1.0 + 0.4 * min(1.0, abs(rpe))
                W[g] += lr * ach * rpe * h / (h @ h + 1.0); np.clip(W[g], -3, 3, out=W[g])
    def ev(rule, g):
        rng2 = np.random.default_rng(90000 + seed); t = (rng2.random((300, D)) < 0.5).astype(np.float32); ok = 0
        for cue in t:
            v = float(W[g] @ being.spikes(cue)); ap = (rng2.random() < 0.5) if abs(v) < 1e-6 else (v > 0.0)
            val = rule_value(cue, rule); ok += 1 if ((ap and val > 0) or (not ap and val < 0)) else 0
        return ok / 300
    phase_run(ruleA, 0); a_before = ev(ruleA, 0)
    phase_run(ruleB, 1); a_after = ev(ruleA, 0); b = ev(ruleB, 1)
    return a_before, a_after, b

def window_acc(c, w=250):
    return np.convolve(c, np.ones(w) / w, mode='valid')

def bench(label, seeds=8, **kw):
    curves, pres, posts = [], [], []
    for s in range(seeds):
        c, pre, post = run_life(seed=s, **kw)
        curves.append(c); pres.append(pre); posts.append(post)
    curve = window_acc(np.mean(curves, axis=0)); n = len(curve)
    st = [curve[n // 10], curve[4 * n // 10], curve[6 * n // 10], curve[-1]]
    print(f"{label:<26} held-out pré {np.mean(pres):.2f} pós {np.mean(posts):.2f} | stream {st[0]:.2f}→{st[1]:.2f} [rev] {st[2]:.2f}→{st[3]:.2f}")
    return np.mean(pres), np.mean(posts), st[-1]

def main():
    print("=== SER (cérebro por órgãos) APRENDE UM DOMÍNIO por plasticidade 3-fatores ===")
    print("regra escondida esparsa; pistas NOVAS a cada trial (GENERALIZAR); reversão na metade. held-out/aleatório=0.50.\n")
    print(">>> APRENDE E GENERALIZA? (held-out >> 0.50; pós-reversão re-sobe)")
    bench("ser COMPLETO")
    bench("controle SEM plasticidade", plastic=False)
    print("\n>>> ABLAÇÃO (lesão) — quanto cada órgão vale?")
    bench("LESÃO sem DOPAMINA", dopamine=False)
    bench("LESÃO sem ACETILCOLINA", ach=False)
    bench("LESÃO sem HIPOCAMPO", hippo=False)
    print("\n>>> PRECISÃO POR PROPÓSITO — sinapse ternária NO REPOUSO (aprende float, age ternário)")
    bench("sinapse FLOAT", ternary_store=False)
    bench("sinapse TERNÁRIA no repouso", ternary_store=True)
    print("\n>>> NICHO DO HIPOCAMPO — mundo com MARCOS (70% das pistas repetem)")
    bench("com HIPOCAMPO (repete)", p_repeat=0.7, hippo=True)
    bench("sem HIPOCAMPO (repete)", p_repeat=0.7, hippo=False)
    print("\n>>> UM DOMÍNIO OU MAIS — aprende A, depois B; lembra de A? (esquecimento catastrófico)")
    bench_md("2 domínios SEM contexto", context=False)
    bench_md("2 domínios COM contexto (fraco)", context=True)
    r = np.array([run_multidomain_gated(seed=s) for s in range(8)]).mean(0)
    print(f"{'2 domínios GATEADO por contexto':<30} A antes {r[0]:.2f} → A depois de aprender B {r[1]:.2f}  |  B {r[2]:.2f}")
    print("Leitura md: sem gate B sobrescreve A (esquece); com sinapse GATEADA por contexto A se mantém = sem esquecimento.")
    print("\nLeitura: COMPLETO generaliza (held-out>0.5) e reaprende pós-reversão; sem dopamina=acaso (dopamina esculpe");
    print("a sinapse); ternário-no-repouso ~= float valida precisão-por-propósito; hipocampo brilha quando há repetição.")

if __name__ == "__main__":
    main()
