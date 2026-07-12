#!/usr/bin/env python3
"""A VIDA DE UM SER — hormônios ao longo do TEMPO, inteligência dia-a-dia (Leonardo 2026-07-09).
Inteligência não vem num dia. Aqui um ser VIVE vários dias num domínio (pistas comida/veneno por regra escondida;
pistas novas -> tem que GENERALIZAR). Junta as LEIS que descobrimos, agindo no TEMPO:
  DOPAMINA (RPE) = gate de plasticidade + motivação (explora mais quando recompensa é escassa).
  CORTISOL (estresse) = sobe com veneno; REDUZ plasticidade (dia ruim aprende menos); decai/reseta no sono.
  SEROTONINA (freio/equilíbrio) = calibra; alta quando calmo. Impede aprendizado impulsivo.
  ADENOSINA (fadiga) = sobe com atividade; força o SONO.
  SONO = consolida (replay das memórias salientes do hipocampo -> reforça) e reseta fadiga/estresse.
Precisão por propósito: spikes binários, sinapse float-no-update, hormônios float. Mede competência (held-out)
DIA A DIA + trajetória dos hormônios. Ablação: sem sono / estresse crônico. numpy, honesto.
"""
import numpy as np

D_IN, H = 24, 96

def sparse_rule(rng, k=6):
    r = np.zeros(D_IN); r[rng.choice(D_IN, k, replace=False)] = rng.choice([-1.0, 1.0], k); return r

def rule_value(cue, rule):
    return 1.0 if float((cue - 0.5) @ rule) > 0 else -1.0

class Being:
    def __init__(self, seed, sleep=True, chronic_stress=False):
        rng = np.random.default_rng(seed)
        self.W1 = rng.choice([-1, 0, 1], (H, D_IN), p=[.25, .5, .25]).astype(float)  # fiação binária/ternária
        self.b1 = -rng.random(H) * 0.4
        self.w2 = np.zeros(H)                                   # sinapse plástica
        self.cortisol = 0.0; self.serotonin = 0.6; self.dopamine = 0.0; self.adenosine = 0.0
        self.recent_reward = 0.0
        self.hippo = []                                        # episódios salientes do dia
        self.sleep_on = sleep; self.chronic = chronic_stress
        self.base_lr = 0.10; self.rng = rng

    def spikes(self, cue):
        return (self.W1 @ cue + self.b1 > 0).astype(float)

    def live_day(self, rule, trials=400):
        for _ in range(trials):
            cue = (self.rng.random(D_IN) < 0.5).astype(float); val = rule_value(cue, rule)
            h = self.spikes(cue); v = float(self.w2 @ h)
            # motivação: dopamina baixa (pouca recompensa recente) -> explora mais
            eps = 0.08 + 0.25 * max(0.0, -self.recent_reward)
            approach = (self.rng.random() < 0.5) if (self.rng.random() < eps or abs(v) < 1e-6) else (v > 0.0)
            if approach:
                rpe = val - v; self.dopamine = rpe                          # DOPAMINA = erro de predição
                lr = self.base_lr * (1 - 0.7 * self.cortisol) * self.serotonin  # cortisol trava, serotonina calibra
                self.w2 += lr * rpe * h / (h @ h + 1.0); np.clip(self.w2, -3, 3, out=self.w2)
                self.recent_reward = 0.9 * self.recent_reward + 0.1 * val
                if val < 0: self.cortisol = min(1.0, self.cortisol + 0.06)  # veneno -> estresse
                if abs(rpe) > 0.6: self.hippo.append((h.copy(), val))       # HIPOCAMPO: guarda o saliente
            self.cortisol *= 0.995 if not self.chronic else 1.0            # estresse crônico não decai
            self.serotonin = np.clip(0.9 * self.serotonin + 0.1 * (1 - self.cortisol), 0.2, 1.0)
            self.adenosine = min(1.0, self.adenosine + 1.0 / trials)

    def night_sleep(self):
        if self.sleep_on and self.hippo:                                   # SONO: consolida o saliente (replay)
            for h, val in self.hippo[-64:]:
                v = float(self.w2 @ h); self.w2 += 0.08 * (val - v) * h / (h @ h + 1.0)
            np.clip(self.w2, -3, 3, out=self.w2)
        self.hippo = []; self.adenosine = 0.0                              # fadiga zera
        self.cortisol *= 0.5                                               # estresse alivia dormindo

def competence(b, rule, seed):
    rng = np.random.default_rng(90000 + seed); test = (rng.random((300, D_IN)) < 0.5).astype(float); ok = 0
    for cue in test:
        v = float(b.w2 @ b.spikes(cue)); ap = (rng.random() < 0.5) if abs(v) < 1e-6 else (v > 0.0)
        val = rule_value(cue, rule); ok += 1 if ((ap and val > 0) or (not ap and val < 0)) else 0
    return ok / 300

def live(days=16, seed=0, sleep=True, chronic=False):
    rng = np.random.default_rng(1000 + seed); rule = sparse_rule(rng)
    b = Being(seed, sleep=sleep, chronic_stress=chronic); comp = []; horm = []
    for d in range(days):
        b.live_day(rule); b.night_sleep()
        comp.append(competence(b, rule, seed)); horm.append((b.cortisol, b.serotonin))
    return np.array(comp), horm

def bench(label, seeds=8, **kw):
    cs = np.array([live(seed=s, **kw)[0] for s in range(seeds)]).mean(0)
    days = [1, 4, 8, 12, len(cs)]
    print(f"{label:<26} " + "  ".join(f"d{d}:{cs[d-1]*100:.0f}%" for d in days))
    return cs

def main():
    print("=== A VIDA DE UM SER — inteligência crescendo DIA A DIA (competência held-out, aleatório=50%) ===")
    print("domínio: pistas comida/veneno por regra escondida; generaliza. hormônios agem no tempo; sono consolida.\n")
    print(">>> competência por dia:")
    normal = bench("ser NORMAL (com sono)")
    bench("ABLAÇÃO: SEM sono", sleep=False)
    bench("ABLAÇÃO: estresse CRÔNICO", chronic=True)
    # trajetória hormonal de um ser normal
    _, horm = live(days=16, seed=0)
    print("\n>>> hormônios ao longo dos dias (ser normal):")
    print("  dia:     " + " ".join(f"{d+1:>4}" for d in range(0, 16, 3)))
    print("  cortisol:" + " ".join(f"{horm[d][0]:>4.2f}" for d in range(0, 16, 3)))
    print("  seroton.:" + " ".join(f"{horm[d][1]:>4.2f}" for d in range(0, 16, 3)))
    print(f"\nHonesto: se NORMAL sobe dia a dia e bate SEM-sono e ESTRESSE-crônico, as leis (dopamina aprende, sono")
    print("consolida, cortisol trava) valem. É o currículo que pode ajudar o treino de byte-models: dias, sono, equilíbrio.")

if __name__ == "__main__":
    main()
