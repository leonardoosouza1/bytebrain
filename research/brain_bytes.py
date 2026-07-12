#!/usr/bin/env python3
"""ÓRGÃOS E SERES SE COMUNICAM POR BYTES (tese IARA, Leonardo 2026-07-09).
Se o cérebro funciona em BYTES (8 bits = língua universal), qualquer órgão fala com qualquer órgão, e SERES
trocam conhecimento por bytes. Aqui: um PROFESSOR (cérebro-por-órgãos já treinado) emite um BYTE por pista
(seu julgamento: aproximar/evitar); um ALUNO aprende do próprio reward MAIS do byte do professor (aprendizado
social). Mede se o byte ACELERA o aprendizado, e a ROBUSTEZ do canal (bytes com bits trocados). Reusa
brain_organs (Being = spikes binários + sinapse plástica + neuromod). Honesto, held-out, multi-seed.
"""
import numpy as np
from brain_organs import Being, sparse_rule, rule_value

def to_byte(approach):
    return 255 if approach else 0                      # mensagem do órgão = 1 BYTE

def from_byte(b, flip, rng):
    if flip > 0:
        for bit in range(8):
            if rng.random() < flip:
                b ^= (1 << bit)                        # canal ruidoso: troca bits
    return 1.0 if b >= 128 else -1.0

def eval_being(b, rule, D, seed):
    rng = np.random.default_rng(90000 + seed); test = (rng.random((300, D)) < 0.5).astype(np.float32); ok = 0
    for cue in test:
        v = float(b.w2 @ b.spikes(cue)); ap = (rng.random() < 0.5) if abs(v) < 1e-6 else (v > 0.0)
        val = rule_value(cue, rule); ok += 1 if ((ap and val > 0) or (not ap and val < 0)) else 0
    return ok / 300

def train_teacher(D, H, seed, trials=2500):
    rng = np.random.default_rng(1000 + seed); rule = sparse_rule(D, rng); t = Being(D, H, seed)
    for _ in range(trials):
        cue = (rng.random(D) < 0.5).astype(np.float32); val = rule_value(cue, rule)
        h = t.spikes(cue); v = float(t.w2 @ h)
        ap = (rng.random() < 0.5) if (abs(v) < 1e-6 or rng.random() < 0.1) else (v > 0.0)
        if ap: t.learn(cue, h, val)
    return t, rule

def run_student(D, H, seed, teacher, rule, trials=800, use_bytes=False, flip=0.0, lr_imit=0.06):
    rng = np.random.default_rng(2000 + seed); s = Being(D, H, seed + 777)
    for _ in range(trials):
        cue = (rng.random(D) < 0.5).astype(np.float32); val = rule_value(cue, rule)
        h = s.spikes(cue); v = float(s.w2 @ h)
        ap = (rng.random() < 0.5) if (abs(v) < 1e-6 or rng.random() < 0.1) else (v > 0.0)
        if ap: s.learn(cue, h, val)                    # aprende do PRÓPRIO reward (quando prova)
        if use_bytes:                                  # + aprende do BYTE do professor (imitação social)
            th = teacher.spikes(cue); byte = to_byte(float(teacher.w2 @ th) > 0.0)
            adv = from_byte(byte, flip, rng)
            s.w2 += lr_imit * (adv - float(s.w2 @ h)) * h / (h @ h + 1.0); np.clip(s.w2, -3, 3, out=s.w2)
    return eval_being(s, rule, D, seed)

def main():
    D, H, seeds = 24, 128, 8
    print("=== ÓRGÃOS/SERES SE COMUNICAM POR BYTES (tese IARA) ===")
    print("professor treina 2500 trials; aluno aprende só 800. held-out (300 pistas novas). aleatório=0.50.\n")
    res = {'professor': [], 'aluno SOLO': [], 'aluno + BYTES': [], 'aluno + BYTES 20% ruído': []}
    for s in range(seeds):
        t, rule = train_teacher(D, H, s)
        res['professor'].append(eval_being(t, rule, D, s))
        res['aluno SOLO'].append(run_student(D, H, s, t, rule, use_bytes=False))
        res['aluno + BYTES'].append(run_student(D, H, s, t, rule, use_bytes=True))
        res['aluno + BYTES 20% ruído'].append(run_student(D, H, s, t, rule, use_bytes=True, flip=0.2))
    for k in res:
        print(f"{k:<26} held-out {np.mean(res[k]):.2f}")
    print("\nHonesto: se 'aluno + BYTES' > 'aluno SOLO', o BYTE transferiu conhecimento entre cérebros (comunicação")
    print("universal por bytes = tese IARA). Se com ruído ainda ajuda, o canal é ROBUSTO (robustez-byte).")

if __name__ == "__main__":
    main()
