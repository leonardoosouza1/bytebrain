#!/usr/bin/env python3
"""BATERIA 4: T17 (K EMERGENTE — nº de domínios emerge sozinho por fusão de clusters, sem eu
fixar K) + T19 (EDIÇÃO DE CONHECIMENTO — suprimir os neurônios de UM fato e ver o modelo
esquecer Paris mas lembrar Tóquio). Uma carga do 4B. Dump battery_04.json + journal."""
import json, random, time
import numpy as np, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"; MAXTOK = 80
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_big.txt"
t0 = time.time()

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; NEUR = NL * INTER
print(f"4B carregado ({time.time()-t0:.0f}s)", flush=True)

kill = [torch.zeros(INTER, device=DEV, dtype=torch.bool) for _ in range(NL)]
cap = [None] * NL
def mk_kill(i):
    def h(m, inp):
        if kill[i].any():
            x = inp[0].clone(); x[..., kill[i]] = 0; return (x,)
    return h
def mk_cap(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
for i in range(NL):
    layers[i].mlp.down_proj.register_forward_pre_hook(mk_kill(i))
    layers[i].mlp.down_proj.register_forward_pre_hook(mk_cap(i))
def clear():
    for i in range(NL): kill[i].zero_()
def fp(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
    with torch.no_grad(): model(ids)
    return torch.stack(cap).reshape(-1).float().cpu().numpy()

# ================= T17: K EMERGENTE =================
def is_prose(s):
    if not (200 <= len(s) <= 900): return False
    if any(b in s for b in ("http", "|", "{{", "==", "[[", "<", "Categoria")): return False
    return sum(c.isalpha() for c in s) / len(s) > 0.72 and s[0].isupper()
random.seed(3); wiki = []; seen = 0
with open(CORPUS, errors="ignore") as f:
    for line in f:
        s = line.strip()
        if not is_prose(s): continue
        seen += 1
        if len(wiki) < 150: wiki.append(s)
        elif random.randint(0, seen) < 150: wiki[random.randint(0, 149)] = s
EXTRA = {"código": ["def f(x):\n    return x*2", "for i in range(9): print(i)", "SELECT * FROM t;",
                    "import numpy as np", "class A:\n    pass", "const g = a => a+1;", "git commit -m 'x'",
                    "while True:\n    break", "x = [i for i in range(5)]", "fn main() {}"],
         "matemática": ["2+2=4", "a²+b²=c²", "∫x dx = x²/2", "√81 = 9", "7*8=56", "f(x)=2x", "5!=120",
                        "log(100)=2", "π≈3.14", "2^6=64"],
         "inglês": ["The cat sleeps.", "Water boils hot.", "Birds fly south.", "The economy grew.",
                    "She reads books.", "Dogs love walks.", "Rain falls today.", "He codes well.",
                    "Music is joy.", "Stars shine bright."],
         "outro": ["El gato duerme.", "Le chat dort.", "Der Hund läuft.", "Il gatto dorme.",
                   "太陽が昇る。", "Кошка спит.", "La lluvia cae.", "Die Sonne scheint.", "猫が寝る。", "Собака бежит."]}
probes = [(s, "wiki") for s in wiki] + [(s, t) for t, L in EXTRA.items() for s in L]
srcs = [t for _, t in probes]; P = len(probes)
print(f"T17: {P} probes heterogêneos ({time.time()-t0:.0f}s)", flush=True)
F = np.zeros((NEUR, P), np.float16)
for j, (s, _) in enumerate(probes):
    F[:, j] = fp(s).astype(np.float16)
Ff = F.astype(np.float32)
glob = Ff.max(1); live = glob > np.median(glob) * 0.05
Fl = Ff[live]; Fl = Fl / (np.linalg.norm(Fl, axis=1, keepdims=True) + 1e-9)
# kmeans fino K=64 → FUSÃO por limiar → K emerge
rng = np.random.default_rng(0); K0 = 64
cent = Fl[rng.choice(len(Fl), K0, replace=False)].copy()
for _ in range(12):
    lab = np.empty(len(Fl), np.int32)
    for s in range(0, len(Fl), 20000):
        lab[s:s+20000] = (Fl[s:s+20000] @ cent.T).argmax(1)
    for k in range(K0):
        m = lab == k
        if m.sum(): cent[k] = Fl[m].mean(0); cent[k] /= np.linalg.norm(cent[k]) + 1e-9
emerge = {}
for tau in [0.90, 0.95, 0.98]:
    # fusão aglomerativa de centróides até nenhuma sim > tau
    C = [c.copy() for c in cent]; sizes = [int((lab == k).sum()) for k in range(K0)]
    C = [c for c, s in zip(C, sizes) if s > 0]; sizes = [s for s in sizes if s > 0]
    merged = True
    while merged and len(C) > 1:
        merged = False; best = (tau, -1, -1)
        for a in range(len(C)):
            for b in range(a + 1, len(C)):
                s = float(C[a] @ C[b])
                if s > best[0]: best = (s, a, b)
        if best[1] >= 0:
            _, a, b = best
            w = sizes[a] + sizes[b]
            C[a] = (C[a] * sizes[a] + C[b] * sizes[b]) / w; C[a] /= np.linalg.norm(C[a]) + 1e-9
            sizes[a] = w; C.pop(b); sizes.pop(b); merged = True
    emerge[str(tau)] = len(C)
print(f"T17 — K EMERGENTE por limiar de fusão: {emerge} (de {K0} iniciais)", flush=True)

# rotular módulos do limiar 0.95 (recomputa atribuição)
tau = 0.95
# refaz fusão guardando mapping
C = [cent[k].copy() for k in range(K0)]; group = list(range(K0))
merged = True
while merged:
    merged = False; best = (tau, -1, -1)
    uniq = sorted(set(group))
    for ai in range(len(uniq)):
        for bi in range(ai + 1, len(uniq)):
            a, b = uniq[ai], uniq[bi]
            s = float(C[a] @ C[b])
            if s > best[0]: best = (s, a, b)
    if best[1] >= 0:
        _, a, b = best
        C[a] = C[a] + C[b]; C[a] /= np.linalg.norm(C[a]) + 1e-9
        group = [a if g == b else g for g in group]; merged = True
lab2 = np.array([group[l] for l in lab])
mods = []
for g in sorted(set(group)):
    m = lab2 == g
    if m.sum() < 100: continue
    prof = Fl[m].mean(0); top = prof.argsort()[::-1][:6]
    dom = Counter(srcs[t] for t in top).most_common(1)[0][0]
    mods.append({"n": int(m.sum()), "fonte": dom})
mods.sort(key=lambda d: -d["n"])
print(f"T17 — módulos emergentes (τ=0.95): {[(m['n'], m['fonte']) for m in mods[:12]]}", flush=True)

# ================= T19: EDIÇÃO — esquecer Paris, lembrar Tóquio =================
@torch.no_grad()
def answer(prompt, n=4):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

FR = ["A capital da França é Paris.", "Paris é a capital da França.", "A França fica na Europa e sua capital é Paris.",
      "A Torre Eiffel fica em Paris, na França."]
CTRL = ["A capital do Japão é Tóquio.", "O Brasil fica na América do Sul.", "A água ferve a 100 graus.",
        "O sol nasce no leste."]
clear()
fr = np.mean([fp(s) for s in FR], 0); ct = np.mean([fp(s) for s in CTRL], 0)
diff = fr - ct
NEDIT = 3000
idx = np.argsort(diff)[::-1][:NEDIT]
base_fr = answer("A capital da França é"); base_jp = answer("A capital do Japão é")
print(f"\nT19 baseline: França → {base_fr!r} | Japão → {base_jp!r}", flush=True)
for g in idx: kill[g // INTER][g % INTER] = True
ed_fr = answer("A capital da França é"); ed_jp = answer("A capital do Japão é")
print(f"T19 após suprimir {NEDIT} neurônios 'França/Paris': França → {ed_fr!r} | Japão → {ed_jp!r}", flush=True)
clear()

R = {"T17_K_emergente": emerge, "T17_modulos_tau095": mods[:15],
     "T19_edicao": {"n_editados": NEDIT, "baseline": {"frança": base_fr, "japão": base_jp},
                    "editado": {"frança": ed_fr, "japão": ed_jp}}}
json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/battery_04.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Lote 4 — T17 K emergente + T19 edição ({int(time.time()-t0)}s)\n")
    f.write(f"- K emergente por limiar: {emerge}\n- módulos τ=0.95: {[(m['n'], m['fonte']) for m in mods[:8]]}\n")
    f.write(f"- edição: França {base_fr!r}→{ed_fr!r} | Japão {base_jp!r}→{ed_jp!r} ({NEDIT} neurônios)\n")
print(f"\nDONE battery_04 ({time.time()-t0:.0f}s)", flush=True)
