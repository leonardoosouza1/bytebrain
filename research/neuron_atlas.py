#!/usr/bin/env python3
"""IARA NEURON ATLAS — decompõe os 350.208 neurônios MLP do Qwen3-4B REAL em domínios que
EMERGEM de conhecimento amplo (Wikipedia PT, 1.3GB).

PASSO A: embedding do próprio modelo p/ cada parágrafo → clusters = domínios emergentes.
PASSO B: ativação |·| média por neurônio acumulada por domínio → cada neurônio é atribuído
ao domínio que mais o ativa (ESPECIALISTA), ou marcado POLYGLOT (núcleo, dispara em tudo) ou
MORTO (quase nunca dispara). Mede onde cada domínio vive por profundidade + similaridade.
Dump → research/neuron_atlas.json. Puro torch+numpy, roda na GPU de casa."""
import json, re, random, time, sys
import numpy as np, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_big.txt"
DEV = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500     # nº de parágrafos
K = int(sys.argv[2]) if len(sys.argv) > 2 else 28       # nº de domínios emergentes
MAXTOK = 96
t0 = time.time()

# ---------- corpus: reservoir sample de parágrafos amplos ----------
random.seed(0); res = []; seen = 0
with open(CORPUS, errors="ignore") as f:
    for line in f:
        line = line.strip()
        if len(line) < 200:
            continue
        seen += 1
        if len(res) < N:
            res.append(line)
        else:
            j = random.randint(0, seen - 1)
            if j < N:
                res[j] = line
print(f"corpus: {len(res)} parágrafos de {seen} candidatos ({time.time()-t0:.0f}s)", flush=True)

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16,
                                             low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; H = model.config.hidden_size
NEUR = NL * INTER
print(f"4B carregado ({time.time()-t0:.0f}s) | {NL}L × {INTER} = {NEUR} neurônios | VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

def enc(t):
    return tok(t, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)

# ---------- PASSO A: embeddings → clusters (domínios emergem) ----------
E = np.zeros((len(res), H), dtype=np.float32)
with torch.no_grad():
    for i, p in enumerate(res):
        E[i] = model(enc(p), output_hidden_states=True).hidden_states[-1][0].mean(0).float().cpu().numpy()
        if i % 500 == 0:
            print(f"  embed {i}/{len(res)} ({time.time()-t0:.0f}s)", flush=True)
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
rng = np.random.default_rng(0)
cent = En[rng.choice(len(En), K, replace=False)].copy()
for _ in range(30):
    lab = (En @ cent.T).argmax(1)
    for k in range(K):
        m = lab == k
        if m.sum():
            cent[k] = En[m].mean(0); cent[k] /= np.linalg.norm(cent[k]) + 1e-9
sizes = Counter(lab.tolist())
print(f"domínios emergentes (K={K}): tamanhos {dict(sorted(sizes.items()))} ({time.time()-t0:.0f}s)", flush=True)

STOP = set(("de a o e que do da em um uma para com os as no na dos das por se ao à mais como foi ser são "
            "pela pelo entre também sua seu suas seus após onde qual está este esta esse essa isso ele ela "
            "the of and to in is dele dela cujo cujas foram tem seus number references category").split())
def topwords(idxs, n=5):
    c = Counter()
    for i in idxs:
        for w in set(re.findall(r"[a-zà-ÿ]{4,}", res[i].lower())):
            if w not in STOP:
                c[w] += 1
    return [w for w, _ in c.most_common(n)]
dom_words = [topwords([i for i in range(len(res)) if lab[i] == k]) for k in range(K)]

# ---------- PASSO B: ativação por neurônio acumulada por domínio ----------
dom_sum = np.zeros((K, NL, INTER), dtype=np.float64); dom_cnt = np.zeros(K, dtype=np.int64)
cap = [None] * NL
def mkhook(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mkhook(i)) for i in range(NL)]
with torch.no_grad():
    for i, p in enumerate(res):
        model(enc(p))
        dom_sum[lab[i]] += torch.stack(cap).cpu().numpy(); dom_cnt[lab[i]] += 1
        if i % 500 == 0:
            print(f"  ativação {i}/{len(res)} ({time.time()-t0:.0f}s)", flush=True)
for h in hs: h.remove()

dom_mean = dom_sum / np.maximum(1, dom_cnt)[:, None, None]      # [K,NL,INTER]
DM = dom_mean.reshape(K, -1)                                     # [K, NEUR]
GM = dom_sum.reshape(K, -1).sum(0) / dom_cnt.sum()              # média global por neurônio
home = DM.argmax(0)                                             # domínio-lar de cada neurônio
maxv, meanv = DM.max(0), DM.mean(0)
sel = (maxv - meanv) / (maxv + meanv + 1e-9)                    # seletividade 0..1
DEAD = GM < (np.median(GM) * 0.05)
SPEC = (~DEAD) & (sel > 0.35)
POLY = (~DEAD) & (~SPEC)
layer_of = np.repeat(np.arange(NL), INTER)

# ---------- montar atlas ----------
domains = []
for k in range(K):
    sm = SPEC & (home == k)
    lh = np.bincount(layer_of[sm], minlength=NL).tolist()
    domains.append({"id": k, "words": dom_words[k], "n_paragrafos": int((lab == k).sum()),
                    "n_especialistas": int(sm.sum()), "camadas": lh})
domains.sort(key=lambda d: -d["n_especialistas"])
per_layer = [{"layer": i, "spec": int((SPEC & (layer_of == i)).sum()),
              "poly": int((POLY & (layer_of == i)).sum()),
              "dead": int((DEAD & (layer_of == i)).sum())} for i in range(NL)]
# similaridade entre domínios (correlação dos perfis de neurônios)
Dn = DM / (np.linalg.norm(DM, axis=1, keepdims=True) + 1e-9)
sim = Dn @ Dn.T
pairs = sorted(([float(sim[a, b]), a, b] for a in range(K) for b in range(a + 1, K)), key=lambda x: -x[0])

out = {"model": "qwen3-4b (gguf→torch)", "n_layers": NL, "n_neurons": NEUR, "n_paragrafos": len(res), "K": K,
       "global": {"dead": int(DEAD.sum()), "polyglot": int(POLY.sum()), "specialist": int(SPEC.sum()),
                  "dead_pct": round(float(DEAD.mean()) * 100, 1), "poly_pct": round(float(POLY.mean()) * 100, 1),
                  "spec_pct": round(float(SPEC.mean()) * 100, 1)},
       "selectivity_mean": round(float(sel[~DEAD].mean()), 3),
       "domains": domains, "per_layer": per_layer,
       "top_similar": [{"a": dom_words[a][:2], "b": dom_words[b][:2], "sim": round(s, 3)} for s, a, b in pairs[:6]],
       "least_similar": [{"a": dom_words[a][:2], "b": dom_words[b][:2], "sim": round(s, 3)} for s, a, b in pairs[-6:]]}
json.dump(out, open("/home/leonardo/projects/LLM/bytebrain/research/neuron_atlas.json", "w"), ensure_ascii=False, indent=1)

print(f"\n=== ATLAS DO QWEN3-4B ({NEUR} neurônios, {len(res)} parágrafos, {time.time()-t0:.0f}s) ===")
g = out["global"]
print(f"MORTOS {g['dead']} ({g['dead_pct']}%) · POLYGLOT/núcleo {g['polyglot']} ({g['poly_pct']}%) · ESPECIALISTAS {g['specialist']} ({g['spec_pct']}%)")
print(f"\ndomínios emergentes (por nº de neurônios especialistas):")
for d in domains[:K]:
    print(f"  {d['n_especialistas']:>6} neurônios · {d['n_paragrafos']:>3}p · [{', '.join(d['words'])}]")
print("\nDONE neuron_atlas → research/neuron_atlas.json", flush=True)
