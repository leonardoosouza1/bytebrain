#!/usr/bin/env python3
"""VALIDAÇÃO CAUSAL do atlas: se os neurônios especialistas de um domínio REALMENTE carregam
aquele domínio, desligá-los (zerar) tem que piorar AQUELE domínio — e não os outros.
Constrói a matriz de especificidade: ablação do domínio X → aumento de loss em cada domínio Y.
Diagonal dominante = o atlas é causal, não correlação. Qwen3-4B real (GGUF→torch)."""
import json, re, random, time, sys
import numpy as np, torch, torch.nn.functional as F
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_big.txt"
DEV = "cuda"; N = 1200; K = 12; MAXTOK = 96; NTEST = 6
t0 = time.time()

random.seed(1); res = []; seen = 0
with open(CORPUS, errors="ignore") as f:
    for line in f:
        line = line.strip()
        if len(line) < 200: continue
        seen += 1
        if len(res) < N: res.append(line)
        else:
            j = random.randint(0, seen - 1)
            if j < N: res[j] = line
print(f"corpus: {len(res)} parágrafos ({time.time()-t0:.0f}s)", flush=True)

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; H = model.config.hidden_size
print(f"4B carregado ({time.time()-t0:.0f}s) | {NL}L×{INTER}", flush=True)
def enc(t): return tok(t, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)

# ablation mask (hook zera neurônios marcados)
kill = [torch.zeros(INTER, device=DEV, dtype=torch.bool) for _ in range(NL)]
def mk_kill(i):
    def h(m, inp):
        if kill[i].any():
            x = inp[0].clone(); x[..., kill[i]] = 0; return (x,)
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_pre_hook(mk_kill(i))
def clear():
    for i in range(NL): kill[i].zero_()

# PASSO A: embeddings → clusters
E = np.zeros((len(res), H), dtype=np.float32)
with torch.no_grad():
    for i, p in enumerate(res):
        E[i] = model(enc(p), output_hidden_states=True).hidden_states[-1][0].mean(0).float().cpu().numpy()
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
rng = np.random.default_rng(0); cent = En[rng.choice(len(En), K, replace=False)].copy()
for _ in range(30):
    lab = (En @ cent.T).argmax(1)
    for k in range(K):
        m = lab == k
        if m.sum(): cent[k] = En[m].mean(0); cent[k] /= np.linalg.norm(cent[k]) + 1e-9
print(f"clusters ({time.time()-t0:.0f}s): {dict(Counter(lab.tolist()))}", flush=True)

STOP = set("de a o e que do da em um uma para com os as no na dos das por se ao à mais como foi ser são pela pelo entre também sua seu suas seus após onde qual está este esta esse essa isso ele ela the of and discussão artigo página".split())
def words(k, n=3):
    c = Counter()
    for i in np.where(lab == k)[0]:
        for w in set(re.findall(r"[a-zà-ÿ]{4,}", res[i].lower())):
            if w not in STOP: c[w] += 1
    return ", ".join(w for w, _ in c.most_common(n))

# PASSO B: ativação por neurônio → home/spec
dom_sum = np.zeros((K, NL, INTER), dtype=np.float64); dom_cnt = np.zeros(K, dtype=np.int64)
cap = [None] * NL
def mk_cap(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mk_cap(i)) for i in range(NL)]
with torch.no_grad():
    for i, p in enumerate(res):
        model(enc(p)); dom_sum[lab[i]] += torch.stack(cap).cpu().numpy(); dom_cnt[lab[i]] += 1
for h in hs: h.remove()
dom_mean = (dom_sum / np.maximum(1, dom_cnt)[:, None, None]).reshape(K, -1)
GM = dom_sum.reshape(K, -1).sum(0) / dom_cnt.sum()
home = dom_mean.argmax(0); maxv, meanv = dom_mean.max(0), dom_mean.mean(0)
sel = (maxv - meanv) / (maxv + meanv + 1e-9)
SPEC = (GM >= np.median(GM) * 0.05) & (sel > 0.35)
spec_count = {k: int((SPEC & (home == k)).sum()) for k in range(K)}
# escolhe os domínios mais fortes (mais especialistas)
chosen = sorted(range(K), key=lambda k: -spec_count[k])[:6]
labels = {k: words(k) for k in chosen}
print(f"\ndomínios escolhidos p/ ablação: {[(spec_count[k], labels[k]) for k in chosen]}", flush=True)

def set_ablation(k):
    clear()
    idx = np.where(SPEC & (home == k))[0]
    for g in idx: kill[g // INTER][g % INTER] = True

@torch.no_grad()
def dom_loss(k):
    tot, n = 0.0, 0
    for i in np.where(lab == k)[0][:NTEST]:
        ids = enc(res[i])
        if ids.shape[1] < 4: continue
        lg = model(ids).logits[0, :-1].float(); tgt = ids[0, 1:]
        l = F.cross_entropy(lg, tgt)
        if torch.isfinite(l): tot += l.item(); n += 1
    return tot / max(1, n)

clear(); base = {k: dom_loss(k) for k in chosen}
print(f"baseline losses: { {labels[k]: round(base[k],3) for k in chosen} } ({time.time()-t0:.0f}s)", flush=True)
mat = {}   # ablated -> {tested -> delta}
for ka in chosen:
    set_ablation(ka)
    mat[ka] = {kt: round(dom_loss(kt) - base[kt], 3) for kt in chosen}
    print(f"  ablação [{labels[ka]}] (n={spec_count[ka]}): " +
          " ".join(f"{labels[kt].split(',')[0]}:{mat[ka][kt]:+.2f}" for kt in chosen), flush=True)
clear()

# especificidade: fração das ablações em que o próprio domínio foi o mais afetado
diag_win = sum(1 for ka in chosen if max(mat[ka], key=mat[ka].get) == ka)
out = {"model": "qwen3-4b", "chosen": [{"id": int(k), "label": labels[k], "n_spec": spec_count[k],
        "base_loss": round(base[k], 3)} for k in chosen],
       "matrix": {int(ka): {int(kt): mat[ka][kt] for kt in chosen} for ka in chosen},
       "diag_wins": diag_win, "n_domains": len(chosen)}
json.dump(out, open("/home/leonardo/projects/LLM/bytebrain/research/neuron_ablate.json", "w"), ensure_ascii=False, indent=1)
print(f"\n=== ESPECIFICIDADE: {diag_win}/{len(chosen)} ablações machucaram MAIS o próprio domínio ===")
print(f"DONE neuron_ablate ({time.time()-t0:.0f}s)", flush=True)
