#!/usr/bin/env python3
"""ELETROESTIMULAÇÃO NEURAL (visão Da Vinci do Leonardo): abrir o Qwen3-4B e aplicar
"eletricidade" direto nos neurônios — SEM inferência pesada (entrada neutra de 2 tokens).
(A) ELETRODO DE DOMÍNIO: injeta corrente nos neurônios de um território (código/mat/chat/inglês)
    e olha QUAIS PALAVRAS o choque evoca na saída (delta logits) + como a onda se propaga.
(B) CAOS: injeta ruído em uma camada (0/9/18/27) com amplitude crescente e mede se a rede
    AMORTECE ou AMPLIFICA a perturbação camada a camada (borda do caos) + KL na saída.
Dump → research/neuro_stim.json + battery_journal.md."""
import json, time
import numpy as np, torch, torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"; MAXTOK = 80
t0 = time.time()

TRAIN = {
 "chat_pt": ["O Brasil é o maior país da América do Sul.", "A fotossíntese ocorre nas plantas.",
   "A capital da França é Paris.", "O coração bombeia sangue pelo corpo.",
   "A gravidade mantém os planetas em órbita.", "As células são a unidade básica da vida."],
 "código": ["def soma(a, b):\n    return a + b", "for i in range(10):\n    print(i)",
   "class Animal:\n    def falar(self): pass", "import numpy as np", "SELECT * FROM users;",
   "const f = (a) => a * 2;"],
 "matemática": ["2 + 2 = 4", "a² + b² = c²", "3 * 7 = 21", "√169 = 13", "f(x) = 3x + 2", "5! = 120"],
 "inglês": ["The cat sat on the mat.", "The economy grew last year.", "Water freezes at zero degrees.",
   "Birds migrate in winter.", "She teaches at school.", "The computer runs fast."],
}

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; NEUR = NL * INTER
print(f"4B carregado ({time.time()-t0:.0f}s) | {NL}L×{INTER}", flush=True)

# infra: estímulo (soma corrente) + captura, na entrada do down_proj
stim = [torch.zeros(INTER, device=DEV, dtype=torch.float16) for _ in range(NL)]
cap = [None] * NL
def mk_stim(i):
    def h(m, inp): return (inp[0] + stim[i],)
    return h
def mk_cap(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
for i in range(NL):
    layers[i].mlp.down_proj.register_forward_pre_hook(mk_stim(i))
    layers[i].mlp.down_proj.register_forward_pre_hook(mk_cap(i))
def clear_stim():
    for i in range(NL): stim[i].zero_()

NEUTRAL = tok("Hoje é", return_tensors="pt").input_ids.to(DEV)   # entrada mínima, 2 tokens
@torch.no_grad()
def run():
    lg = model(NEUTRAL).logits[0, -1].float()
    acts = torch.stack(cap).cpu().numpy()   # [NL, INTER]
    return lg, acts

# ---- territórios (rápido) ----
def fp(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
    with torch.no_grad(): model(ids)
    return torch.stack(cap).reshape(-1).float().cpu().numpy()
TYPES = list(TRAIN)
tmean = np.zeros((len(TYPES), NEUR), np.float32)
for ti, t in enumerate(TYPES):
    tmean[ti] = np.mean([fp(s) for s in TRAIN[t]], 0)
home = tmean.argmax(0); mx, mn = tmean.max(0), tmean.mean(0)
sel = (mx - mn) / (mx + mn + 1e-9); GM = tmean.mean(0)
SPEC = (GM > np.median(GM) * 0.05) & (sel > 0.30)
terr = {t: (SPEC & (home == ti)).reshape(NL, INTER) for ti, t in enumerate(TYPES)}
tm_l = {t: tmean[ti].reshape(NL, INTER) for ti, t in enumerate(TYPES)}
print(f"territórios ({time.time()-t0:.0f}s): { {t: int(terr[t].sum()) for t in TYPES} }", flush=True)

clear_stim(); lg_base, act_base = run()
bnorm = np.linalg.norm(act_base, axis=1) + 1e-9
top_base = [tok.decode([i]) for i in lg_base.topk(5).indices.tolist()]
print(f"baseline (entrada neutra 'Hoje é') → top tokens: {top_base}", flush=True)

# ---- (A) eletrodo de domínio ----
R = {"electrode": {}, "chaos": {}}
for t in TYPES:
    for beta in [3.0]:
        clear_stim()
        for i in range(NL):
            m = terr[t][i]
            if m.any():
                v = torch.from_numpy((beta * tm_l[t][i] * m).astype(np.float16)).to(DEV)
                stim[i] += v
        lg, act = run()
        dev = (np.linalg.norm(act - act_base, axis=1) / bnorm).round(3).tolist()
        delta = (lg - lg_base)
        evoked = [tok.decode([i]).strip() or repr(tok.decode([i])) for i in delta.topk(10).indices.tolist()]
        kl = Fn.kl_div(Fn.log_softmax(lg, -1), Fn.softmax(lg_base, -1), reduction="sum").item()
        R["electrode"][t] = {"beta": beta, "evoked": evoked, "kl": round(kl, 3), "wave": dev}
        print(f"⚡ eletrodo [{t:11}] β={beta}: evoca {evoked[:6]} | KL {kl:.2f}", flush=True)

# ---- (B) caos: ruído numa camada, medir amortecimento ----
gen = torch.Generator(device="cpu").manual_seed(0)
for site in [0, 9, 18, 27]:
    for sigma in [0.5, 2.0, 8.0]:
        clear_stim()
        scale = float(act_base[site].mean())
        noise = torch.randn(INTER, generator=gen).to(DEV, torch.float16) * (sigma * scale)
        stim[site] += noise
        lg, act = run()
        dev = (np.linalg.norm(act - act_base, axis=1) / bnorm)
        kl = Fn.kl_div(Fn.log_softmax(lg, -1), Fn.softmax(lg_base, -1), reduction="sum").item()
        # razão de amortecimento média nas 5 camadas seguintes ao site
        seg = dev[site:min(site + 6, NL)]
        damp = float(np.mean(seg[1:] / (seg[:-1] + 1e-9))) if len(seg) > 1 else 0.0
        R["chaos"][f"L{site}_s{sigma}"] = {"wave": dev.round(3).tolist(), "kl": round(kl, 3), "damp_ratio": round(damp, 3)}
        print(f"🌊 ruído L{site:>2} σ={sigma:>3}: pico {dev.max():.2f} | amortecimento {damp:.2f} | KL saída {kl:.3f}", flush=True)
clear_stim()

json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/neuro_stim.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Eletroestimulação Da Vinci (Qwen3-4B, {int(time.time()-t0)}s)\n")
    for t in TYPES:
        f.write(f"- eletrodo {t}: evoca {R['electrode'][t]['evoked'][:5]} (KL {R['electrode'][t]['kl']})\n")
    f.write(f"- caos: damp médios { {k: v['damp_ratio'] for k, v in R['chaos'].items() if '_s2.0' in k} }\n")
print(f"\nDONE neuro_stim ({time.time()-t0:.0f}s)", flush=True)
