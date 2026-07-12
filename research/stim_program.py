#!/usr/bin/env python3
"""PROGRAMA DE ELETRICIDADE (não um teste — uma série), respondendo as perguntas do Leonardo:
A) DOSE-RESPOSTA: qual a corrente certa? (β 0.5→8 em matemática/código → KL + tokens evocados)
B) POSIÇÃO DO ELETRODO: chocar cedo/meio/fundo — onde a corrente pega?
C) PILOTAR A GERAÇÃO: corrente suave no território muda o ESTILO do texto gerado?
D) POR QUE Q4/ESCULPIDA FUNCIONAM: ruído nível-quantização em TODAS as camadas → a rede engole?
E) ROTEADOR BARATO: a informação de domínio já está nas primeiras camadas? (rotear lendo só 4/8/12)
Dump → research/stim_program.json + journal."""
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
TEST = {
 "chat_pt": ["A Lua influencia as marés.", "Napoleão foi imperador.", "O Sol é uma estrela.",
   "As bactérias são microscópicas.", "A independência foi em 1822.", "O petróleo é combustível."],
 "código": ["def fatorial(n):\n    return 1 if n==0 else n*fatorial(n-1)", "arr = [x*x for x in range(5)]",
   "git push origin main", "df.groupby('col').sum()", "public int add(int a){return a;}", "let x = vec![1,2];"],
 "matemática": ["7 * 8 = 56", "100 - 37 = 63", "derivada de x³ é 3x²", "2^10 = 1024", "π r²", "média de 4 e 6 é 5"],
 "inglês": ["The ocean covers the planet.", "Electric cars reduce pollution.", "Ancient Rome was vast.",
   "Vaccines protect health.", "The river flows east.", "Music brings people joy."],
}

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; NEUR = NL * INTER
print(f"4B carregado ({time.time()-t0:.0f}s)", flush=True)

stim = [torch.zeros(INTER, device=DEV, dtype=torch.float16) for _ in range(NL)]
cap = [None] * NL
def mks(i):
    def h(m, inp): return (inp[0] + stim[i],)
    return h
def mkc(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
for i in range(NL):
    layers[i].mlp.down_proj.register_forward_pre_hook(mks(i))
    layers[i].mlp.down_proj.register_forward_pre_hook(mkc(i))
def clear():
    for i in range(NL): stim[i].zero_()
def fp(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
    with torch.no_grad(): model(ids)
    return torch.stack(cap).reshape(-1).float().cpu().numpy()

TYPES = list(TRAIN)
tmean = np.stack([np.mean([fp(s) for s in TRAIN[t]], 0) for t in TYPES])
home = tmean.argmax(0); mx, mn = tmean.max(0), tmean.mean(0)
sel = (mx - mn) / (mx + mn + 1e-9); GM = tmean.mean(0)
SPEC = (GM > np.median(GM) * 0.05) & (sel > 0.30)
terr = {t: (SPEC & (home == ti)).reshape(NL, INTER) for ti, t in enumerate(TYPES)}
tm_l = {t: tmean[ti].reshape(NL, INTER) for ti, t in enumerate(TYPES)}
print(f"territórios ({time.time()-t0:.0f}s)", flush=True)

NEUTRAL = tok("Hoje é", return_tensors="pt").input_ids.to(DEV)
@torch.no_grad()
def probe_out():
    return model(NEUTRAL).logits[0, -1].float()
clear(); lg_base = probe_out()

def set_electrode(t, beta, lo=0, hi=None):
    clear(); hi = hi if hi is not None else NL
    for i in range(lo, hi):
        m = terr[t][i]
        if m.any():
            stim[i] += torch.from_numpy((beta * tm_l[t][i] * m).astype(np.float16)).to(DEV)

R = {}
# ---- A) dose-resposta ----
R["A_dose"] = {}
for t in ["matemática", "código"]:
    R["A_dose"][t] = []
    for beta in [0.5, 1.0, 2.0, 4.0, 8.0]:
        set_electrode(t, beta)
        lg = probe_out()
        kl = Fn.kl_div(Fn.log_softmax(lg, -1), Fn.softmax(lg_base, -1), reduction="sum").item()
        ev = [tok.decode([i]).strip() for i in (lg - lg_base).topk(4).indices.tolist()]
        R["A_dose"][t].append({"beta": beta, "kl": round(kl, 3), "evoca": ev})
        print(f"A [{t:10}] β={beta:>3}: KL {kl:7.2f} evoca {ev}", flush=True)

# ---- B) posição do eletrodo (matemática, β=2) ----
R["B_pos"] = []
for lo, hi, nome in [(0, 12, "cedo 0-11"), (12, 24, "meio 12-23"), (24, 36, "fundo 24-35")]:
    set_electrode("matemática", 2.0, lo, hi)
    lg = probe_out()
    kl = Fn.kl_div(Fn.log_softmax(lg, -1), Fn.softmax(lg_base, -1), reduction="sum").item()
    ev = [tok.decode([i]).strip() for i in (lg - lg_base).topk(4).indices.tolist()]
    R["B_pos"].append({"faixa": nome, "kl": round(kl, 3), "evoca": ev})
    print(f"B eletrodo {nome}: KL {kl:.2f} evoca {ev}", flush=True)

# ---- C) pilotar a geração ----
@torch.no_grad()
def gen(n=28):
    ids = tok("Hoje o dia está bonito e", return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
clear(); base_txt = gen()
R["C_steer"] = {"baseline": base_txt}
for t, beta in [("matemática", 1.0), ("matemática", 2.0), ("código", 1.0), ("código", 2.0)]:
    set_electrode(t, beta)
    R["C_steer"][f"{t}_b{beta}"] = gen()
    print(f"C steer [{t} β={beta}]: {R['C_steer'][f'{t}_b{beta}'][:80]!r}", flush=True)
clear()

# ---- D) ruído nível-quantização em TODAS as camadas ----
R["D_quant_noise"] = []
g = torch.Generator(device="cpu").manual_seed(0)
for sigma in [0.02, 0.05, 0.1, 0.2, 0.4]:
    clear()
    for i in range(NL):
        scale = float(tmean.mean(0).reshape(NL, INTER)[i].mean())
        stim[i] += torch.randn(INTER, generator=g).to(DEV, torch.float16) * (sigma * scale)
    lg = probe_out()
    kl = Fn.kl_div(Fn.log_softmax(lg, -1), Fn.softmax(lg_base, -1), reduction="sum").item()
    R["D_quant_noise"].append({"sigma": sigma, "kl": round(kl, 4)})
    print(f"D ruído global σ={sigma}: KL {kl:.4f}", flush=True)
clear()

# ---- E) roteador lendo só as primeiras K camadas ----
R["E_router_cheap"] = {}
for klay in [4, 8, 12, 36]:
    lim = klay * INTER
    tm2 = tmean[:, :lim]
    h2 = tm2.argmax(0); mx2, mn2 = tm2.max(0), tm2.mean(0)
    s2 = (mx2 - mn2) / (mx2 + mn2 + 1e-9)
    S2 = (tm2.mean(0) > np.median(tm2.mean(0)) * 0.05) & (s2 > 0.30)
    spec2 = {ti: (S2 & (h2 == ti)) for ti in range(len(TYPES))}
    base2 = {ti: float(tm2[ti, spec2[ti]].mean()) if spec2[ti].any() else 1.0 for ti in range(len(TYPES))}
    ok, tot = 0, 0
    for t in TEST:
        ti_true = TYPES.index(t)
        for q in TEST[t]:
            v = fp(q)[:lim]
            sc = [v[spec2[ti]].mean() / (base2[ti] + 1e-9) if spec2[ti].any() else 0 for ti in range(len(TYPES))]
            ok += (int(np.argmax(sc)) == ti_true); tot += 1
    R["E_router_cheap"][f"{klay}_camadas"] = f"{ok}/{tot}"
    print(f"E roteador lendo só {klay:>2} camadas: {ok}/{tot} = {ok/tot*100:.0f}%", flush=True)

json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/stim_program.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Programa de eletricidade ({int(time.time()-t0)}s)\n")
    f.write(f"- dose: {json.dumps(R['A_dose'], ensure_ascii=False)[:300]}\n- posição: {R['B_pos']}\n")
    f.write(f"- steer base: {base_txt[:60]!r}\n- ruído global: {R['D_quant_noise']}\n- roteador barato: {R['E_router_cheap']}\n")
print(f"DONE stim_program ({time.time()-t0:.0f}s)", flush=True)
