#!/usr/bin/env python3
"""BATERIA 1 (rumo ao super-modelo LEVE): testes de compressão/modularidade no Qwen3-4B real,
uma carga só. Define territórios de domínio, depois: 1) só-núcleo faz chat? 2) podar mortos+símbolo
é grátis? 3) quais camadas podar? 4) curva de compressão (top-X% neurônios). 7) computação
condicional (núcleo+1 domínio). 8) fundir 2 especialistas. 9) amplificar/suprimir domínio.
Grava research/battery_journal.md + battery_01.json."""
import json, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"; MAXTOK = 80
t0 = time.time(); R = {}

TRAIN = {
 "chat_pt": ["O Brasil é o maior país da América do Sul.", "A fotossíntese ocorre nas plantas verdes.",
   "A Segunda Guerra Mundial terminou em 1945.", "O coração bombeia sangue pelo corpo.",
   "A capital da França é Paris.", "A gravidade mantém os planetas em órbita.",
   "As células são a unidade básica da vida.", "O Sol é uma estrela de tamanho médio."],
 "código": ["def soma(a, b):\n    return a + b", "for i in range(10):\n    print(i)",
   "class Animal:\n    def falar(self): pass", "import numpy as np", "SELECT * FROM users;",
   "const f = (a) => a * 2;", "try:\n    x=1\nexcept: pass", "while n > 0:\n    n -= 1"],
 "matemática": ["2 + 2 = 4", "a² + b² = c²", "3 * 7 = 21", "√169 = 13", "f(x) = 3x + 2",
   "12 / 4 = 3", "5! = 120", "log(1000) = 3"],
 "inglês": ["The cat sat on the mat.", "Photosynthesis converts sunlight.", "The economy grew last year.",
   "Water freezes at zero degrees.", "Birds migrate in winter.", "Gravity pulls objects down.",
   "She teaches at school.", "The computer runs fast."],
 "outro idioma": ["El gato duerme en el sofá.", "Le chat dort sur le canapé.", "Der Hund läuft im Park.",
   "Il gatto dorme sul divano.", "太陽は東から昇ります。", "Кошка спит на диване.",
   "La economía creció mucho.", "Die Sonne scheint hell."],
 "símbolo": [")(*&^%$#@!", "xkq9 ##@@ 837", "0101 1100", "??? ...", "Ω≈ç√∫", "@@@ ###", "42 42 42", ">>> <<<"],
}
GEN = ["Oi, tudo bem? Como você está hoje?", "Me explique o que é fotossíntese de forma simples.",
       "Qual a capital do Japão?", "Escreva uma frase bonita sobre o Brasil.",
       "O que você acha da inteligência artificial?", "Conte uma curiosidade sobre o espaço."]

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; NEUR = NL * INTER
print(f"4B carregado ({time.time()-t0:.0f}s) | {NEUR} neurônios", flush=True)

scale = [torch.ones(INTER, device=DEV, dtype=torch.float16) for _ in range(NL)]
def mk_scale(i):
    def h(m, inp):
        if not torch.all(scale[i] == 1): return (inp[0] * scale[i],)
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_pre_hook(mk_scale(i))
def reset():
    for i in range(NL): scale[i].fill_(1.0)
def kill_mask(boolmask):   # boolmask [NEUR] True=zerar
    reset()
    idx = np.where(boolmask)[0]
    for g in idx: scale[g // INTER][g % INTER] = 0.0

cap = [None] * NL
def mk_cap(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
def fp(text):
    hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mk_cap(i)) for i in range(NL)]
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
    with torch.no_grad(): model(ids)
    for h in hs: h.remove()
    return torch.stack(cap).reshape(-1).float().cpu().numpy()

@torch.no_grad()
def loss_on(texts):
    tl, n = 0.0, 0
    for s in texts:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
        if ids.shape[1] < 4: continue
        l = F.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:])
        if torch.isfinite(l): tl += l.item(); n += 1
    return tl / max(1, n)

# ---- territórios ----
TYPES = list(TRAIN)
tmean = np.zeros((len(TYPES), NEUR), np.float32)
for ti, t in enumerate(TYPES):
    acc = np.zeros(NEUR, np.float64)
    for s in TRAIN[t]: acc += fp(s)
    tmean[ti] = acc / len(TRAIN[t])
GM = tmean.mean(0); home = tmean.argmax(0); mx, mn = tmean.max(0), tmean.mean(0)
sel = (mx - mn) / (mx + mn + 1e-9); alive = GM > np.median(GM) * 0.05
DEAD = ~alive; SPEC = alive & (sel > 0.30); CORE = alive & ~SPEC
spec_of = {ti: (SPEC & (home == ti)) for ti in range(len(TYPES))}
layer_of = np.repeat(np.arange(NL), INTER)
print(f"territórios ({time.time()-t0:.0f}s): núcleo {CORE.sum()} spec {SPEC.sum()} morto {DEAD.sum()}", flush=True)

reset(); base_gen = loss_on(GEN)
R["base"] = {"gen_loss": round(base_gen, 3), "core": int(CORE.sum()), "spec": int(SPEC.sum()), "dead": int(DEAD.sum()),
             "spec_por_dominio": {TYPES[ti]: int(spec_of[ti].sum()) for ti in range(len(TYPES))}}
print(f"baseline chat loss: {base_gen:.3f}", flush=True)

# T1 só-núcleo (mata todos os especialistas)
kill_mask(SPEC); R["T1_core_only"] = {"chat_loss": round(loss_on(GEN), 3), "delta": round(loss_on(GEN)-base_gen, 3),
    "neuronios_ativos_pct": round(float(CORE.mean())*100, 1)}
print(f"T1 só-núcleo ({int(CORE.mean()*100)}% ativos): chat loss {R['T1_core_only']['chat_loss']} (Δ{R['T1_core_only']['delta']:+})", flush=True)

# T2 podar mortos + símbolo
kill_mask(DEAD | spec_of[TYPES.index("símbolo")]); d = loss_on(GEN)
R["T2_prune_dead_sym"] = {"chat_loss": round(d, 3), "delta": round(d-base_gen, 3),
    "podados": int((DEAD | spec_of[TYPES.index("símbolo")]).sum())}
print(f"T2 podar mortos+símbolo ({R['T2_prune_dead_sym']['podados']}): chat {R['T2_prune_dead_sym']['chat_loss']} (Δ{R['T2_prune_dead_sym']['delta']:+})", flush=True)

# T3 redundância de profundidade (zerar MLP de cada camada)
dl = []
for i in range(NL):
    reset(); scale[i].fill_(0.0); dl.append((i, round(loss_on(GEN)-base_gen, 3)))
dl.sort(key=lambda x: x[1])
R["T3_depth"] = {"mais_podaveis": dl[:6], "menos_podaveis": dl[-4:]}
print(f"T3 profundidade — camadas mais podáveis (Δchat): {dl[:6]}", flush=True)

# T4 curva de compressão (manter top-X% dos neurônios vivos por ativação geral)
genfp = np.mean([fp(s) for s in GEN], 0)
order = np.argsort(genfp)[::-1]
curve = {}
for pct in [1.0, 0.75, 0.5, 0.3, 0.15]:
    k = int(NEUR * pct); keep = np.zeros(NEUR, bool); keep[order[:k]] = True
    kill_mask(~keep); curve[f"{int(pct*100)}%"] = round(loss_on(GEN), 3)
R["T4_compression"] = curve
print(f"T4 curva de compressão (chat loss por % neurônios): {curve}", flush=True)

# T7 computação condicional (núcleo + 1 domínio, resto morto) — inferência leve por domínio
cond = {}
for ti, t in enumerate(TYPES):
    if t in ("símbolo",): continue
    keep = CORE | spec_of[ti]
    kill_mask(~keep)
    test = TRAIN[t][4:]  # held-in-ish
    cond[t] = {"ativos_pct": round(float(keep.mean())*100, 1), "dom_loss": round(loss_on(test), 3)}
R["T7_conditional"] = cond
print(f"T7 computação condicional: { {t: cond[t]['dom_loss'] for t in cond} }", flush=True)

# T8 fundir 2 especialistas (núcleo + código + matemática)
c, mth = TYPES.index("código"), TYPES.index("matemática")
keep = CORE | spec_of[c] | spec_of[mth]; kill_mask(~keep)
R["T8_merge_cod_mat"] = {"cod_loss": round(loss_on(TRAIN["código"]), 3), "mat_loss": round(loss_on(TRAIN["matemática"]), 3),
    "pt_loss": round(loss_on(TRAIN["chat_pt"]), 3), "ativos_pct": round(float(keep.mean())*100, 1)}
print(f"T8 fundir cód+mat: {R['T8_merge_cod_mat']}", flush=True)

# T9 amplificar/suprimir um domínio (código)
def scale_dom(ti, f):
    reset(); idx = np.where(spec_of[ti])[0]
    for g in idx: scale[g // INTER][g % INTER] = f
amp = {}
for f in [0.0, 0.5, 1.0, 2.0, 3.0]:
    scale_dom(c, f); amp[f"x{f}"] = round(loss_on(TRAIN["código"]), 3)
R["T9_amplify_code"] = amp
print(f"T9 amplificar/suprimir código (loss de código): {amp}", flush=True)

reset()
json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/battery_01.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Lote 1 — compressão/modularidade (Qwen3-4B, {int(time.time()-t0)}s)\n")
    f.write(f"- núcleo {R['base']['core']} · spec {R['base']['spec']} · morto {R['base']['dead']} · chat baseline {base_gen:.3f}\n")
    f.write(f"- T1 só-núcleo ({R['T1_core_only']['neuronios_ativos_pct']}% ativos): chat Δ{R['T1_core_only']['delta']:+}\n")
    f.write(f"- T2 podar mortos+símbolo: chat Δ{R['T2_prune_dead_sym']['delta']:+}\n")
    f.write(f"- T3 camadas mais podáveis: {dl[:5]}\n")
    f.write(f"- T4 compressão: {curve}\n")
    f.write(f"- T7 condicional: { {t: cond[t]['dom_loss'] for t in cond} }\n")
    f.write(f"- T8 fundir cód+mat: {R['T8_merge_cod_mat']}\n")
    f.write(f"- T9 amplificar código: {amp}\n")
print(f"\nDONE battery_01 ({time.time()-t0:.0f}s)", flush=True)
