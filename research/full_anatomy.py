#!/usr/bin/env python3
"""ANATOMIA COMPLETA GENÉRICA — roda a bateria inteira em QUALQUER modelo transformers:
1) ARENA (fatos/aritmética/mini-MMLU/código), 2) TERRITÓRIOS + roteador (cheio e 4-camadas),
3) ATLAS (mortos/polyglot/especialistas por profundidade), 4) ONDAS (amortecimento de ruído),
5) ELETRODO (chocar matemática → o que evoca). Uso: full_anatomy.py <model_dir> [nome]
Dump → research/anatomy_<nome>.json. Testa se a anatomia Qwen é UNIVERSAL entre famílias."""
import json, sys, time
import numpy as np, torch, torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]; NAME = sys.argv[2] if len(sys.argv) > 2 else MODEL.rstrip("/").split("/")[-1]
DEV = "cuda"; MAXTOK = 80
t0 = time.time()

FACTS = [("A capital da França é", "paris"), ("A capital do Japão é", "tóqu"), ("A capital do Brasil é", "brasília"),
         ("A água ferve a", "100"), ("O maior planeta do sistema solar é", "júpiter"),
         ("A velocidade da luz é aproximadamente", "300"), ("O autor de Dom Casmurro é", "machado"),
         ("A Segunda Guerra Mundial terminou em", "1945"), ("O elemento químico de símbolo O é o", "oxig"),
         ("A capital da Itália é", "roma")]
ARITH = [("27 + 45 = ", "72"), ("9 * 7 = ", "63"), ("84 - 29 = ", "55"), ("144 / 12 = ", "12"),
         ("15 + 38 = ", "53"), ("6 * 12 = ", "72"), ("100 - 64 = ", "36"), ("8 * 8 = ", "64")]
MMLU = [("Qual órgão bombeia o sangue?", "A) pulmão B) coração C) fígado D) rim", "b"),
        ("Qual planeta é o planeta vermelho?", "A) Vênus B) Júpiter C) Marte D) Saturno", "c"),
        ("Quem pintou a Mona Lisa?", "A) Van Gogh B) Da Vinci C) Picasso D) Monet", "b"),
        ("Qual é o maior oceano?", "A) Atlântico B) Índico C) Ártico D) Pacífico", "d"),
        ("H2O é a fórmula da:", "A) água B) sal C) açúcar D) amônia", "a"),
        ("Quantos lados tem um hexágono?", "A) 5 B) 6 C) 7 D) 8", "b"),
        ("A fotossíntese produz:", "A) CO2 B) oxigênio C) nitrogênio D) metano", "b"),
        ("A independência do Brasil foi em:", "A) 1500 B) 1822 C) 1889 D) 1922", "b"),
        ("O DNA fica principalmente no:", "A) núcleo B) membrana C) citoplasma D) ribossomo", "a"),
        ("Quanto é 2 elevado a 5?", "A) 16 B) 32 C) 64 D) 8", "b")]
CODE = [("def soma(a, b):\n    return ", ["a + b", "a+b"]),
        ("# função que retorna o dobro\ndef dobro(x):\n    return ", ["x * 2", "2 * x", "x*2", "2*x"]),
        ("def fatorial(n):\n    if n == 0:\n        return 1\n    return ", ["fatorial"]),
        ("lista = [1, 2, 3]\nprint(len(", ["lista"])]
TRAIN = {
 "português": ["O Brasil é o maior país da América do Sul.", "A fotossíntese ocorre nas plantas.",
   "A capital da França é Paris.", "O coração bombeia sangue.", "A gravidade mantém os planetas.",
   "As células são a unidade da vida.", "A economia cresceu este ano.", "O rio corre para o mar."],
 "código": ["def soma(a, b):\n    return a + b", "for i in range(10):\n    print(i)", "class A:\n    pass",
   "import numpy as np", "SELECT * FROM users;", "const f = a => a * 2;", "while n > 0:\n    n -= 1",
   "x = [i for i in range(5)]"],
 "matemática": ["2 + 2 = 4", "a² + b² = c²", "3 * 7 = 21", "√169 = 13", "f(x) = 3x + 2", "5! = 120",
   "12 / 4 = 3", "2^8 = 256"],
 "inglês": ["The cat sat on the mat.", "The economy grew last year.", "Water freezes cold.",
   "Birds migrate in winter.", "She teaches at school.", "The computer runs fast.",
   "Dogs love long walks.", "Rain falls in spring."],
}
TEST = {
 "português": ["A Lua influencia as marés.", "Napoleão foi imperador.", "O Sol é uma estrela.",
   "As bactérias são microscópicas.", "A independência foi em 1822.", "O petróleo é combustível."],
 "código": ["def fatorial(n):\n    return 1 if n==0 else n*fatorial(n-1)", "arr = [x*x for x in range(5)]",
   "git push origin main", "df.groupby('col').sum()", "public int add(int a){return a;}", "let x = vec![1,2];"],
 "matemática": ["7 * 8 = 56", "100 - 37 = 63", "derivada de x³ é 3x²", "2^10 = 1024", "π r²", "média de 4 e 6 é 5"],
 "inglês": ["The ocean covers the planet.", "Electric cars reduce pollution.", "Ancient Rome was vast.",
   "Vaccines protect health.", "The river flows east.", "Music brings people joy."],
}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers)
INTER = layers[0].mlp.down_proj.in_features; NEUR = NL * INTER
print(f"[{NAME}] {sum(p.numel() for p in model.parameters())/1e9:.2f}B | {NL}L×{INTER}={NEUR} | VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB ({time.time()-t0:.0f}s)", flush=True)

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

R = {"name": NAME, "n_layers": NL, "inter": INTER}

# 1) ARENA
@torch.no_grad()
def gen(prompt, n=8):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
a = {}
a["fatos"] = round(sum(kw in gen(p, 8) for p, kw in FACTS) / len(FACTS), 2)
a["aritmética"] = round(sum(kw in gen(p, 5) for p, kw in ARITH) / len(ARITH), 2)
ok = 0
for q, ops, ans in MMLU:
    r = gen(f"Pergunta: {q}\nOpções: {ops}\nResposta correta: ", 3)
    ok += (next((c for c in r if c in "abcd"), "?") == ans)
a["mini-MMLU"] = round(ok / len(MMLU), 2)
a["código"] = round(sum(any(k in gen(p, 12) for k in kws) for p, kws in CODE) / len(CODE), 2)
a["composto"] = round((a["fatos"] + a["aritmética"] + a["mini-MMLU"] + a["código"]) / 4, 3)
R["arena"] = a
print(f"1 ARENA: {a}", flush=True)

# 2) TERRITÓRIOS + roteador (cheio e 4 camadas)
TYPES = list(TRAIN)
tmean = np.stack([np.mean([fp(s) for s in TRAIN[t]], 0) for t in TYPES])
def router_acc(lim_layers):
    lim = lim_layers * INTER
    tm = tmean[:, :lim]
    h = tm.argmax(0); mx, mn = tm.max(0), tm.mean(0)
    sel = (mx - mn) / (mx + mn + 1e-9)
    S = (tm.mean(0) > np.median(tm.mean(0)) * 0.05) & (sel > 0.30)
    spec = {ti: (S & (h == ti)) for ti in range(len(TYPES))}
    base = {ti: float(tm[ti, spec[ti]].mean()) if spec[ti].any() else 1.0 for ti in range(len(TYPES))}
    ok = tot = 0
    for t in TEST:
        ti_true = TYPES.index(t)
        for q in TEST[t]:
            v = fp(q)[:lim]
            sc = [v[spec[ti]].mean() / (base[ti] + 1e-9) if spec[ti].any() else 0 for ti in range(len(TYPES))]
            ok += (int(np.argmax(sc)) == ti_true); tot += 1
    return ok, tot
home = tmean.argmax(0); mx, mn = tmean.max(0), tmean.mean(0)
sel = (mx - mn) / (mx + mn + 1e-9); GM = tmean.mean(0)
DEAD = GM <= np.median(GM) * 0.05
SPEC = (~DEAD) & (sel > 0.30)
terr = {TYPES[ti]: int((SPEC & (home == ti)).sum()) for ti in range(len(TYPES))}
ok_f, tot = router_acc(NL); ok_4, _ = router_acc(4)
R["territorios"] = terr
R["router"] = {"full": f"{ok_f}/{tot}", "4layers": f"{ok_4}/{tot}"}
print(f"2 TERRITÓRIOS: {terr} | roteador cheio {ok_f}/{tot} · 4-camadas {ok_4}/{tot}", flush=True)

# 3) ATLAS por profundidade
layer_of = np.repeat(np.arange(NL), INTER)
R["dead_per_layer"] = np.bincount(layer_of[DEAD], minlength=NL).tolist()
R["spec_per_layer"] = np.bincount(layer_of[SPEC], minlength=NL).tolist()
R["dead_pct"] = round(float(DEAD.mean()) * 100, 1); R["spec_pct"] = round(float(SPEC.mean()) * 100, 1)
third = NL // 3
sd = R["spec_per_layer"]
print(f"3 ATLAS: mortos {R['dead_pct']}% | spec {R['spec_pct']}% | spec cedo/meio/fundo: {sum(sd[:third])}/{sum(sd[third:2*third])}/{sum(sd[2*third:])}", flush=True)

# 4) ONDAS (ruído σ=2 em cedo/meio/fundo → amortecimento)
NEUTRAL = tok("Hoje é", return_tensors="pt").input_ids.to(DEV)
@torch.no_grad()
def probe():
    lg = model(NEUTRAL).logits[0, -1].float()
    return lg, torch.stack(cap).cpu().numpy()
clear(); lg_b, act_b = probe()
bnorm = np.linalg.norm(act_b, axis=1) + 1e-9
g = torch.Generator(device="cpu").manual_seed(0)
waves = {}
for site in [0, NL // 2, NL - 6]:
    clear()
    scale = float(act_b[site].mean())
    stim[site] += torch.randn(INTER, generator=g).to(DEV, torch.float16) * (2.0 * scale)
    lg, act = probe()
    dev = np.linalg.norm(act - act_b, axis=1) / bnorm
    seg = dev[site:min(site + 6, NL)]
    damp = float(np.mean(seg[1:] / (seg[:-1] + 1e-9)))
    kl = Fn.kl_div(Fn.log_softmax(lg, -1), Fn.softmax(lg_b, -1), reduction="sum").item()
    waves[f"L{site}"] = {"damp": round(damp, 2), "kl": round(kl, 3)}
clear()
R["ondas"] = waves
print(f"4 ONDAS: {waves}", flush=True)

# 5) ELETRODO matemática β=2
tm_l = tmean[TYPES.index("matemática")].reshape(NL, INTER)
mask = (SPEC & (home == TYPES.index("matemática"))).reshape(NL, INTER)
clear()
for i in range(NL):
    if mask[i].any():
        stim[i] += torch.from_numpy((2.0 * tm_l[i] * mask[i]).astype(np.float16)).to(DEV)
lg, _ = probe()
ev = [tok.decode([i]).strip() for i in (lg - lg_b).topk(6).indices.tolist()]
clear()
R["eletrodo_matematica"] = ev
print(f"5 ELETRODO matemática: evoca {ev}", flush=True)

json.dump(R, open(f"/home/leonardo/projects/LLM/bytebrain/research/anatomy_{NAME}.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Anatomia completa — {NAME} ({int(time.time()-t0)}s)\n- arena {a}\n- territórios {terr} | roteador {R['router']}\n- mortos {R['dead_pct']}% spec {R['spec_pct']}% | ondas {waves}\n- eletrodo mat: {ev}\n")
print(f"\nDONE full_anatomy {NAME} ({time.time()-t0:.0f}s)", flush=True)
