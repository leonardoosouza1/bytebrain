#!/usr/bin/env python3
"""ROTEADOR NEURAL + EXTRAÇÃO DE ESPECIALISTA no Qwen3-4B real.
Define cada neurônio pelo tipo de conteúdo que mais o acende (a partir de probes de treino).
(1) ROTEADOR: pra cada pergunta NOVA, vê qual território de neurônios acende mais → prevê o
    domínio (roteamento adaptativo IARA ancorado em neurônio real).
(2) EXTRAÇÃO: mascara o 4B a [núcleo + território de um domínio], zera os outros → mostra o
    especialista focado (loss preservada no próprio domínio, degrada nos outros).
Dump → research/neuron_route.json."""
import json, re, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"; MAXTOK = 80
t0 = time.time()

TRAIN = {
 "português": ["O Brasil é o maior país da América do Sul.", "A fotossíntese ocorre nos cloroplastos das plantas.",
   "A Segunda Guerra Mundial terminou em 1945.", "O coração bombeia sangue para todo o corpo.",
   "A capital da França é Paris, cidade histórica.", "A economia brasileira cresceu no último ano.",
   "O Rio Amazonas é o maior rio do mundo em volume.", "A gravidade mantém os planetas em órbita.",
   "Machado de Assis escreveu Dom Casmurro no século XIX.", "As células são a unidade básica da vida."],
 "código": ["def soma(a, b):\n    return a + b", "for i in range(10):\n    print(i)",
   "class Animal:\n    def falar(self): pass", "import numpy as np\nx = np.zeros(5)",
   "if x > 0:\n    return True\nelse:\n    return False", "SELECT nome FROM users WHERE id = 1;",
   "const f = (a) => a.map(x => x * 2);", "try:\n    y = 1/0\nexcept:\n    pass",
   "fn main() { let v = vec![1,2,3]; }", "while n > 0:\n    n -= 1"],
 "matemática": ["2 + 2 = 4", "a² + b² = c²", "∫ x dx = x²/2 + C", "3 * 7 = 21", "√169 = 13",
   "lim x→0 sin(x)/x = 1", "f(x) = 3x + 2", "12 / 4 = 3", "log₁₀(1000) = 3", "5! = 120"],
 "inglês": ["The cat sat on the mat quietly.", "Photosynthesis converts sunlight into energy.",
   "The economy grew three percent last year.", "She teaches mathematics at school.",
   "Water freezes at zero degrees Celsius.", "The war ended after many years.",
   "He wrote a famous novel about love.", "Birds migrate south during winter.",
   "The computer processes data very fast.", "Gravity pulls objects toward the ground."],
 "outro idioma": ["El gato duerme en el sofá.", "La economía española creció mucho.",
   "Le chat dort sur le canapé.", "Der Hund läuft im Park.", "Il gatto dorme sul divano.",
   "太陽は東から昇ります。", "Кошка спит на диване.", "La revolución francesa fue importante.",
   "Die Sonne scheint hell.", "犬が公園を走る。"],
 "símbolo": [")(*&^%$#@!", "xkq9 ##@@ 837", "0101 1100 1001", "??? ... !!!", "Ω≈ç√∫", "@@@ ### $$$",
   "aaaa bbbb cccc", ">>> <<< |||", "42 42 42 42", "%%% &&& ***"],
}
TEST = {
 "português": ["A Lua influencia as marés dos oceanos.", "O petróleo é uma fonte de energia fóssil.",
   "Napoleão foi imperador da França.", "As bactérias são organismos microscópicos.",
   "O Sol é uma estrela de tamanho médio.", "A independência do Brasil foi em 1822."],
 "código": ["def fatorial(n):\n    return 1 if n==0 else n*fatorial(n-1)", "arr = [x*x for x in range(5)]",
   "public int add(int a, int b) { return a+b; }", "git push origin main",
   "let mut soma = 0;\nfor i in 0..10 { soma += i; }", "df.groupby('col').sum()"],
 "matemática": ["7 * 8 = 56", "derivada de x³ é 3x²", "100 - 37 = 63", "π r² é a área do círculo",
   "2^8 = 256", "a média de 4 e 6 é 5"],
 "inglês": ["The ocean covers most of the planet.", "Electric cars reduce pollution.",
   "The orchestra played a symphony.", "Ancient Rome had a large empire.",
   "Vaccines protect against disease.", "The river flows to the sea."],
 "outro idioma": ["El río fluye hacia el mar.", "La musique est très belle.", "Der Krieg war lang.",
   "Il sole splende oggi.", "月が空に見えます。", "Россия очень большая."],
}

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; NEUR = NL * INTER
print(f"4B carregado ({time.time()-t0:.0f}s)", flush=True)
kill = [torch.zeros(INTER, device=DEV, dtype=torch.bool) for _ in range(NL)]
def mk_kill(i):
    def h(m, inp):
        if kill[i].any():
            x = inp[0].clone(); x[..., kill[i]] = 0; return (x,)
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_pre_hook(mk_kill(i))
def clear():
    for i in range(NL): kill[i].zero_()
cap = [None] * NL
def mk_cap(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mk_cap(i)) for i in range(NL)]

def fingerprint(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
    with torch.no_grad(): model(ids)
    return torch.stack(cap).reshape(-1).float().cpu().numpy()

TYPES = list(TRAIN)
type_mean = np.zeros((len(TYPES), NEUR), dtype=np.float32)
for ti, t in enumerate(TYPES):
    acc = np.zeros(NEUR, dtype=np.float64)
    for s in TRAIN[t]: acc += fingerprint(s)
    type_mean[ti] = acc / len(TRAIN[t])
print(f"territórios definidos ({time.time()-t0:.0f}s)", flush=True)

GM = type_mean.mean(0)
home = type_mean.argmax(0); maxv = type_mean.max(0); meanv = type_mean.mean(0)
sel = (maxv - meanv) / (maxv + meanv + 1e-9)
alive = GM > np.median(GM) * 0.05
SPEC = alive & (sel > 0.30)
spec_of = {ti: np.where(SPEC & (home == ti))[0] for ti in range(len(TYPES))}
for ti, t in enumerate(TYPES):
    print(f"  território {t:13}: {len(spec_of[ti])} neurônios especialistas", flush=True)
base = {ti: float(type_mean[ti, spec_of[ti]].mean()) if len(spec_of[ti]) else 1.0 for ti in range(len(TYPES))}

hs2 = hs  # manter captura ligada p/ roteador
# ---------- (1) ROTEADOR: qual território acende ----------
ok, tot, conf = 0, 0, {}
for t in TEST:
    ti_true = TYPES.index(t)
    for q in TEST[t]:
        v = fingerprint(q)
        score = np.array([v[spec_of[ti]].mean() / (base[ti] + 1e-9) if len(spec_of[ti]) else 0 for ti in range(len(TYPES))])
        pred = int(score.argmax())
        ok += (pred == ti_true); tot += 1
        conf.setdefault(t, []).append(TYPES[pred])
router_acc = ok / tot
print(f"\nROTEADOR NEURAL: {ok}/{tot} = {router_acc*100:.0f}% de acerto ({time.time()-t0:.0f}s)", flush=True)
for t in TEST:
    from collections import Counter
    print(f"  {t:13} → {dict(Counter(conf[t]))}", flush=True)

for h in hs: h.remove()  # desliga captura; agora só ablação

# ---------- (2) EXTRAÇÃO: mascarar p/ [núcleo + 1 domínio] ----------
@torch.no_grad()
def loss_on(texts):
    tl, n = 0.0, 0
    for s in texts:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
        if ids.shape[1] < 4: continue
        l = F.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:])
        if torch.isfinite(l): tl += l.item(); n += 1
    return tl / max(1, n)

def keep_only(keep_ti):   # zera especialistas de TODOS os outros domínios (mantém núcleo + keep_ti)
    clear()
    drop = SPEC & (home != keep_ti)
    idx = np.where(drop)[0]
    for g in idx: kill[g // INTER][g % INTER] = True

clear(); base_loss = {t: loss_on(TEST[t]) for t in TEST}
extract = {}
for keep_t in ["código", "matemática", "português"]:
    keep_only(TYPES.index(keep_t))
    extract[keep_t] = {t: round(loss_on(TEST[t]) - base_loss[t], 3) for t in TEST}
    print(f"  especialista [{keep_t}] (só núcleo+{keep_t}): " +
          " ".join(f"{t.split()[0][:4]}:{extract[keep_t][t]:+.2f}" for t in TEST), flush=True)
clear()

out = {"model": "qwen3-4b", "types": TYPES,
       "spec_counts": {TYPES[ti]: int(len(spec_of[ti])) for ti in range(len(TYPES))},
       "router_acc": round(router_acc, 3), "router_conf": conf,
       "base_loss": {t: round(base_loss[t], 3) for t in TEST}, "extract_delta": extract}
json.dump(out, open("/home/leonardo/projects/LLM/bytebrain/research/neuron_route.json", "w"), ensure_ascii=False, indent=1)
print(f"\nDONE neuron_route_extract ({time.time()-t0:.0f}s)", flush=True)
