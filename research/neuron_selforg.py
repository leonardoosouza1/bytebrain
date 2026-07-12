#!/usr/bin/env python3
"""AUTO-ORGANIZAÇÃO NEURAL (visão do Leonardo): não impor domínios clusterizando texto.
Em vez disso: jogar um fluxo GENÉRICO e heterogêneo ("caos") no Qwen3-4B, capturar a
impressão-digital de ativação de CADA neurônio, e clusterizar os NEURÔNIOS pela co-ativação
(quem dispara junto). Os módulos se estabilizam sozinhos; a gente só NOMEIA depois, olhando
o que acende cada um. Também testa a hipótese: os neurônios MORTOS na Wikipedia-PT acordam
com outro conteúdo (código/mat/idiomas)? = eram de outro domínio.
Dump → research/neuron_selforg.json."""
import json, re, random, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_big.txt"
DEV = "cuda"; NWIKI = 360; KMOD = 24; MAXTOK = 80
t0 = time.time()

# ---------- corpus LIMPO (só prosa de artigo) ----------
def is_prose(s):
    if not (200 <= len(s) <= 1200): return False
    if any(b in s for b in ("http", "www.", "Categoria", "Ficheiro", "Predefinição", "ISBN",
                            "|", "{{", "}}", "==", "[[", "#", "*", "<", ">")): return False
    letters = sum(c.isalpha() for c in s); digits = sum(c.isdigit() for c in s)
    if letters / len(s) < 0.72 or digits / len(s) > 0.15: return False
    if not s[0].isupper() or s[-1] not in ".!?": return False
    if len(s.split()) < 8: return False
    return True

random.seed(2); wiki = []; seen = 0
with open(CORPUS, errors="ignore") as f:
    for line in f:
        s = line.strip()
        if not is_prose(s): continue
        seen += 1
        if len(wiki) < NWIKI: wiki.append(s)
        elif random.randint(0, seen - 1) < NWIKI: wiki[random.randint(0, NWIKI - 1)] = s
print(f"corpus limpo: {len(wiki)} parágrafos de prosa (de {seen}) ({time.time()-t0:.0f}s)", flush=True)

# ---------- fluxo heterogêneo (o "caos") — tags só p/ interpretar DEPOIS ----------
CODE = ["def soma(a,b):\n    return a+b", "for i in range(10):\n    print(i)", "class Dog:\n    def bark(self): pass",
    "import numpy as np\narr = np.zeros((3,3))", "fn main() { println!(\"hi\"); }", "let x = [1,2,3].map(|v| v*2);",
    "SELECT * FROM users WHERE age > 18;", "public static void main(String[] args) {}", "const f = async () => await g();",
    "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    pass", "#include <stdio.h>\nint main(){return 0;}",
    "git commit -m 'fix bug' && git push origin main", "npm install react && npm run build",
    "x = [i**2 for i in range(20) if i%2==0]", "struct Point { x: f64, y: f64 }", "while True:\n    break",
    "df = pd.read_csv('data.csv').groupby('id').mean()", "curl -X POST http://api/v1 -d '{}'",
    "match opt { Some(v) => v, None => 0 }", "func add(a int, b int) int { return a+b }"]
MATH = ["2 + 2 = 4", "∫ x² dx = x³/3 + C", "a² + b² = c²", "lim x→0 sin(x)/x = 1", "3 * 7 = 21",
    "E = mc²", "√144 = 12", "f(x) = 2x + 1", "Σ n = n(n+1)/2", "P(A∩B) = P(A)P(B)",
    "det(A) = ad - bc", "dy/dx = 2x", "log₂(8) = 3", "π ≈ 3.14159", "n! = n·(n-1)!",
    "15% de 200 = 30", "matriz 3x3 identidade", "2^10 = 1024", "cos(0) = 1", "5 + 8 * 2 = 21"]
EN = ["The quick brown fox jumps over the lazy dog.", "Photosynthesis converts light into chemical energy.",
    "The stock market fell sharply on Monday.", "She studied medicine at the university.",
    "Water boils at one hundred degrees Celsius.", "The president signed the new law yesterday.",
    "Machine learning models require large datasets.", "The Roman Empire lasted for centuries.",
    "He plays the guitar in a rock band.", "Climate change affects global weather patterns.",
    "The recipe calls for two cups of flour.", "Astronomers discovered a new exoplanet.",
    "The economy grew by three percent last year.", "Ancient Egyptians built enormous pyramids.",
    "The novel explores themes of love and loss.", "Vaccines train the immune system to fight disease.",
    "The river flows through several countries.", "Electric cars are becoming more popular.",
    "The orchestra performed a beautiful symphony.", "Programmers debug code to fix errors."]
OUTRO = ["El gato duerme en el sofá durante la tarde.", "La economía española creció el año pasado.",
    "Le chat dort sur le canapé pendant l'après-midi.", "La révolution française a commencé en 1789.",
    "Der Hund läuft schnell durch den großen Park.", "Die deutsche Wirtschaft wächst dieses Jahr.",
    "Il gatto dorme sul divano tutto il giorno.", "犬が公園を走っています。", "猫はソファで寝ています。",
    "Кошка спит на диване весь день.", "El río fluye a través de la montaña.", "La musique classique est belle.",
    "太陽は東から昇ります。", "Россия — самая большая страна мира.", "Der Krieg dauerte mehrere Jahre."]
CAOS = ["xkq9 ##@@ 837 zzz!!!", "aaaa bbbb cccc dddd", "3.14 2.71 1.61 0.577", ")(*&^%$#@!", "0101 1100 1001 0011",
    "asdf jkl; qwer uiop", "??? ... !!! ,,, ;;;", "Ω≈ç√∫˜µ≤≥÷", "42 42 42 42 42", "AAAA aaaa AAAA aaaa"]

probes = [(s, "wiki") for s in wiki] + [(s, "código") for s in CODE] + [(s, "matemática") for s in MATH] + \
         [(s, "inglês") for s in EN] + [(s, "outro idioma") for s in OUTRO] + [(s, "caos/símbolo") for s in CAOS]
srcs = [t for _, t in probes]
P = len(probes)
print(f"fluxo heterogêneo: {P} probes ({dict((t, srcs.count(t)) for t in set(srcs))}) ({time.time()-t0:.0f}s)", flush=True)

# ---------- capturar impressão-digital de cada neurônio ----------
tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size; NEUR = NL * INTER
print(f"4B carregado ({time.time()-t0:.0f}s) | {NEUR} neurônios", flush=True)
cap = [None] * NL
def mk(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mk(i)) for i in range(NL)]
F = np.zeros((NEUR, P), dtype=np.float16)
with torch.no_grad():
    for j, (s, _) in enumerate(probes):
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
        model(ids); F[:, j] = torch.stack(cap).reshape(-1).cpu().numpy().astype(np.float16)
        if j % 100 == 0: print(f"  probe {j}/{P} ({time.time()-t0:.0f}s)", flush=True)
for h in hs: h.remove()
Ff = F.astype(np.float32)

# ---------- hipótese: morto na wiki acorda em outro domínio? ----------
wiki_idx = [j for j in range(P) if srcs[j] == "wiki"]
other_idx = [j for j in range(P) if srcs[j] != "wiki"]
wiki_max = Ff[:, wiki_idx].max(1); glob_max = Ff.max(1)
DEAD_WIKI = wiki_max < (np.median(glob_max) * 0.05)
woke = {}
for t in ["código", "matemática", "inglês", "outro idioma", "caos/símbolo"]:
    idx = [j for j in range(P) if srcs[j] == t]
    a = Ff[:, idx].max(1)
    woke[t] = int((DEAD_WIKI & (a > np.median(glob_max) * 0.15)).sum())
print(f"\nmortos-na-wiki: {int(DEAD_WIKI.sum())} | acordam em: {woke} ({time.time()-t0:.0f}s)", flush=True)

# ---------- auto-organização: clusterizar os NEURÔNIOS pela co-ativação ----------
live = glob_max > (np.median(glob_max) * 0.05)
Fl = Ff[live]; Fl = Fl / (np.linalg.norm(Fl, axis=1, keepdims=True) + 1e-9)   # padrão, não magnitude
rng = np.random.default_rng(0); cent = Fl[rng.choice(len(Fl), KMOD, replace=False)].copy()
for it in range(18):
    lab = np.empty(len(Fl), dtype=np.int32)
    for s in range(0, len(Fl), 20000):
        lab[s:s+20000] = (Fl[s:s+20000] @ cent.T).argmax(1)
    for k in range(KMOD):
        m = lab == k
        if m.sum(): cent[k] = Fl[m].mean(0); cent[k] /= np.linalg.norm(cent[k]) + 1e-9
print(f"módulos auto-organizados: {KMOD} (18 iters) ({time.time()-t0:.0f}s)", flush=True)

layer_of = np.repeat(np.arange(NL), INTER)[live]
mods = []
for k in range(KMOD):
    m = lab == k
    if m.sum() == 0: continue
    prof = Fl[m].mean(0)                       # ativação média do módulo por probe
    top = prof.argsort()[::-1][:8]
    top_src = [srcs[t] for t in top]
    from collections import Counter
    dom_src = Counter(top_src).most_common(1)[0]
    lh = np.bincount(layer_of[m], minlength=NL).tolist()
    mods.append({"id": int(k), "n": int(m.sum()), "fonte_dominante": dom_src[0], "fonte_frac": round(dom_src[1]/8, 2),
                 "top_fontes": top_src, "top_probes": [probes[t][0][:60] for t in top[:4]], "camadas": lh})
mods.sort(key=lambda d: -d["n"])

out = {"model": "qwen3-4b", "n_neurons": NEUR, "n_probes": P, "n_modules": len(mods),
       "dead_wiki": int(DEAD_WIKI.sum()), "dead_wake": woke,
       "modules": mods}
json.dump(out, open("/home/leonardo/projects/LLM/bytebrain/research/neuron_selforg.json", "w"), ensure_ascii=False, indent=1)
print(f"\n=== MÓDULOS QUE SE AUTO-ORGANIZARAM ({time.time()-t0:.0f}s) ===")
for md in mods:
    print(f"  {md['n']:>6} neurônios · fonte {md['fonte_dominante']:12} ({int(md['fonte_frac']*100)}%) · ex: {md['top_probes'][0][:45]}")
print("\nDONE neuron_selforg → research/neuron_selforg.json", flush=True)
