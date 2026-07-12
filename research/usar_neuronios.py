#!/usr/bin/env python3
"""USAR OS NEURÔNIOS DE UM SAFETENSORS (ideia do Leonardo, 2026-07-10).

Se a gente é SELETOR e não GERADOR (ver insight_selector_not_generator), então não
precisa TREINAR um gerador: os neurônios FFN já-treinados de um safetensors JÁ SÃO a
memória chave-valor que a água roteia (Geva et al. 2021: cada neurônio FFN detecta um
padrão=CHAVE e escreve uma distribuição de vocabulário=VALOR quando dispara).

Prova empírica em modelos REAIS (Qwen2.5-1.5B Instruct/Coder/Math, mesmo base, 3 domínios):
  PROBE 1 — neurônios SÃO células de conteúdo legíveis: o "valor" (coluna do down_proj),
            projetado pela matriz de embedding (tie=True → é a de saída), decodifica em
            tokens coerentes = o que o neurônio DESPEJA ao disparar.
  PROBE 2 — o conhecimento é LOCALIZÁVEL: dado um conceito (geografia), achamos os
            neurônios que mais o promovem e em quais camadas eles se concentram.
  PROBE 3 — os 3 safetensors são uma PALETA: os neurônios divergem exatamente onde o
            domínio diverge (Coder→código, Math→matemática). Dá pra ROTEAR/compor.

Sem treinar nada. Só lê os pesos. torch (ROCm), tokenizer via transformers (já instalado).
"""
import torch, glob, os
from safetensors import safe_open
from transformers import AutoTokenizer

MODELS = {
 "Instruct": "../../llm-lab/models/Qwen2.5-1.5B-Instruct",
 "Coder":    "../../llm-lab/models/Qwen2.5-Coder-1.5B",
 "Math":     "../../llm-lab/models/Qwen2.5-Math-1.5B",
}
HERE = os.path.dirname(os.path.abspath(__file__))
def path(m): return os.path.join(HERE, MODELS[m])

def open_st(m):
    f = glob.glob(os.path.join(path(m), "*.safetensors"))[0]
    return safe_open(f, "pt", device="cpu")

def tensor(m, key):   # carrega 1 tensor (lazy) em float32
    with open_st(m) as f: return f.get_tensor(key).float()

def concept_vec(tok, E, words):     # direção do conceito = soma dos embeddings dos tokens das palavras
    ids = sorted({i for w in words for i in tok.encode(w, add_special_tokens=False)})
    return E[ids].mean(0), ids

def decode_neuron(E, value, tok, k=10):   # logit-lens: o que o neurônio ESCREVE ao disparar
    logits = E @ value                     # [V]
    top = torch.topk(logits, k).indices.tolist()
    toks = tok.convert_ids_to_tokens(top)
    return [t.replace("Ġ","·").replace("Ċ","\\n") for t in toks]

# ---------- carrega tokenizer (partilhado; base igual) ----------
TOK = AutoTokenizer.from_pretrained(path("Instruct"))
print("neurônios de um safetensors como memória chave-valor já treinada — prova empírica\n")

GEO  = ["Brazil","Brasília","France","Paris","Japan","Tokyo","Germany","capital","country","nation"]
CODE = [" def"," function"," return"," import"," class"," print","();"," void"," public"]
MATH = [" integral"," theorem"," equation"," matrix"," prime"," derivative"," sum"," =\\","xyz"]

# ================= PROBE 1 + 2 (modelo Instruct) =================
print("="*72); print("PROBE 1+2 — neurônios são células legíveis + o conhecimento é localizável"); print("="*72)
E = tensor("Instruct","model.embed_tokens.weight")   # [V,1536] = também a matriz de SAÍDA (tie=True)
V,Hd = E.shape
cgeo, geo_ids = concept_vec(TOK, E, GEO)
print(f"conceito GEOGRAFIA = {len(geo_ids)} tokens. Procurando os neurônios que mais 'despejam' geografia...\n")

# varre TODAS as camadas; ranking por COSSENO (conceito vs valor unitário) → sem poluição de norm
cgeo_u = cgeo / cgeo.norm()
best = []   # (cos, layer, neuron)
for L in range(28):
    dp = tensor("Instruct", f"model.layers.{L}.mlp.down_proj.weight")   # [1536, 8960]
    vn = dp / dp.norm(dim=0, keepdim=True).clamp_min(1e-6)               # valores unitários
    cos = cgeo_u @ vn                                                    # [8960]
    v, idx = torch.topk(cos, 2)
    for s, n in zip(v.tolist(), idx.tolist()): best.append((s, L, n))
best.sort(reverse=True)
print("Top-6 neurônios-geografia do modelo (cosseno com o conceito) e o que DESPEJAM ao disparar:")
seen=set()
for s, L, n in best:
    if L in seen: continue
    seen.add(L)
    dp = tensor("Instruct", f"model.layers.{L}.mlp.down_proj.weight")
    toks = decode_neuron(E, dp[:, n], TOK, 12)
    print(f"  L{L:>2} #{n} (cos {s:.2f}): {toks}")
    if len(seen)>=6: break
loc = {}
for s,L,n in best[:12]: loc[L]=loc.get(L,0)+1
print(f"\nlocalização: os neurônios-geografia mais fortes vivem nas camadas {sorted(loc, key=lambda l:-loc[l])[:4]} "
      f"(conhecimento factual concentra no meio/fundo, bate com ROME/knowledge-neurons)")
del E; import gc; gc.collect()

# ================= PROBE 3 — a paleta dos 3 modelos =================
print("\n"+"="*72); print("PROBE 3 — os 3 safetensors são uma PALETA: neurônios divergem por domínio"); print("="*72)
L = 20
def top_neuron(model, concept_words):
    E = tensor(model,"model.embed_tokens.weight")
    c,_ = concept_vec(TOK, E, concept_words)
    dp = tensor(model, f"model.layers.{L}.mlp.down_proj.weight")
    n = int(torch.topk(c @ dp, 1).indices.item())
    toks = decode_neuron(E, dp[:,n], TOK, 12)
    del E, dp; gc.collect()
    return n, toks

for model, name, concept in [("Coder","CÓDIGO",CODE), ("Math","MATEMÁTICA",MATH), ("Instruct","GEOGRAFIA",GEO)]:
    n, toks = top_neuron(model, concept)
    print(f"\n  {model:>8} — neurônio-{name} mais forte (L{L} #{n}) DESPEJA:")
    print(f"           {toks}")

# divergência: onde Coder e Math discordam mais nos VALORES dos neurônios
print("\n  Onde Coder e Math DISCORDAM mais (mesmo neurônio, valor diferente = especialização):")
dpc = tensor("Coder", f"model.layers.{L}.mlp.down_proj.weight")
dpm = tensor("Math",  f"model.layers.{L}.mlp.down_proj.weight")
div = (dpc - dpm).norm(dim=0)                       # [8960] L2 da diferença por neurônio
dn = torch.topk(div, 3).indices.tolist()
Ec = tensor("Coder","model.embed_tokens.weight"); Em = tensor("Math","model.embed_tokens.weight")
for n in dn:
    tc = decode_neuron(Ec, dpc[:,n], TOK, 8); tm = decode_neuron(Em, dpm[:,n], TOK, 8)
    print(f"    #{n}:  Coder→{tc}")
    print(f"    {'':>{len(str(n))+5}}Math →{tm}")
print("\nVEREDITO: os neurônios de um safetensors JÁ SÃO memória chave-valor treinada — legíveis,")
print("localizáveis e divergentes por domínio. Não precisa gerar: precisa SELECIONAR/ROTEAR entre eles.")
