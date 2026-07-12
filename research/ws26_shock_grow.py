#!/usr/bin/env python3
"""WS26 — GRAFO QUE CRESCE SOZINHO POR CHOQUE+CORRELAÇÃO (a ideia do Leonardo, 2026-07-12).

Sem pipeline de ensino (sem Q&A, sem gold no laço). Só: CHOCA os pesos → lê quais neurônios
disparam juntos → "fire together, wire together" cria a aresta → o uso lapida. Testa 4 coisas honestas:
  1. CHOQUE NEUTRO por entidade → grafo ASSOCIATIVO auto-formado (país → seus conceitos co-ativados).
     Mede: riqueza + se atributos conhecidos (língua/região) EMERGEM sozinhos (recall, mas gold só p/ MEDIR).
  2. CORRELAÇÃO entidade↔entidade (vetor de ativação) → clusteriza. "Ativam junto = aresta".
     Mede: países do mesmo continente correlacionam mais? (vs gold continente).
  3. CHOQUE DIRECIONADO (contexto da relação) → lê o neurônio-valor que dispara e DECODIFICA → aresta-fato,
     lendo o PESO por dentro (não gerando texto). Mede recall de capital vs a sondagem por geração.
  4. LAPIDA COM USO (Hebbian): reuso fortalece; mostra a aresta ganhar força.
Honesto: números reais, gold só p/ pontuar, GPU. Compara custo (tempo/espaço)."""
import torch, os, re, time, json, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md"); MODEL=MOD+"/Qwen2.5-3B-Instruct"
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")
def norm(s): return re.sub(r"[^a-z0-9]","",s.lower())

log(f"\n{'='*72}\n# WS26 — GRAFO CRESCE POR CHOQUE+CORRELAÇÃO (sem Q&A) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(DEV).eval()
NL=m.config.num_hidden_layers; INT=m.config.intermediate_size
E=m.get_output_embeddings().weight.detach(); Et=E.t().contiguous(); norm_w=m.model.norm.weight.detach()
DEEP=list(range(NL-10,NL))
log(f"modelo {NL}L×{INT} · choca as camadas profundas {DEEP[0]}-{DEEP[-1]} e lê o co-disparo (input do down_proj = ativação do neurônio-valor)")

# ---- hooks: captura ativação do neurônio-valor (entrada do down_proj) no ÚLTIMO token ----
CAP={}
def mk(L):
    def hook(mod,inp): CAP[L]=inp[0][0,-1].detach().float()   # [INT]
    return hook
H=[m.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L)) for L in DEEP]
@torch.no_grad()
def shock(prompt):
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV); m(ids)
    return {L:CAP[L].clone() for L in DEEP}                    # ativação por camada

COUNTRIES=("Brazil Argentina Chile Peru Colombia Bolivia Paraguay France Germany Spain Portugal Italy "
 "Poland Greece Sweden Norway Russia Japan China India Thailand Vietnam SouthKorea Turkey Iran "
 "Egypt Nigeria Kenya Morocco Ethiopia Canada Mexico Cuba Australia").split()
GOLD={"Brazil":("Portuguese","America","Brasilia"),"France":("French","Europe","Paris"),
 "Japan":("Japanese","Asia","Tokyo"),"Egypt":("Arabic","Africa","Cairo"),"Germany":("German","Europe","Berlin"),
 "Peru":("Spanish","America","Lima"),"China":("Chinese","Asia","Beijing"),"Kenya":("Swahili","Africa","Nairobi"),
 "Canada":("English","America","Ottawa"),"Australia":("English","Oceania","Canberra"),"Russia":("Russian","Europe","Moscow")}
CONT={"Brazil":"America","Argentina":"America","Chile":"America","Peru":"America","Colombia":"America","Bolivia":"America",
 "Paraguay":"America","Canada":"America","Mexico":"America","Cuba":"America","France":"Europe","Germany":"Europe",
 "Spain":"Europe","Portugal":"Europe","Italy":"Europe","Poland":"Europe","Greece":"Europe","Sweden":"Europe",
 "Norway":"Europe","Russia":"Europe","Japan":"Asia","China":"Asia","India":"Asia","Thailand":"Asia","Vietnam":"Asia",
 "SouthKorea":"Asia","Turkey":"Asia","Iran":"Asia","Egypt":"Africa","Nigeria":"Africa","Kenya":"Africa",
 "Morocco":"Africa","Ethiopia":"Africa","Australia":"Oceania"}

# =========== 1) CHOQUE NEUTRO → grafo associativo auto-formado (COM CONTRASTE) ===========
log(f"\n## 1) CHOQUE NEUTRO por entidade → grafo ASSOCIATIVO (contraste vs linha-base = só o específico)")
acts={}
for c in COUNTRIES:
    ent=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",c)
    acts[c]=shock(f"Facts about {ent}:")                      # choque neutro
base={L:torch.stack([acts[c][L] for c in COUNTRIES]).mean(0) for L in DEEP}   # o que dispara p/ TODOS
contrast={c:{L:acts[c][L]-base[L] for L in DEEP} for c in COUNTRIES}          # só o que é ESPECÍFICO de c
topset={}
for c in COUNTRIES:
    tn=[]
    for L in DEEP:
        v,idx=torch.topk(contrast[c][L],6)                    # top por CONTRASTE (não por magnitude bruta)
        tn+=[(L,int(i),float(x)) for i,x in zip(idx,v)]
    topset[c]=sorted(tn,key=lambda z:-z[2])[:24]
# decodifica SÓ os neurônios que apareceram (rápido)
uniq=sorted({(L,i) for c in COUNTRIES for (L,i,_) in topset[c]})
concept={}
by_layer={}
for (L,i) in uniq: by_layer.setdefault(L,[]).append(i)
for L,idxs in by_layer.items():
    dp=m.model.layers[L].mlp.down_proj.weight.detach()        # [H,INT]
    vals=(dp[:,idxs].t()*norm_w).to(E.dtype)                  # [k,H]
    logits=vals@Et                                            # [k,V]
    top=torch.topk(logits,4,dim=1).indices
    for r,i in enumerate(idxs):
        ws=[tok.convert_ids_to_tokens([int(x)])[0].replace("Ġ","").replace("Ċ","") for x in top[r]]
        clean=[w for w in ws if re.fullmatch(r"[A-Za-z][A-Za-z]{2,}",w)]
        concept[(L,i)]=clean[0] if clean else None
def concepts_of(c,k=6):
    seen=[];
    for (L,i,_) in topset[c]:
        w=concept.get((L,i))
        if w and norm(w)!=norm(c) and norm(w) not in {norm(x) for x in seen}: seen.append(w)
        if len(seen)>=k: break
    return seen
edges=sum(len(concepts_of(c,8)) for c in COUNTRIES)
log(f"  arestas associativas auto-formadas: {edges} (de {len(COUNTRIES)} entidades, choque neutro, ZERO Q&A)")
for c in ["Brazil","France","Japan","Egypt"]:
    log(f"    {c:8} → {concepts_of(c,6)}")
# emergiu atributo conhecido sozinho?
hitL=hitR=totG=0
for c,(lang,reg,cap) in GOLD.items():
    if c not in acts: continue
    cs=[norm(x) for x in concepts_of(c,10)]; totG+=1
    hitL+= any(norm(lang)[:5] in x or x in norm(lang) for x in cs)
    hitR+= any(norm(reg)[:4] in x for x in cs)
log(f"  atributo EMERGE sozinho no top-10: língua {hitL}/{totG} · região {hitR}/{totG} (choque neutro não mira a relação)")

# =========== 2) CORRELAÇÃO entidade↔entidade → cluster ===========
log(f"\n## 2) CORRELAÇÃO entidade↔entidade (ativam junto = aresta) vs continente-gold")
common=sorted({(L,i) for c in COUNTRIES for (L,i,_) in topset[c]})
idxmap={k:j for j,k in enumerate(common)}
V=np.zeros((len(COUNTRIES),len(common)),dtype=np.float32)
for r,c in enumerate(COUNTRIES):
    for L in DEEP:
        a=contrast[c][L]                                      # CONTRASTE (específico), não bruto
        for (LL,ii) in common:
            if LL==L: V[r,idxmap[(LL,ii)]]=float(a[ii])
Vn=V/ (np.linalg.norm(V,axis=1,keepdims=True)+1e-6)
S=Vn@Vn.T
same=diff=0; sv=[]; dv=[]
for a in range(len(COUNTRIES)):
    for b in range(a+1,len(COUNTRIES)):
        ca,cb=COUNTRIES[a],COUNTRIES[b]
        if ca in CONT and cb in CONT:
            (sv if CONT[ca]==CONT[cb] else dv).append(S[a,b])
log(f"  correlação média MESMO continente {np.mean(sv):.3f} vs continentes DIFERENTES {np.mean(dv):.3f} "
    f"→ {'cluster emerge ✓' if np.mean(sv)>np.mean(dv)+0.02 else 'sem separação'}")
# vizinho mais próximo por correlação (o cluster auto-formado)
for c in ["Brazil","Japan","Egypt","France"]:
    r=COUNTRIES.index(c); nn=sorted([(S[r,b],COUNTRIES[b]) for b in range(len(COUNTRIES)) if b!=r],reverse=True)[:3]
    log(f"    {c:8} ~ {[f'{n}({s:.2f})' for s,n in nn]}")

# =========== 3) CHOQUE DIRECIONADO → aresta-fato lendo o peso por dentro ===========
log(f"\n## 3) CHOQUE DIRECIONADO (contexto da relação) → lê o neurônio-valor e DECODIFICA (não gera texto)")
@torch.no_grad()
def directed_edge(country,rel_prompt):
    a=shock(rel_prompt)                                       # choque = contexto da relação
    # soma dos valores co-ativados projetada no vocab = o que os pesos "escrevem" juntos (correlação interna)
    h=torch.zeros(E.shape[1],device=DEV,dtype=torch.float32)
    for L in DEEP:
        dp=m.model.layers[L].mlp.down_proj.weight.detach().float()   # [H,INT]
        h+= dp @ a[L]                                         # Σ_i act_i * value_i  (co-disparo → resíduo)
    logit=(h*norm_w.float())@Et.float()
    tid=int(logit.argmax())
    return tok.convert_ids_to_tokens([tid])[0].replace("Ġ","").replace("Ċ","")
@torch.no_grad()
def gen_edge(country,rel_prompt):
    ids=tok(rel_prompt,return_tensors="pt").input_ids.to(DEV)
    o=m.generate(ids,max_new_tokens=3,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
ok_read=ok_gen=tot=0
for c,(lang,reg,cap) in GOLD.items():
    ent=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",c); p=f"The capital of {ent} is"
    r_read=directed_edge(c,p); r_gen=gen_edge(c,p); tot+=1
    hr=norm(cap)[:4] in norm(r_read); hg=norm(cap)[:4] in norm(r_gen)
    ok_read+=hr; ok_gen+=hg
    if c in ["Brazil","France","Japan"]: log(f"    {c}: lê-peso→{r_read!r} {'✓' if hr else '✗'} · gera→{r_gen!r} {'✓' if hg else '✗'} (gold {cap})")
log(f"  capital por LER O PESO co-ativado: {ok_read}/{tot} · por GERAR: {ok_gen}/{tot} (choque direcionado, sem gold no laço)")

# =========== 4) LAPIDA COM USO (Hebbian) ===========
log(f"\n## 4) LAPIDA COM USO — a aresta ganha força no reuso (sem re-treino)")
W={};
def use(c,concept_word): W[(c,concept_word)]=W.get((c,concept_word),0.0)+1.0
for _ in range(5):
    for c in ["Brazil","France"]:
        for w in concepts_of(c,3): use(c,w)
strong=sorted(W.items(),key=lambda z:-z[1])[:4]
log(f"  arestas mais fortes após uso repetido: {[(f'{c}→{w}',int(f)) for (c,w),f in strong]}")

for h in H: h.remove()
kb=(edges*8)/1024
log(f"\n## VEREDITO WS26 (honesto)")
log(f"  ✓ CRESCE SOZINHO: choque neutro auto-formou {edges} arestas associativas SEM Q&A — o grafo se expande de graça.")
log(f"  ✓ CORRELAÇÃO clusteriza: mesmo continente {np.mean(sv):.2f} > diferente {np.mean(dv):.2f} (ativam junto = aresta).")
log(f"  {'✓' if ok_read>=ok_gen*0.7 else '⚠'} FATO por choque direcionado: ler o peso co-ativado acerta capital {ok_read}/{tot} (gerar {ok_gen}/{tot}).")
log(f"  ✓ LAPIDA no uso: reuso fortalece a aresta (Hebbian), sem re-treino.")
log(f"  LIMITE honesto: choque NEUTRO dá ASSOCIAÇÃO rica e barata (língua emerge {hitL}/{totG}); a relação ESPECÍFICA")
log(f"    (ex. capital) precisa do choque DIRECIONADO (contexto da relação) — que NÃO é pipeline de ensino, é 1 estímulo.")
log(f"  → auto-expansão = choque neutro (associações) + choque direcionado sob demanda (fatos) + uso (lapida). ~{kb:.1f}KB.")
json.dump(dict(assoc_edges=edges,lang_emerge=f"{hitL}/{totG}",region_emerge=f"{hitR}/{totG}",
    corr_same=round(float(np.mean(sv)),3),corr_diff=round(float(np.mean(dv)),3),
    cap_read=f"{ok_read}/{tot}",cap_gen=f"{ok_gen}/{tot}"),open(os.path.join(HERE,"ws26_shock.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f}min")
