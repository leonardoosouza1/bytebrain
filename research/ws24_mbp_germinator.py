#!/usr/bin/env python3
"""FASE 2 — GERMINADOR BYTE-NATIVO com MBP (Multi-Byte Prediction) — MTP→MBP do Leonardo (2026-07-12).

O salto de arquitetura: os órgãos falam BYTE. Germinador = transformer byte pequeno com K
CABEÇAS que preveem os PRÓXIMOS K BYTES de uma vez (MBP = análogo byte do Multi-Token
Prediction). Benefícios: (1) BYTE-nativo = robusto a typo (a tese-mãe), universal; (2) MBP =
prevê à frente (speculativo, mais rápido); (3) uma língua só entre os órgãos.
Treina em fatos (Q→A) + PT geral + AUGMENT de typo. Mede: perplexidade, MBP (bytes-à-frente
corretos), ROBUSTEZ a typo, e se germina fato. torch (GPU). Honesto."""
import torch, torch.nn as nn, os, re, time, random, json, math
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

# ---------- corpus: fatos (Q→A) + typo-augment + PT geral ----------
FACTS=[("Brazil","Brasilia"),("Argentina","Buenos Aires"),("Peru","Lima"),("Chile","Santiago"),
 ("France","Paris"),("Germany","Berlin"),("Spain","Madrid"),("Portugal","Lisbon"),("Italy","Rome"),
 ("Japan","Tokyo"),("China","Beijing"),("India","New Delhi"),("Egypt","Cairo"),("Kenya","Nairobi"),
 ("Canada","Ottawa"),("Mexico","Mexico City"),("Russia","Moscow"),("Poland","Warsaw"),("Greece","Athens"),
 ("Norway","Oslo"),("Sweden","Stockholm"),("Thailand","Bangkok"),("Vietnam","Hanoi"),("Nigeria","Abuja")]
def sentences():
    S=[]
    for c,cap in FACTS:
        S+=[f"The capital of {c} is {cap}.\n", f"Q: capital of {c}? A: {cap}.\n",
            f"{cap} is the capital of {c}.\n"]
    PT=os.path.join(HERE,"../data/pt_corpus.txt")
    if os.path.exists(PT):
        raw=open(PT,"rb").read(300000).decode("utf-8","ignore")
        S+= [s.strip()+"\n" for s in re.split(r"(?<=[.!?])\s+",raw) if 30<len(s.strip())<100][:200]
    return S
KEY={"a":"s","e":"r","i":"o","o":"p","r":"t","s":"d","t":"y","n":"m","l":"k","c":"x"}
def typo(w,r):
    if len(w)<4: return w
    i=r.randrange(1,len(w)-1); k=r.random()
    if k<0.5: return w[:i]+w[i+1]+w[i]+w[i+2:]          # swap
    return w[:i]+KEY.get(w[i].lower(),w[i])+w[i+1:]      # keyboard

SENTS=sentences()
# augment: perguntas com typo no país -> resposta certa (robustez)
rng=random.Random(7); AUG=[]
for c,cap in FACTS:
    for _ in range(3): AUG.append(f"Q: capital of {typo(c,rng)}? A: {cap}.\n")
CORPUS="".join(SENTS+AUG)
DATA=torch.tensor(list(CORPUS.encode("utf-8")),dtype=torch.long)
log(f"\n{'='*72}\n# FASE 2 — GERMINADOR BYTE MBP (Multi-Byte Prediction) — {time.strftime('%H:%M')}\n{'='*72}")
log(f"corpus: {len(SENTS)} frases + {len(AUG)} augment-typo = {len(DATA)/1024:.0f}KB de bytes")

# ---------- modelo: transformer byte com K cabeças MBP ----------
V,D,H,L,CTX,K = 256, 256, 4, 4, 96, 4
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1=nn.LayerNorm(D); s.ln2=nn.LayerNorm(D)
        s.attn=nn.MultiheadAttention(D,H,batch_first=True); s.mlp=nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D))
    def forward(s,x,mask):
        a,_=s.attn(s.ln1(x),s.ln1(x),s.ln1(x),attn_mask=mask,need_weights=False); x=x+a
        return x+s.mlp(s.ln2(x))
class MBP(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(V,D); s.pos=nn.Embedding(CTX,D)
        s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.lnf=nn.LayerNorm(D)
        s.heads=nn.ModuleList([nn.Linear(D,V) for _ in range(K)])   # K cabeças: byte t+1..t+K
    def forward(s,x):
        T=x.shape[1]; mask=torch.triu(torch.full((T,T),float('-inf'),device=x.device),1)
        h=s.emb(x)+s.pos(torch.arange(T,device=x.device))
        for b in s.blocks: h=b(h,mask)
        h=s.lnf(h)
        return [head(h) for head in s.heads]                        # lista de [B,T,V]

model=MBP().to(DEV); opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
nparams=sum(p.numel() for p in model.parameters()); log(f"MBP: {nparams/1e6:.1f}M params · {K} cabeças (prevê {K} bytes à frente) · ctx {CTX}")

def batch(bs=64):
    ix=torch.randint(0,len(DATA)-CTX-K,(bs,))
    x=torch.stack([DATA[i:i+CTX] for i in ix]).to(DEV)
    ys=[torch.stack([DATA[i+1+k:i+1+k+CTX] for i in ix]).to(DEV) for k in range(K)]  # alvo de cada cabeça
    return x,ys
t0=time.time()
for step in range(3500):
    x,ys=batch()
    outs=model(x); loss=sum(nn.functional.cross_entropy(outs[k].reshape(-1,V),ys[k].reshape(-1)) for k in range(K))/K
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1)%700==0: log(f"  passo {step+1}/3500 · loss {loss.item():.3f} · {time.time()-t0:.0f}s")

# ---------- avaliação ----------
model.eval()
@torch.no_grad()
def generate(prompt,n=40):
    ids=list(prompt.encode());
    for _ in range(n):
        x=torch.tensor(ids[-CTX:],device=DEV)[None]
        nb=int(model(x)[0][0,-1].argmax())                          # cabeça 0 = próximo byte
        ids.append(nb)
        if nb==10: break
    return bytes(ids).decode("utf-8","ignore")
@torch.no_grad()
def mbp_lookahead():
    """quantos bytes à frente as cabeças acertam (o ganho do MBP)?"""
    x,ys=batch(256); outs=model(x)
    acc=[float((outs[k].argmax(-1)==ys[k]).float().mean()) for k in range(K)]
    return acc

log(f"\n## AVALIAÇÃO do germinador byte MBP")
acc=mbp_lookahead()
log(f"  MBP lookahead (acerto por cabeça): " + " · ".join(f"byte+{k+1}={a:.0%}" for k,a in enumerate(acc)))
log(f"    → cabeça 0 (próximo byte) {acc[0]:.0%}; cabeças à frente ainda acertam {acc[-1]:.0%} = pode gerar {sum(1 for a in acc if a>0.5)} bytes/passo (speculativo)")
# germina fato LIMPO e com TYPO (robustez)
tests=[("Q: capital of Brazil? A:","Brasilia"),("Q: capital of Japan? A:","Tokyo"),("Q: capital of France? A:","Paris")]
tt=[("Q: capital of Brzil? A:","Brasilia"),("Q: capital of Japn? A:","Tokyo"),("Q: capital of Frnace? A:","Paris")]
def hit(g,gold): return gold.lower() in g.lower()
clean=sum(hit(generate(p),gd) for p,gd in tests); dirty=sum(hit(generate(p),gd) for p,gd in tt)
log(f"  germina fato LIMPO: {clean}/{len(tests)} · com TYPO no país: {dirty}/{len(tt)} (byte-nativo = robusto)")
for p,gd in tests[:2]: log(f"    {p!r} → {generate(p)[len(p):][:22]!r}")
for p,gd in tt[:2]: log(f"    (typo) {p!r} → {generate(p)[len(p):][:22]!r}")
torch.save(model.state_dict(),os.path.join(HERE,"iara_byte_germinator.pt"))
log(f"\nVEREDITO FASE 2: germinador BYTE-nativo com MBP treinado ({nparams/1e6:.1f}M) — prevê {sum(1 for a in acc if a>0.5)} bytes/passo, "
    f"germina fato {clean}/{len(tests)} limpo e {dirty}/{len(tt)} com TYPO (robustez byte). Salvo iara_byte_germinator.pt")
json.dump(dict(params=nparams,mbp_acc=acc,clean=f"{clean}/{len(tests)}",typo=f"{dirty}/{len(tt)}"),
    open(os.path.join(HERE,"ws24_mbp.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f}min")
