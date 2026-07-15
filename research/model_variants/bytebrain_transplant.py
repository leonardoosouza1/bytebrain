#!/usr/bin/env python3
"""TRANSPLANTE — inicializar o ByteBrain com o que o QWEN JÁ APRENDEU (não estrutura inventada). 2026-07-14.
Ideia do Leonardo: se o Qwen soube (0.76 bpb), pega o que ele aprendeu e JÁ DEIXA no byte-model, pra
quando treinar ficar melhor. Extrai os embeddings de BYTE do Qwen (aprendidos) → projeta (PCA) → inicializa
a ByteGPT do projeto. A/B: init aleatório vs init-transplante. Mede bpb no passo 0 (calcula o ganho) e a curva."""
import sys, os, math, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytebrain_scale import ByteGPT
DEV="cuda"; torch.manual_seed(0); np.random.seed(0); ID="Qwen/Qwen2.5-VL-3B-Instruct"; Dm=256

# ---------- 1) EXTRAIR o que o Qwen aprendeu sobre cada byte ----------
print("[extraindo embeddings de BYTE aprendidos pelo Qwen…]",flush=True)
from transformers import AutoTokenizer, AutoModelForImageTextToText
tk=AutoTokenizer.from_pretrained(ID)
qm=AutoModelForImageTextToText.from_pretrained(ID,dtype=torch.float16).to("cuda").eval()
E=qm.get_input_embeddings().weight.detach().float()             # [V, 2048] aprendido
byteE=torch.zeros(256,E.shape[1])
be=getattr(tk,"byte_encoder",None)
got=0
for b in range(256):
    tid=None
    if be is not None:
        c=be.get(b); tid=tk.convert_tokens_to_ids(c) if c is not None else None
    if tid is None or tid<0:
        ids=tk(bytes([b]).decode("latin-1"),add_special_tokens=False).input_ids; tid=ids[0] if ids else None
    if tid is not None and tid>=0: byteE[b]=E[tid].cpu(); got+=1
print(f"  bytes com embedding do Qwen: {got}/256")
# projeta 2048 -> Dm por PCA (mantém as direções mais informativas que o Qwen aprendeu)
X=byteE-byteE.mean(0,keepdim=True); U,S,V=torch.pca_lowrank(X,q=Dm,niter=4)
proj=(X@V[:,:Dm]); proj=proj/proj.std()*0.5                    # escala p/ init saudável
del qm, E; torch.cuda.empty_cache()
print(f"  variância explicada pelos {Dm} comps: {(S[:Dm].pow(2).sum()/X.pow(2).sum()).item():.1%}")

# ---------- 2) corpus PT ----------
raw=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt","rb").read(40_000_000)
arr=np.frombuffer(raw,np.uint8); n=len(arr); CTX,B=192,48; tr=arr[:int(n*0.95)]; va=arr[int(n*0.95):]
def batch(data):
    i=np.random.randint(0,len(data)-CTX-1,B); return torch.from_numpy(np.stack([data[j:j+CTX+1] for j in i]).astype(np.int64)).to(DEV)
@torch.no_grad()
def bpb(m,it=60):
    m.eval(); t=0.0
    for _ in range(it): x=batch(va); t+=F.cross_entropy(m(x[:,:-1]).reshape(-1,256),x[:,1:].reshape(-1)).item()
    m.train(); return t/it/math.log(2)

def run(nome,transplante,steps=1800):
    m=ByteGPT(4,Dm,4,emb="free").to(DEV)
    if transplante:
        with torch.no_grad(): m.tok.weight.copy_(proj.to(DEV))    # TRANSPLANTE: o byte-model começa com o que o Qwen aprendeu
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.01); t0=time.time(); m.train()
    print(f"\n### {nome}")
    print(f"  passo    0 (só init)  bpb_val {bpb(m):.3f}   <- o quanto o transplante JÁ SABE antes de treinar")
    for st in range(1,steps+1):
        x=batch(tr); loss=F.cross_entropy(m(x[:,:-1]).reshape(-1,256),x[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st%450==0: print(f"  passo {st:>4}          bpb_val {bpb(m):.3f}   ({time.time()-t0:.0f}s)")
    b=bpb(m,140); print(f"  -> FINAL {b:.3f}"); return b

print("\n"+"="*62+"\n# A/B: transplantar a base do Qwen ajuda o ByteBrain?\n"+"="*62)
r0=run("ALEATÓRIO (do zero)",False)
r1=run("TRANSPLANTE (init com embeddings do Qwen)",True)
print("\n"+"="*62)
print(f"PLACAR bpb final:  aleatório {r0:.3f}  ·  transplante {r1:.3f}  ({100*(r0-r1)/r0:+.1f}%)")
print("="*62)
