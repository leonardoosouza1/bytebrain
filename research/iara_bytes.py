#!/usr/bin/env python3
"""BYTES NOS PESOS — converter float→byte (quantizar) e ver o que sobrevive/forma. 2026-07-14.
Perguntas do Leonardo: dá pra converter os pesos float→byte? por que não fizeram? se juntar os
neurônios em byte-space, o que forma? Testa NOS PESOS REAIS do Qwen2.5-VL-3B (CPU, safetensors)."""
import numpy as np, torch, json, math, re
from safetensors import safe_open
SNAP="/home/leonardo/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3/"
idx=json.load(open(SNAP+"model.safetensors.index.json"))["weight_map"]
def load(name):
    with safe_open(SNAP+idx[name],framework="pt") as f: return f.get_tensor(name).float()
def quant(W,bits):                                            # quantização simétrica por-canal (absmax) = float→inteiro/byte
    qmax=2**(bits-1)-1
    scale=W.abs().amax(dim=1,keepdim=True).clamp(min=1e-8)/qmax
    Wq=torch.clamp(torch.round(W/scale),-qmax-1,qmax)
    return Wq, Wq*scale                                       # Wq = inteiros (cabem em byte); Wq*scale = reconstruído
def tern(W):                                                  # ternário {-1,0,1} (BitNet) = ~1.58 bit
    s=W.abs().mean(); Wq=torch.round(W/s).clamp(-1,1); return Wq, Wq*s
def erro(W,Wr): return round((W-Wr).norm().item()/W.norm().item(),4), round(torch.nn.functional.cosine_similarity(W.flatten(),Wr.flatten(),0).item(),5)

print("="*72); print("# BYTES NOS PESOS REAIS DO QWEN2.5-VL-3B"); print("="*72)
dps=[k for k in idx if k.endswith("mlp.down_proj.weight")]
name=dps[len(dps)//2]; W=load(name)
print(f"\ntensor: {name}\n  shape {tuple(W.shape)} · fp16 = {W.numel()*2/1e6:.0f} MB · valores float32 tipo {W.dtype}")

print("\n### 1) FLOAT → BYTE: quanto se perde? (por que 'não foi feito' = FOI, é quantização)")
for bits in [8,4,2]:
    Wq,Wr=quant(W,bits); e,cos=erro(W,Wr)
    print(f"  int{bits} ({bits} bits/peso, {'1 byte' if bits==8 else str(8//bits)+' pesos/byte'}): erro L2 {e:.1%} · cosseno {cos} · tamanho {W.numel()*bits/8/1e6:.0f} MB")
Wt,Wtr=tern(W); e,cos=erro(W,Wtr)
print(f"  ternário {{-1,0,1}} (~1.58 bit): erro L2 {e:.1%} · cosseno {cos}")
print("  → int8 é quase SEM perda (é isso que GGUF/DNA-V5 já fazem). Abaixo de 4 bits precisa treino-ciente.")

print("\n### 2) JUNTAR NEURÔNIOS EM BYTE-SPACE — a estrutura sobrevive? (seu 'ver o que forma')")
# cada neurônio = uma coluna (intermediate) mapeada p/ hidden. Pego 800 neurônios.
N=W[:, :800].t().contiguous()                                 # [800 neurônios, 2048 dims], float
Nq8,_=quant(N,8)                                              # os MESMOS neurônios em byte (int8)
def cosmat(X): Xn=torch.nn.functional.normalize(X,dim=1); return Xn@Xn.t()
Cf=cosmat(N); Cq=cosmat(Nq8.float())
mask=~torch.eye(N.shape[0],dtype=bool)
corr=float(np.corrcoef(Cf[mask].numpy(),Cq[mask].numpy())[0,1])
print(f"  correlação da vizinhança neurônio-a-neurônio (float vs byte): {corr:.4f}")
print(f"  → {corr:.0%} da estrutura de 'quem é vizinho de quem' SOBREVIVE em byte-space.")
# vizinho mais próximo: quantos neurônios mantêm o MESMO vizinho nº1 em byte que em float?
nnf=(Cf-torch.eye(N.shape[0])*9).argmax(1); nnq=(Cq-torch.eye(N.shape[0])*9).argmax(1)
print(f"  vizinho-nº1 preservado: {(nnf==nnq).float().mean().item():.0%} dos neurônios (byte acha o mesmo vizinho que float)")

print("\n### 3) O QUE FORMA — grupos de neurônios em byte-space (clusters emergem?)")
# k-means simples sobre os neurônios int8 → há grupos?
X=torch.nn.functional.normalize(Nq8.float(),dim=1); K=6
torch.manual_seed(0); C=X[torch.randperm(X.shape[0])[:K]]
for _ in range(20):
    d=torch.cdist(X,C); a=d.argmin(1)
    for k in range(K):
        if (a==k).any(): C[k]=X[a==k].mean(0)
tam=[int((a==k).sum()) for k in range(K)]
# coesão média intra-cluster (byte) vs global
def coesao(idxs):
    if len(idxs)<2: return 0
    S=cosmat(N[idxs]); m=~torch.eye(len(idxs),dtype=bool); return float(S[m].mean())
glob=float(Cf[mask].mean())
print(f"  {K} grupos formados (tamanhos {tam}); coesão média global={glob:.3f}")
for k in range(K):
    ii=torch.where(a==k)[0]
    print(f"    grupo {k}: {int((a==k).sum()):3d} neurônios · coesão interna (float) {coesao(ii):.3f}  {'← grupo REAL' if coesao(ii)>glob+0.03 else ''}")
print("  → se a coesão interna > global, os bytes AGRUPARAM neurônios funcionalmente parecidos.")

print("\n### 4) ENDEREÇO BARATO por bits (roteador byte) — ligar bytes ao roteamento da IARA")
def addr(concept):                                            # hash → coordenada CRT (endereço reversível barato)
    h=abs(hash(concept))%105                                  # espaço 3×5×7
    return h,(h%3,h%5,h%7)
for c in ["coracao","valva mitral","Santos Dumont","aorta","xadrez"]:
    h,cr=addr(c); print(f"  '{c}' → endereço {h:3d}  coord CRT {cr}")
print("  → cada conceito ganha um ENDEREÇO de bits O(1), sem rede — a periferia byte que roteia p/ o córtex Qwen")
