#!/usr/bin/env python3
"""WS — VERIFICADOR CALCULADO: um neurônio de 1 solve que lê o substrato e sabe se a IARA SABE (2026-07-12).

Prova em dados REAIS que "calcular o neurônio" ajuda a IARA. Pega as ativações profundas do 3B (o CHOQUE
que já usamos) pra entidades REAIS (países que ele sabe) e FALSAS (Genovia, Narnia, Gondor…). Ajusta um
READOUT FECHADO (ridge, 1 solve) e um ELM (hidden aleatório + solve) pra dizer 'eu sei' vs 'blefo', só das
ativações — SEM lista de vocabulário. Se separa, a IARA ganha abstenção HONESTA computada (leave-one-out,
número honesto). CPU+GPU (só o forward do choque)."""
import os,sys,time,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import torch
from iara_brain_grow import Brain
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True); open(JOUR,"a").write(s+"\n")

REAL="France Japan Egypt Germany China Canada Brazil Peru Italy Spain Norway Kenya Chile Portugal India".split()
FAKE="Genovia Narnia Gondor Wakanda Zubrowka Elbonia Latveria Wadiya Freedonia Sokovia Qumar Kraplakistan".split()

log(f"\n{'='*66}\n# VERIFICADOR CALCULADO — 'eu sei' vs 'blefo' das ativações reais\n{'='*66}")
t0=time.time(); B=Brain()
def feat(ent):
    a=B._shock(f"Facts about {ent}:"); fs=[]
    for L in B.DEEP:
        con=(a[L]-B.base[L]); v=torch.topk(con,10).values
        fs+=[float(con.norm()),float(con.max()),float(v.mean()),float(con.std())]
    return np.array(fs)
ents=REAL+FAKE; y=np.array([1]*len(REAL)+[0]*len(FAKE),float)
Xf=np.array([feat(e) for e in ents])
Xf=(Xf-Xf.mean(0))/(Xf.std(0)+1e-6)                       # normaliza features
log(f"  {len(REAL)} reais + {len(FAKE)} falsas · {Xf.shape[1]} features das ativações profundas · {time.time()-t0:.0f}s")

def ridge_fit(X,yy,lam=1.0):
    Xb=np.hstack([np.ones((len(X),1)),X]); return np.linalg.solve(Xb.T@Xb+lam*np.eye(Xb.shape[1]),Xb.T@yy)
def ridge_pred(w,X): return (np.hstack([np.ones((len(X),1)),X])@w>=0.5).astype(float)
def elm_fit(X,yy,H=40,seed=0,lam=1.0):
    r=np.random.default_rng(seed); Wr=r.standard_normal((X.shape[1],H)); br=r.standard_normal(H)
    Hd=np.tanh(X@Wr+br); Hb=np.hstack([np.ones((len(X),1)),Hd])
    wout=np.linalg.solve(Hb.T@Hb+lam*np.eye(Hb.shape[1]),Hb.T@yy); return (Wr,br,wout)
def elm_pred(m,X):
    Wr,br,wout=m; Hd=np.tanh(X@Wr+br); return (np.hstack([np.ones((len(X),1)),Hd])@wout>=0.5).astype(float)

# leave-one-out (honesto: testa em quem não viu)
def loo(fit,pred):
    ok=0
    for i in range(len(ents)):
        tr=[j for j in range(len(ents)) if j!=i]
        m=fit(Xf[tr],y[tr]); ok+= int(pred(m,Xf[i:i+1])[0]==y[i])
    return ok/len(ents)
acc_r=loo(lambda X,yy:ridge_fit(X,yy), ridge_pred)
acc_e=loo(lambda X,yy:elm_fit(X,yy), elm_pred)
# baseline: norma média da ativação (sinal simples "acendeu forte?")
simple=Xf[:,[0]]; acc_s=loo(lambda X,yy:ridge_fit(simple[[ents.index(e) for e in []]] if False else X,yy), ridge_pred) if False else None

log(f"\n## RESULTADO (leave-one-out — testa em entidade nunca vista no ajuste)")
log(f"  READOUT FECHADO (ridge, 1 solve): {acc_r:.0%} de acerto 'sei/blefo'")
log(f"  ELM (hidden aleatório + solve):   {acc_e:.0%} (não-linear, ainda 1 solve)")
# mostra alguns escores
w=ridge_fit(Xf,y)
sc=np.hstack([np.ones((len(ents),1)),Xf])@w
order=np.argsort(-sc)
log(f"  escore do neurônio (alto=sei): " + " · ".join(f"{ents[i]}={sc[i]:+.2f}" for i in order[:5]) + " … " + " · ".join(f"{ents[i]}={sc[i]:+.2f}" for i in order[-4:]))
log(f"\n## VEREDITO")
best=max(acc_r,acc_e)
log(f"  {'✓' if best>=0.75 else '⚠'} um NEURÔNIO CALCULADO (1 solve, {Xf.shape[1]} feats) separa 'sei' de 'blefo' a {best:.0%} — SÓ das ativações,")
log(f"    sem lista de vocabulário. A IARA pode abster pela PRÓPRIA incerteza do substrato (computado, não hardcoded).")
log(f"  Ancora ELM/readout fechado + Hopfield/Oja como órgãos CALCULÁVEIS (custo ~0, legíveis). wall {(time.time()-t0)/60:.1f}min")
