#!/usr/bin/env python3
"""WS — PERCEPTRON À MÃO (Rosenblatt 1957) e a ponte com a IARA (2026-07-12).

Reproduz o que o Igor Venancio fez no caderno: roda a regra do perceptron epoch-a-epoch pra AND/OR
(convergem) e XOR (NÃO converge — Minsky-Papert 1969). Depois mostra a lição pra IARA: um SELETOR
linear sozinho tem teto (separabilidade linear); a COMPOSIÇÃO de seletores cruza o XOR — que é
EXATAMENTE o multi-hop do grafo da IARA (região ∩ língua = um AND de seletores). E a regra de update
(w += erro·x, só quando erra) É a dopamina=RPE: aprende só na surpresa. CPU, instantâneo, honesto."""
import numpy as np
X=np.array([[0,0],[1,0],[0,1],[1,1]],float)
def train(y,lr=1.0,epochs=30):
    w=np.zeros(3); Xb=np.hstack([np.ones((4,1)),X]); trace=[]
    for ep in range(1,epochs+1):
        errs=0
        for xb,yi in zip(Xb,y):
            pred=1 if xb@w>=0 else 0; e=yi-pred
            if e!=0: w=w+lr*e*xb; errs+=1          # SÓ atualiza no erro (=surpresa=dopamina)
        trace.append((ep,errs))
        if errs==0: return w,ep,trace
    return w,None,trace
def acc(w,y):
    Xb=np.hstack([np.ones((4,1)),X]); return np.mean([(1 if xb@w>=0 else 0)==yi for xb,yi in zip(Xb,y)])
def pred(w,x): return 1 if np.array([1,*x])@w>=0 else 0

print("="*64); print("# PERCEPTRON À MÃO — o SELETOR atômico da IARA"); print("="*64)
for name,y in [("AND",[0,0,0,1]),("OR",[0,1,1,1]),("NAND",[1,1,1,0])]:
    w,ep,tr=train(np.array(y,float))
    print(f"  {name:5} → {'convergiu' if ep else 'NÃO convergiu'} em epoch {ep} · pesos [w0={w[0]:.0f} w1={w[1]:.0f} w2={w[2]:.0f}] · acc {acc(w,np.array(y,float)):.0%}")

print("\n## XOR — o teto do seletor linear (Minsky-Papert 1969 → AI Winter)")
wx,epx,trx=train(np.array([0,1,1,0],float))
print(f"  perceptron ÚNICO no XOR → {'convergiu' if epx else 'NUNCA converge'} · melhor acc {acc(wx,np.array([0,1,1,0],float)):.0%} (fica preso, oscila)")
print(f"  (o erro por epoch nunca zera: {[e for _,e in trx[:8]]}…)")

print("\n## A LIÇÃO PRA IARA — COMPOSIÇÃO cruza o XOR (o 'MLP sem backprop')")
print("  XOR = (A OR B) AND (A NAND B) — dois seletores lineares compostos:")
wo,_,_=train(np.array([0,1,1,1],float)); wn,_,_=train(np.array([1,1,1,0],float))
def hidden(x): return [pred(wo,x),pred(wn,x)]              # camada escondida = 2 perceptrons
H=np.array([hidden(x) for x in X]);
wout=np.zeros(3); Xb=np.hstack([np.ones((4,1)),H])          # AND sobre as features escondidas
for _ in range(30):
    for xb,yi in zip(Xb,[0,1,1,0]):
        p=1 if xb@wout>=0 else 0; e=yi-p
        if e!=0: wout=wout+e*xb
xor_acc=np.mean([ (1 if np.array([1,*hidden(x)])@wout>=0 else 0)==t for x,t in zip(X,[0,1,1,0])])
print(f"  h1=OR, h2=NAND, saída=AND(h1,h2) → XOR resolvido, acc {xor_acc:.0%}")
print(f"  ISSO É o multi-hop da IARA: região ∩ língua = um AND de seletores → cruza o que 1 hop não cruza.")

print("\n## VEREDITO (ponte com a IARA)")
print("  1. PERCEPTRON = o seletor atômico. A IARA inteira é 'selector not generator' → o perceptron é o tijolo.")
print("  2. REGRA w+=erro·x SÓ no erro = DOPAMINA=RPE (aprende na surpresa). O perceptron valida os hormônios (1957!).")
print("  3. TETO XOR = por que roteamento linear único falha; a IARA COMPÕE (grafo multi-hop) = a camada extra sem backprop.")
print("  4. OBSERVÁVEL: 3 pesos + 1 reta desenhável → perfeito pro observatório (ver o neurônio aprender a linha).")
