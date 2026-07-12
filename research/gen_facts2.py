#!/usr/bin/env python3
"""DISTILL FATOS DO QWEN (GPU, ampliado) — Qwen completa greedy prompts factuais; guarda a completude + confiança.
Saída = corpus PT de fatos verdadeiros (nas palavras do Qwen) pro byte-student treinar. Cobre o probe + amplo."""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda"; M="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-1.5B-Instruct"; OUT="/home/leonardo/projects/LLM/bytebrain/data/qwen_facts.txt"
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16).to(DEV).eval()
CAP={"do Brasil":"Brasil","da França":"França","de Portugal":"Portugal","da Argentina":"Argentina","do Japão":"Japão",
     "da Itália":"Itália","da Espanha":"Espanha","da Alemanha":"Alemanha","da Rússia":"Rússia","da China":"China",
     "do Egito":"Egito","do México":"México","do Canadá":"Canadá","do Chile":"Chile","do Peru":"Peru",
     "da Colômbia":"Colômbia","da Grécia":"Grécia","da Inglaterra":"Inglaterra","da Índia":"Índia","da Austrália":"Austrália",
     "do Uruguai":"Uruguai","da Bolívia":"Bolívia","do Paraguai":"Paraguai","dos Estados Unidos":"EUA"}
AUTOR={"Dom Casmurro":"", "O Cortiço":"", "Iracema":"", "Memórias Póstumas de Brás Cubas":"", "Vidas Secas":"", "Grande Sertão: Veredas":"", "Capitães da Areia":""}
SEEDS=[f"A capital {k} é a cidade de" for k in CAP]+[f"O livro {l} foi escrito por" for l in AUTOR]+[
 "A água é composta pelos elementos químicos","A fórmula química da água é","A fotossíntese é o processo pelo qual as plantas",
 "O maior planeta do sistema solar é","O menor planeta do sistema solar é","O planeta mais próximo do Sol é",
 "A velocidade da luz no vácuo é de aproximadamente","O Sol é uma estrela que fornece luz e",
 "O coração humano tem a função de bombear","A célula é a menor unidade de um ser",
 "O oxigênio é essencial para a respiração dos seres","A Terra leva aproximadamente 365 dias para",
 "O Brasil foi colonizado pelos","A língua oficial do Brasil é o","O maior oceano do planeta é o oceano",
 "O metal que enferruja quando exposto ao ar e à água é o","O gás que as plantas absorvem na fotossíntese é o",
 "A independência do Brasil foi proclamada no ano de","O primeiro presidente do Brasil foi",
 "A Revolução Francesa começou no ano de","A Segunda Guerra Mundial terminou no ano de",
 "O elemento químico de símbolo O é o","O elemento químico de símbolo Fe é o","O elemento químico de símbolo H é o",
 "O rio mais extenso do mundo é o","A montanha mais alta do mundo é o","O maior país do mundo em área é a",
 "A força que atrai os objetos para o centro da Terra é a","O processo de transformação de líquido em gás é a",
 "O número de continentes da Terra é","O satélite natural da Terra é a","A estrela mais próxima da Terra é o",
 "Os seres vivos são formados por unidades básicas chamadas","O sangue é bombeado pelo",
 "A capital dos Estados Unidos é a cidade de","A moeda oficial do Brasil é o","A moeda oficial dos Estados Unidos é o",
 "O autor da teoria da relatividade foi","O cientista que formulou as leis da gravitação foi",
 "A pintura Mona Lisa foi feita por","A obra Romeu e Julieta foi escrita por"]
@torch.no_grad()
def complete(prompt,n=26):
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV); confs=[]
    for _ in range(n):
        p=torch.softmax(m(ids).logits[0,-1].float(),-1); top=torch.topk(p,1); c=int(top.indices[0])
        if c==tok.eos_token_id: break
        confs.append(float(top.values[0])); ids=torch.cat([ids,torch.tensor([[c]],device=DEV)],1)
    txt=tok.decode(ids[0],skip_special_tokens=True); body=txt[len(prompt):]
    for e in [". ",".\n","\n"]:
        if e in body: body=body[:body.index(e)+1]; break
    return " ".join((prompt+body).split()), sum(confs)/max(1,len(confs))
rows=[]
for s in SEEDS:
    txt,conf=complete(s); rows.append((conf,txt)); print(f"[{conf:.2f}] {txt}",flush=True)
rows.sort(reverse=True); kept=[t for c,t in rows if c>=0.45]
open(OUT,"w").write("\n".join(t.strip() for t in kept)+"\n")
print(f"\nwrote {len(kept)}/{len(rows)} fatos (conf>=0.45) -> {OUT}\nDONE",flush=True)
