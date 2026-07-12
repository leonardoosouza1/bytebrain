#!/usr/bin/env python3
"""DISTILL LIMPO — Qwen-Instruct pelo CHAT TEMPLATE (perguntar, não completar). Gera fatos em frase curta,
formato afirmativo (pro completion do byte-student) + formato P:/R: (pro Q&A). Filtra lixo (': A', 'A)', vazio)."""
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV="cuda"; M="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-1.5B-Instruct"
OUT="/home/leonardo/projects/LLM/bytebrain/data/qwen_facts.txt"
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16).to(DEV).eval()
SYS="Você é um professor. Responda em UMA frase curta, afirmativa e factual, começando repetindo o sujeito. Ex: 'A capital da França é Paris.' Não use listas nem alternativas."
QS=[("Qual é a capital do Brasil?","A capital do Brasil é"),("Qual é a capital da França?","A capital da França é"),
 ("Qual é a capital do Japão?","A capital do Japão é"),("Qual é a capital da Itália?","A capital da Itália é"),
 ("Qual é a capital de Portugal?","A capital de Portugal é"),("Qual é a capital da Alemanha?","A capital da Alemanha é"),
 ("Qual é a capital da Argentina?","A capital da Argentina é"),("Qual é a capital dos Estados Unidos?","A capital dos Estados Unidos é"),
 ("Qual é a capital da Rússia?","A capital da Rússia é"),("Qual é a capital da Espanha?","A capital da Espanha é"),
 ("Qual é a fórmula química da água?","A fórmula da água é"),("Qual é o maior planeta do sistema solar?","O maior planeta do sistema solar é"),
 ("Qual é o planeta mais próximo do Sol?","O planeta mais próximo do Sol é"),("Qual é a velocidade da luz no vácuo?","A velocidade da luz é de aproximadamente"),
 ("O que é fotossíntese?","A fotossíntese é"),("O que é uma célula?","A célula é"),("Qual é a função do coração?","O coração serve para"),
 ("Em que ano foi proclamada a independência do Brasil?","A independência do Brasil foi proclamada em"),
 ("Em que ano terminou a Segunda Guerra Mundial?","A Segunda Guerra Mundial terminou em"),
 ("Quem escreveu Dom Casmurro?","Dom Casmurro foi escrito por"),("Quem escreveu Vidas Secas?","Vidas Secas foi escrito por"),
 ("Quem pintou a Mona Lisa?","A Mona Lisa foi pintada por"),("Quem criou a teoria da relatividade?","A teoria da relatividade foi criada por"),
 ("Qual é o maior oceano do mundo?","O maior oceano do mundo é"),("Qual é o rio mais extenso do mundo?","O rio mais extenso do mundo é"),
 ("Qual é o satélite natural da Terra?","O satélite natural da Terra é"),("Qual é a língua oficial do Brasil?","A língua oficial do Brasil é"),
 ("Qual é a moeda oficial do Brasil?","A moeda oficial do Brasil é"),("Qual é o elemento de símbolo Fe?","O elemento de símbolo Fe é"),
 ("Quantos continentes existem na Terra?","O número de continentes é")]
@torch.no_grad()
def ask(q):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":q}]
    enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True)
    enc={k:v.to(DEV) for k,v in enc.items()}; L=enc["input_ids"].shape[1]
    out=m.generate(**enc,max_new_tokens=40,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,L:],skip_special_tokens=True).strip()
def clean(a):
    a=" ".join(a.split()); a=a.split("\n")[0]
    if re.search(r":\s*[A-D]\)|: A\.?$|\b[A-D]\)\s",a) or len(a)<8: return None
    if "." in a: a=a[:a.index(".")+1]
    return a if len(a)>=8 else None
rows=[]
for q,stem in QS:
    a=ask(q); c=clean(a)
    print(f"{'✓' if c else '✗'} {q} -> {a[:70]!r}",flush=True)
    if c: rows.append((stem,c))
with open(OUT,"w") as f:
    for stem,c in rows:
        f.write(c.strip()+"\n")                         # afirmativo (do Qwen)
        f.write(f"P: {[q for q,s in QS if s==stem][0]}\nR: {c.strip()}\n")   # formato Q&A
print(f"\nwrote {len(rows)}/{len(QS)} fatos limpos -> {OUT}\nDONE",flush=True)
