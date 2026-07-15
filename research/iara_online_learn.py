#!/usr/bin/env python3
"""PROVA DE CONCEITO — a IARA aprende NOS PESOS (não só num grafo externo). 2026-07-14.
Ideia do Leonardo: os neurônios do Qwen são TRANSPLANTADOS pra IARA (base herdada) + uma
LoRA = as sinapses PRÓPRIAS dela, que o ensino AJUSTA. Aprendizado gated por DOPAMINA
(plasticidade modulada por recompensa): erro→surpresa alta→dopamina alta→passo maior→
consolida. Teste CLOSED-BOOK (sem contexto/retrieval): ela SABE de verdade depois? Esquece o resto?"""
import torch,time
from transformers import AutoProcessor,AutoModelForImageTextToText,AutoTokenizer
from peft import LoraConfig,get_peft_model
ID="Qwen/Qwen2.5-VL-3B-Instruct"
print("[carregando base herdada (Qwen2.5-VL-3B)…]",flush=True)
proc=AutoProcessor.from_pretrained(ID); tok=AutoTokenizer.from_pretrained(ID)
model=AutoModelForImageTextToText.from_pretrained(ID,dtype=torch.float16).to("cuda")
def ask(q,n=40):
    m=[{"role":"user","content":[{"type":"text","text":q}]}]
    t=proc.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
    ins=proc(text=[t],return_tensors="pt").to("cuda")
    with torch.no_grad(): out=model.generate(**ins,max_new_tokens=n,do_sample=False)
    return " ".join(proc.batch_decode(out[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].split())[:120]

# fatos INVENTADOS: o Qwen não pode saber → se ela responder depois, veio dos PESOS dela
FATOS=[("Qual a capital de Zúnia?","A capital de Zúnia é Morvek."),
       ("Do que é feita a liga chamada vurânio?","Vurânio é uma liga de cobre com o metal fictício zorel."),
       ("Quem descobriu o elemento flogisto-K?","O elemento flogisto-K foi descoberto pela cientista Ana Beluk.")]
CTRL=[("Qual a capital da França?","paris"),("Quanto é 7 vezes 8?","56")]

print("\n=== ANTES (closed-book, base herdada, sem treino) ===")
for q,_ in FATOS: print(f"  {q}\n    → {ask(q)}")
print("  [controle]");
for q,k in CTRL: print(f"  {q} → {ask(q)}")

# LoRA = as sinapses próprias da IARA (base fica de transplante; a LoRA é ajustada)
lcfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=get_peft_model(model,lcfg); model.print_trainable_parameters()
model.enable_input_require_grads(); model.gradient_checkpointing_enable(); model.config.use_cache=False
opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)

def batch(q,a):
    m=[{"role":"user","content":[{"type":"text","text":q}]}]
    p=proc.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
    fids=tok(p+a,return_tensors="pt").input_ids.to("cuda")
    plen=tok(p,return_tensors="pt").input_ids.shape[1]
    lab=fids.clone(); lab[:,:plen]=-100
    return fids,lab

# REHEARSAL = memória de consolidação (replay do hipocampo p/ não esquecer). Começa com âncoras conhecidas.
import random; random.seed(0)
REHEARSAL=[("Qual a capital da França?","A capital da França é Paris."),
           ("Quanto é 7 vezes 8?","7 vezes 8 é igual a 56."),
           ("Qual a capital do Brasil?","A capital do Brasil é Brasília.")]
def consolida(q,a,dopamina,maxpassos=30,alvo=0.35):           # aprende gentil + replay; para cedo (não martela até 0)
    ids,lab=batch(q,a); model.train(); t0=time.time(); l0=None
    for i in range(maxpassos):
        for g in opt.param_groups: g["lr"]=8e-5*dopamina       # LR ∝ dopamina
        opt.zero_grad(); loss=model(input_ids=ids,labels=lab).loss; loss.backward(); opt.step()
        if l0 is None: l0=loss.item()
        if REHEARSAL:                                          # 1 passo de REPLAY de um fato antigo (anti-esquecimento)
            rq,ra=random.choice(REHEARSAL); rids,rlab=batch(rq,ra)
            opt.zero_grad(); (model(input_ids=rids,labels=rlab).loss).backward(); opt.step()
        if loss.item()<alvo: break                            # consolidou o suficiente → para (evita sobrescrever)
    model.eval(); model.config.use_cache=True
    return l0,loss.item(),i+1,time.time()-t0

print("\n=== ENSINO (consolidação gentil + replay, dopamina alta) ===")
for q,a in FATOS:
    l0,l1,ps,dt=consolida(q,a,dopamina=1.0)
    REHEARSAL.append((q,a))                                    # o novo fato entra no replay (não some ao aprender o próximo)
    print(f"  ensinei ({ps} passos, {dt:.0f}s): {a[:55]}   loss {l0:.2f}→{l1:.2f}")

print("\n=== DEPOIS (closed-book — veio dos PESOS) ===")
ok=0
for q,a in FATOS:
    r=ask(q); key=a.split()[-1].strip(".").lower(); hit=key in r.lower(); ok+=hit
    print(f"  {'✅' if hit else '❌'} {q}  →  {r}")
print(f"  APRENDEU nos pesos: {ok}/{len(FATOS)}")
print("  [controle — reteve o que já sabia? (anti-esquecimento)]")
ret=0
for q,k in CTRL: r=ask(q); good=k in r.lower(); ret+=good; print(f"  {'✅' if good else '❌ ESQUECEU'} {q} → {r}")
# checa se o 1º inventado sobreviveu ao ensino do 3º
r=ask(FATOS[0][0]); print(f"  {'✅' if 'morvek' in r.lower() else '❌ ESQUECEU'} (fato-1 após aprender fato-3) {FATOS[0][0]} → {r}")
print(f"\nPLACAR: aprendeu {ok}/{len(FATOS)} · reteve controle {ret}/{len(CTRL)} · VRAM pico {round(torch.cuda.max_memory_allocated()/1e9,2)}GB")
