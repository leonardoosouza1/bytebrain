#!/usr/bin/env python3
"""IARA QUE CRESCE — agente pesquisador + bombardeio com curiosidade, aprendendo NOS PESOS. 2026-07-14.
Ciclo (ideia do Leonardo): perguntam algo → se ela NÃO sabe (closed-book) → pico de cortisol/dopamina
→ o professor (Claude, subscription) pesquisa a resposta → ela CONSOLIDA nos pesos (LoRA, gated por
dopamina, com replay do hipocampo p/ não esquecer) → e fica CURIOSA (pergunta sobre o que não conhece).
Mede a CURVA de aprendizado e a retenção enquanto escala o conhecimento."""
import torch,time,subprocess,json,re,random
from transformers import AutoProcessor,AutoModelForImageTextToText,AutoTokenizer
from peft import LoraConfig,get_peft_model
random.seed(0); ID="Qwen/Qwen2.5-VL-3B-Instruct"; CLAUDE="/home/leonardo/.local/bin/claude"
print("[carregando base herdada + LoRA (sinapses da IARA)…]",flush=True)
proc=AutoProcessor.from_pretrained(ID); tok=AutoTokenizer.from_pretrained(ID)
model=AutoModelForImageTextToText.from_pretrained(ID,dtype=torch.float16).to("cuda")
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.enable_input_require_grads(); model.config.use_cache=True
opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=8e-5)
HORM={"dopamina":0.2,"cortisol":0.1,"curiosidade":0.3}
REHEARSAL=[("Qual a capital da França?","A capital da França é Paris."),("Quanto é 2 mais 2?","2 mais 2 é igual a 4.")]

def ask(q,n=40):
    m=[{"role":"user","content":[{"type":"text","text":q}]}]; t=proc.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
    ins=proc(text=[t],return_tensors="pt").to("cuda")
    with torch.no_grad(): out=model.generate(**ins,max_new_tokens=n,do_sample=False)
    return " ".join(proc.batch_decode(out[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].split())
def batch(q,a):
    m=[{"role":"user","content":[{"type":"text","text":q}]}]; p=proc.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
    fids=tok(p+" "+a,return_tensors="pt").input_ids.to("cuda"); plen=tok(p,return_tensors="pt").input_ids.shape[1]
    lab=fids.clone(); lab[:,:plen]=-100; return fids,lab
def consolida(q,a,dopamina):
    model.gradient_checkpointing_enable(); model.config.use_cache=False
    ids,lab=batch(q,a); model.train()
    for i in range(30):
        for g in opt.param_groups: g["lr"]=8e-5*dopamina
        opt.zero_grad(); loss=model(input_ids=ids,labels=lab).loss; loss.backward(); opt.step()
        for rq,ra in ([REHEARSAL[-1]] if REHEARSAL else [])+ (random.sample(REHEARSAL,min(2,len(REHEARSAL))) if REHEARSAL else []):
            rids,rlab=batch(rq,ra); opt.zero_grad(); model(input_ids=rids,labels=rlab).loss.backward(); opt.step()  # replay forte (recente+aleatório)
        if loss.item()<0.25: break
    model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache=True; REHEARSAL.append((q,a))
    return i+1,loss.item()
def pesquisar(topico,n=6):                                    # AGENTE pesquisador = Claude (subscription)
    pr=(f"Você é um pesquisador. Gere {n} fatos VERDADEIROS e específicos sobre '{topico}', cada um como "
        f'par pergunta/resposta curto. Responda SÓ um JSON: [{{"q":"pergunta?","a":"resposta em 1 frase"}}]. Sem texto fora do JSON.')
    try:
        r=subprocess.run([CLAUDE,"-p",pr],capture_output=True,text=True,timeout=120).stdout
        j=re.search(r"\[.*\]",r,re.S); return json.loads(j.group(0))[:n] if j else []
    except Exception as e: print("  (pesquisa falhou:",str(e)[:50],")"); return []
def curiosidade(a):                                          # extrai algo que ela pode não conhecer p/ perguntar de volta
    ents=[w for w in re.findall(r"[A-ZÀ-Ý][a-zà-ÿ]{3,}",a)]
    return f"O que é {ents[-1]}?" if ents else None
def sabe(q,a):
    key=nrmkey(a); r=ask(q,24); return key in nrm(r), r
def nrm(s): return re.sub(r"[^a-z0-9 ]"," ",s.lower())
STOPK={"uma","para","como","pela","pelo","sobre","dois","tres","cerca","mais","seus","suas","onde","quando","muito","entre","aproximadamente","tipicamente","cada","pode","capaz","principalmente"}
def nrmkey(a):                                                # chave = nome próprio distintivo (senão a palavra mais longa)
    caps=[w for w in re.findall(r"\b[A-ZÀ-Ý][a-zà-ÿ]{3,}",a)]
    if caps: return nrm(max(caps,key=len))
    ws=[w for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}",a) if w.lower() not in STOPK]
    return nrm(max(ws,key=len)) if ws else nrm(a)

TOPICOS=["o Sol e as estrelas","os vulcões da Terra"]
aprendidos=[]
for topico in TOPICOS:
    print(f"\n{'='*70}\n# TÓPICO: {topico}\n{'='*70}")
    fatos=pesquisar(topico,6); print(f"  agente pesquisou {len(fatos)} fatos")
    for f in fatos:
        q,a=f.get("q","").strip(),f.get("a","").strip()
        if len(q)<5 or len(a)<5: continue
        knew,pre=sabe(q,a)
        if knew:
            HORM["dopamina"]=max(0.2,HORM["dopamina"]*0.9); print(f"  • já sabia: {q}")
            continue
        # NÃO sabia → cortisol/curiosidade sobem, dopamina alta (surpresa) → consolida
        HORM["cortisol"]=min(1,HORM["cortisol"]+0.3); HORM["curiosidade"]=min(1,HORM["curiosidade"]+0.2); HORM["dopamina"]=1.0
        ps,l=consolida(q,a,dopamina=HORM["dopamina"]); aprendidos.append((q,a))
        print(f"  🔥 não sabia → APRENDEU ({ps}p, loss {l:.2f}): {a[:60]}")
        HORM["dopamina"]=min(1,HORM["dopamina"]); HORM["cortisol"]*=0.6
        # CURIOSIDADE: pergunta sobre algo do que aprendeu
        cq=curiosidade(a)
        if cq and HORM["curiosidade"]>0.5:
            k2,_=sabe(cq,"?")
            if not k2:
                ca=pesquisar(cq,1)
                if ca:
                    a2=ca[0].get("a","").strip()
                    if len(a2)>5: ps2,_=consolida(cq,a2,dopamina=0.8); aprendidos.append((cq,a2)); print(f"     ❓curiosa: {cq} → aprendeu ({ps2}p): {a2[:50]}")
            HORM["curiosidade"]*=0.7
    # QUIZ closed-book do tópico
    ok=sum(1 for q,a in aprendidos[-6:] if nrmkey(a) in nrm(ask(q,24)))
    print(f"  ► quiz closed-book do tópico: {ok}/{min(6,len(aprendidos[-6:]))} · hormônios {({k:round(v,2) for k,v in HORM.items()})}")

# RETENÇÃO FINAL: sabe tudo que aprendeu? esqueceu as âncoras?
print(f"\n{'='*70}\n# RETENÇÃO FINAL (closed-book, tudo que ela aprendeu vivendo)\n{'='*70}")
ret=sum(1 for q,a in aprendidos if nrmkey(a) in nrm(ask(q,24)))
print(f"  reteve {ret}/{len(aprendidos)} fatos aprendidos ({100*ret/max(1,len(aprendidos)):.0f}%)")
for q,k in [("Qual a capital da França?","paris"),("Quanto é 2 mais 2?","4")]:
    r=ask(q,20); print(f"  âncora {'✅' if k in nrm(r) else '❌ ESQUECEU'}: {q} → {r[:40]}")
print(f"  VRAM pico {round(torch.cuda.max_memory_allocated()/1e9,2)}GB · total aprendido nos pesos: {len(aprendidos)} fatos")
