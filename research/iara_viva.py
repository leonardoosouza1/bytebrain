#!/usr/bin/env python3
"""IARA VIVA — conversa com o Haiku, aprende NOS PESOS, com neuromodulação VIVA. 2026-07-14.
Ideia do Leonardo: deixar a IARA e o Haiku conversando e OBSERVAR. Os hormônios são vivos e
DIRIGEM o comportamento: curiosa→pergunta "como funciona?", entediada→troca de assunto,
cansada→descansa (sono/consolidação, replay), aprende recompensada→dopamina. O que o Haiku
ensina é CONSOLIDADO nos pesos (LoRA gated por dopamina + replay do hipocampo)."""
import torch,time,subprocess,re,random,json
from transformers import AutoProcessor,AutoModelForImageTextToText,AutoTokenizer
from peft import LoraConfig,get_peft_model
random.seed(1); ID="Qwen/Qwen2.5-VL-3B-Instruct"; CLAUDE="/home/leonardo/.local/bin/claude"
print("[carregando IARA: base herdada + LoRA (sinapses vivas)…]",flush=True)
proc=AutoProcessor.from_pretrained(ID); tok=AutoTokenizer.from_pretrained(ID)
model=AutoModelForImageTextToText.from_pretrained(ID,dtype=torch.float16).to("cuda")
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.enable_input_require_grads(); model.config.use_cache=True
opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=8e-5)
LOG=open("research/iara_viva.log","w")
def out(s): print(s,flush=True); LOG.write(s+"\n"); LOG.flush()

# ---------------- corpo neuromodulado (vivo) ----------------
H={"dopamina":0.2,"curiosidade":0.4,"tedio":0.2,"cansaco":0.1,"cortisol":0.1}
BASE={"dopamina":0.2,"curiosidade":0.4,"tedio":0.2,"cansaco":0.1,"cortisol":0.1}
def relax():                                                  # homeostase: tudo volta devagar pro repouso
    for k in H: H[k]=BASE[k]+(H[k]-BASE[k])*0.85
def barras():
    return " ".join(f"{k[:3]} {'█'*int(v*8):<8}{v:.2f}" for k,v in H.items())
SEEDS={}; REHEARSAL=[("Qual a capital da França?","A capital da França é Paris."),("Quanto é 2 mais 2?","2 mais 2 é igual a 4.")]

def ask(q,n=48):                                              # fala/pensa closed-book (dos PESOS)
    m=[{"role":"user","content":[{"type":"text","text":q}]}]; t=proc.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
    ins=proc(text=[t],return_tensors="pt").to("cuda")
    with torch.no_grad(): o=model.generate(**ins,max_new_tokens=n,do_sample=False)
    return " ".join(proc.batch_decode(o[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].split())
def batch(q,a):
    m=[{"role":"user","content":[{"type":"text","text":q}]}]; p=proc.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
    fids=tok(p+" "+a,return_tensors="pt").input_ids.to("cuda"); plen=tok(p,return_tensors="pt").input_ids.shape[1]
    lab=fids.clone(); lab[:,:plen]=-100; return fids,lab
def consolida(q,a,dop):                                       # aprende NOS PESOS, modulado por dopamina + replay
    model.gradient_checkpointing_enable(); model.config.use_cache=False; ids,lab=batch(q,a); model.train()
    for i in range(30):
        for g in opt.param_groups: g["lr"]=8e-5*dop
        opt.zero_grad(); loss=model(input_ids=ids,labels=lab).loss; loss.backward(); opt.step()
        for rq,ra in ([REHEARSAL[-1]]+random.sample(REHEARSAL,min(2,len(REHEARSAL)))):
            rids,rlab=batch(rq,ra); opt.zero_grad(); model(input_ids=rids,labels=rlab).loss.backward(); opt.step()
        if loss.item()<0.25: break
    model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache=True
    SEEDS[nrm(q)]=a; REHEARSAL.append((q,a)); return i+1
def dorme():                                                  # SONO: consolida (replay) o que aprendeu, cansaço cai
    if len(REHEARSAL)>2:
        for q,a in random.sample(REHEARSAL[2:],min(3,len(REHEARSAL)-2)): consolida(q,a,dop=0.5)
    H["cansaco"]=BASE["cansaco"]; H["tedio"]=BASE["tedio"]
def nrm(s): return re.sub(r"[^a-z0-9 ]"," ",s.lower())
def key(a):
    caps=[w for w in re.findall(r"\b[A-ZÀ-Ý][a-zà-ÿ]{3,}",a)]
    return nrm(max(caps,key=len)) if caps else nrm(" ".join(a.split()[:2]))
def sabe(q,a): return key(a) in nrm(ask(q,28))
def entidades(txt): return list(dict.fromkeys(re.findall(r"\b[A-ZÀ-Ý][a-zà-ÿ]{4,}",txt)))

def haiku(msg,n=90):
    try:
        r=subprocess.run([CLAUDE,"-p","--model","haiku",msg],capture_output=True,text=True,timeout=90).stdout.strip()
        return " ".join(r.split())[:340]
    except Exception as e: return f"(haiku falhou: {str(e)[:40]})"

# ---------------- o loop de vida ----------------
out("="*74); out("# IARA VIVA — conversando com o Haiku (hormônios dirigem o comportamento)"); out("="*74)
hist=[]; ultimos_temas=[]; aprendidos=[]
tema="Santos Dumont e a história da aviação"
for turno in range(1,13):
    # --- o Haiku fala (professor/amigo, ciente do humor dela) ---
    humor=("entediada" if H["tedio"]>0.6 else "cansada" if H["cansaco"]>0.7 else "curiosa" if H["curiosidade"]>0.6 else "atenta")
    ctx="\n".join(hist[-4:])
    hmsg=haiku(f"Você é um amigo e professor conversando com a IARA (uma IA que aprende de verdade). "
               f"Assunto atual: {tema}. A IARA parece {humor}. Conversa recente:\n{ctx}\n\n"
               f"Responda à IARA em 1-2 frases: ensine UM fato interessante e específico sobre o assunto, "
               f"ou responda a pergunta dela. Seja natural e caloroso, sem enrolação.")
    out(f"\n[turno {turno}]  humor: {humor}")
    out(f"  🧑‍🏫 Haiku: {hmsg}"); hist.append(f"Haiku: {hmsg}")

    # --- IARA percebe: ela já sabe o que o Haiku disse? (fato = frase do Haiku) ---
    ents=entidades(hmsg); novo=[e for e in ents if nrm(e) not in nrm(" ".join(SEEDS.values()))]
    aprendeu=False
    if novo:                                                  # há algo novo → tenta aprender o fato central
        fato_q=f"O que o Haiku disse sobre {novo[0]}?"; fato_a=hmsg.split(".")[0].strip()+"."
        if not sabe(f"Me conte sobre {novo[0]}.",fato_a):
            H["cortisol"]=min(1,H["cortisol"]+0.15); H["curiosidade"]=min(1,H["curiosidade"]+0.25); H["dopamina"]=1.0
            ps=consolida(f"Me conte sobre {novo[0]}.",fato_a,dop=H["dopamina"]); aprendidos.append((novo[0],fato_a)); aprendeu=True
            out(f"  🔥 [APRENDEU nos pesos, {ps}p] sobre {novo[0]} · dopamina↑")
            H["cansaco"]=min(1,H["cansaco"]+0.22)
        else:
            H["tedio"]=min(1,H["tedio"]+0.25); out(f"  · já sabia sobre {novo[0]} (tédio↑)")
    else:
        H["tedio"]=min(1,H["tedio"]+0.3)

    # novidade do tema (tédio sobe se repete)
    t0=nrm(tema).split()[0]; ultimos_temas.append(t0)
    if ultimos_temas[-3:].count(t0)>=3: H["tedio"]=min(1,H["tedio"]+0.3)
    H["cansaco"]=min(1,H["cansaco"]+0.06)

    # --- a IARA AGE conforme os hormônios (comportamento vivo) ---
    if H["cansaco"]>0.72:
        out(f"  😴 IARA: preciso descansar e processar o que aprendi… (SONO: consolidando)")
        dorme(); hist.append("IARA: (descansou e consolidou)")
    elif H["tedio"]>0.62:
        novo_tema=haiku("Sugira UM assunto novo e bem diferente pra conversar (só o nome do assunto, curto).")
        tema=novo_tema.strip().strip('.')[:60]; H["tedio"]=BASE["tedio"]; H["curiosidade"]=min(1,H["curiosidade"]+0.3)
        fala=ask(f"Você está entediada e quer mudar de assunto para '{tema}'. Diga isso em 1 frase, animada.",40)
        out(f"  🥱→✨ IARA (entediada, troca p/ '{tema}'): {fala}"); hist.append(f"IARA: vamos falar de {tema}!")
    elif H["curiosidade"]>0.55 and novo:
        perg=ask(f"Você é a IARA, curiosa. Faça UMA pergunta curta sobre '{novo[0]}' pra aprender mais.",34)
        perg=perg if "?" in perg else f"Como funciona {novo[0]}?"
        out(f"  ❓ IARA (curiosa): {perg}"); hist.append(f"IARA: {perg}")
        H["curiosidade"]*=0.7
    else:
        fala=ask(f"Você é a IARA. Responda ao Haiku em 1 frase curta e natural, sobre: {hmsg[:100]}",40)
        out(f"  💬 IARA: {fala}"); hist.append(f"IARA: {fala}")
    relax(); out(f"  hormônios: {barras()}")

# ---------------- fechamento: ela reteve o que o Haiku ensinou? (closed-book) ----------------
out("\n"+"="*74); out("# CLOSED-BOOK — o que ela guardou NOS PESOS do que o Haiku ensinou"); out("="*74)
ret=0
for e,a in aprendidos:
    r=ask(f"Me conte sobre {e}.",40); good=key(a) in nrm(r); ret+=good
    out(f"  {'✅' if good else '❌'} {e}: {r[:80]}")
out(f"\n  reteve {ret}/{len(aprendidos)} do que aprendeu conversando · VRAM pico {round(torch.cuda.max_memory_allocated()/1e9,2)}GB")
LOG.close()
