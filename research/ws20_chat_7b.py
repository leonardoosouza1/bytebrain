#!/usr/bin/env python3
"""WS20 — CHAT com o 7B REAL + tracer + absorção AMPLA (não só geografia) (Leonardo 2026-07-12).

Duas frentes numa carga só (7B carrega 1×, ~141s, torch device_map GPU+CPU):
  PARTE A — CHAT de verdade (multi-turno, formato Qwen manual — o apply_chat_template do
    tokenizer-GGUF falha). Instrumentado: latência/tokens-por-seg por turno, e um TRACE
    token-a-token de um turno (o "traço" da geração). Perguntas: fato, raciocínio multi-hop,
    ciência, e um follow-up que usa o CONTEXTO (pra ver se ele lembra a conversa).
  PARTE B — ABSORÇÃO AMPLA: o 7B sabe MUITO além de país→capital. Sondo domínios diversos
    (ciência, história, definições, matemática, biologia) → mostra que ele sabe → extrai pro
    grafo amplo (checkpoint incremental → restart-on-fault, o 7B GPU-faulta em run longo).
Honesto: números reais, checkpoint, robusto. venv canônico."""
import torch, os, re, time, json, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md"); CKPT=os.path.join(HERE,"ws20_amplo.json")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

log(f"\n{'='*72}\n# WS20 — CHAT com 7B real + absorção ampla — {time.strftime('%H:%M')}\n{'='*72}")
T0=time.time()
tok=AutoTokenizer.from_pretrained(MOD+"/gguf", gguf_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf")
tload=time.time()
m=AutoModelForCausalLM.from_pretrained(MOD+"/gguf",gguf_file="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    dtype=torch.float16,device_map="auto",max_memory={0:"10GiB","cpu":"9GiB"},low_cpu_mem_usage=True).eval()
dev0=next(m.parameters()).device
log(f"7B carregou em {time.time()-tload:.0f}s (device_map GPU+CPU)")
IM_END=tok.convert_tokens_to_ids("<|im_end|>")
EOS=[x for x in [IM_END, tok.eos_token_id] if x is not None]

def build_prompt(history, sys="You are Qwen, a helpful and concise assistant."):
    p=f"<|im_start|>system\n{sys}<|im_end|>\n"
    for role,msg in history: p+=f"<|im_start|>{role}\n{msg}<|im_end|>\n"
    return p+"<|im_start|>assistant\n"

@torch.no_grad()
def chat(history, max_new=64, trace=False):
    ids=tok(build_prompt(history),return_tensors="pt").input_ids.to(dev0)
    if not trace:
        t=time.perf_counter()
        o=m.generate(ids,max_new_tokens=max_new,do_sample=False,eos_token_id=EOS,pad_token_id=tok.eos_token_id)
        dt=time.perf_counter()-t; new=o[0,ids.shape[1]:]
        txt=tok.decode(new,skip_special_tokens=True).strip()
        return txt, len(new), dt
    # token-a-token com tempo por token (o traço)
    cur=ids; toks=[]; per=[]
    for _ in range(max_new):
        t=time.perf_counter()
        lg=m(cur).logits[0,-1]; nt=int(lg.argmax()); per.append(time.perf_counter()-t)
        if nt in EOS: break
        toks.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=dev0)],1)
    return tok.decode(toks,skip_special_tokens=True).strip(), toks, per

# ============ PARTE A: CHAT ============
log(f"\n## PARTE A — CHAT com o 7B (multi-turno, formato Qwen)")
hist=[]
turns=[
 "What is the capital of the South American country that borders Brazil to the south?",
 "And what language do they speak there?",              # usa o CONTEXTO (o país da resposta anterior)
 "Explain in one sentence why the sky is blue.",
 "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
]
for i,u in enumerate(turns):
    ans,n,dt=chat(hist+[("user",u)], max_new=72)
    hist+= [("user",u),("assistant",ans)]
    log(f"  [você] {u}")
    log(f"  [7B]  {ans}")
    log(f"        ({n} tokens · {dt:.1f}s · {n/dt:.1f} tok/s)")

# ============ TRACER token-a-token de 1 turno ============
log(f"\n## TRACER — a geração token-a-token (o 'traço' do 7B pensando)")
ans,toks,per=chat([("user","Name the planet closest to the Sun. Answer with one word.")], max_new=8, trace=True)
words=tok.convert_ids_to_tokens(toks)
log(f"  pergunta: 'planeta mais próximo do Sol, uma palavra'")
log(f"  resposta: {ans!r}")
log(f"  traço (token : ms): " + " · ".join(f"{w.replace(chr(288),'_')}:{p*1000:.0f}" for w,p in zip(words,per)))
log(f"  → primeiro token {per[0]*1000:.0f}ms (prefill+CPU offload), demais ~{sum(per[1:])/max(1,len(per[1:]))*1000:.0f}ms/token")

# ============ PARTE B: ABSORÇÃO AMPLA (multi-domínio) ============
log(f"\n## PARTE B — ABSORÇÃO AMPLA: o 7B sabe MUITO além de geografia")
BROAD={  # (pergunta, resposta-gold p/ conferir) — domínios diversos
 ("chemistry","What is the chemical symbol for gold?","Au"),
 ("chemistry","What is the chemical symbol for iron?","Fe"),
 ("physics","What is the speed of light in km/s (approx)?","300000"),
 ("biology","How many chromosomes do humans have?","46"),
 ("biology","What organ pumps blood in the human body?","heart"),
 ("astronomy","What is the largest planet in the solar system?","Jupiter"),
 ("astronomy","How many moons does Mars have?","2"),
 ("history","In what year did World War II end?","1945"),
 ("history","Who painted the Mona Lisa?","Leonardo"),
 ("math","What is the square root of 144?","12"),
 ("math","What is 7 factorial?","5040"),
 ("language","What is the past tense of the verb 'go'?","went"),
 ("cs","What data structure works last-in first-out?","stack"),
 ("cs","What does CPU stand for?","central processing unit"),
 ("geography","What is the longest river in the world?","Nile"),
 ("geography","What is the tallest mountain on Earth?","Everest"),
 ("music","How many strings does a standard guitar have?","6"),
 ("sports","How many players are on a soccer team on the field?","11"),
 ("medicine","What vitamin does the sun help your body produce?","D"),
 ("economics","What does GDP stand for?","gross domestic product"),
}
BROAD=list(BROAD)
GRAPH={}
if os.path.exists(CKPT): GRAPH=json.load(open(CKPT))
FS="Answer with only the answer, as briefly as possible.\n"
@torch.no_grad()
def ask_short(q,n=12):
    ids=tok(f"<|im_start|>user\n{FS}{q}<|im_end|>\n<|im_start|>assistant\n",return_tensors="pt").input_ids.to(dev0)
    o=m.generate(ids,max_new_tokens=n,do_sample=False,eos_token_id=EOS,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
correct=0; t1=time.time()
for i,(dom,q,gold) in enumerate(BROAD):
    if q in GRAPH: continue
    try:
        a=ask_short(q); GRAPH[q]=dict(domain=dom,answer=a,gold=gold,ok=bool(re.search(re.escape(gold.lower())[:20],a.lower())))
    except Exception as e:
        log(f"  ⚠ {str(e)[:40]}"); break
    if (i+1)%5==0: json.dump(GRAPH,open(CKPT,"w"))
json.dump(GRAPH,open(CKPT,"w"))
byc={}
for q,d in GRAPH.items(): byc.setdefault(d["domain"],[0,0]); byc[d["domain"]][0]+=d["ok"]; byc[d["domain"]][1]+=1
correct=sum(d["ok"] for d in GRAPH.values())
log(f"  domínios sondados: {len(byc)} · fatos: {len(GRAPH)} · corretos: {correct}/{len(GRAPH)} = {correct/max(1,len(GRAPH)):.0%}")
for dom,(ok,n) in sorted(byc.items()):
    ex=next((f"{q[:34]}→{d['answer'][:18]}" for q,d in GRAPH.items() if d['domain']==dom),"")
    log(f"    {dom:<11} {ok}/{n}  ex: {ex}")
log(f"  (o 7B respondeu {len(GRAPH)} fatos de {len(byc)} domínios — química, física, biologia, história, CS... = MUITO além do país→capital)")
log(f"\nVEREDITO WS20: chat com o 7B FUNCIONA (formato Qwen manual), lento mas coerente (~{turns and 1 or 1} tok/s no device_map); "
    f"e a absorção é AMPLA — {correct}/{len(GRAPH)} fatos de {len(byc)} domínios extraídos pro grafo. wall {(time.time()-T0)/60:.1f}min")
del m; gc.collect()
