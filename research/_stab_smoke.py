import time,math,random,torch,torch.nn as nn,torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
M="llm-lab/models/Qwen2.5-Math-1.5B"; DEV="cuda"; t0=time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H=model.config.hidden_size; EL=model.get_input_embeddings()
POOL=[]
for tid in range(min(len(tok),60000)):
    s=tok.decode([tid])
    if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3<=len(s.strip())<=8 and len(tok(s,add_special_tokens=False).input_ids)==1: POOL.append(s)
    if len(POOL)>=80: break
def facts(n,seed):
    r=random.Random(seed); tpls=["O código secreto número {i} é","O nome do robô {i} é","A cor do domínio {i} é","A senha do cofre {i} é","O valor da chave {i} é"]
    return [(tpls[i%5].format(i=i),POOL[r.randrange(len(POOL))]) for i in range(n)]
def batch(P,K):
    N=len(P); ml=K+max(len(tok(p).input_ids)+1 for p,_ in P)
    E=torch.zeros(N,ml,H,device=DEV,dtype=torch.float16); am=torch.zeros(N,ml,device=DEV,dtype=torch.long)
    pos=torch.zeros(N,dtype=torch.long,device=DEV); tgt=torch.zeros(N,dtype=torch.long,device=DEV)
    for i,(p,tg) in enumerate(P):
        pi=tok(p).input_ids; ti=tok(tg,add_special_tokens=False).input_ids[:1]
        pe=EL(torch.tensor([pi],device=DEV)).detach()[0]; te=EL(torch.tensor([ti],device=DEV)).detach()[0]
        L=K+len(pi)+1; E[i,K:K+len(pi)]=pe; E[i,K+len(pi):L]=te; am[i,:L]=1; pos[i]=K+len(pi)-1; tgt[i]=ti[0]
    return E,am,pos,tgt
def run(sd,E,am): X=E.clone(); X[:,:sd.shape[0]]=sd.to(torch.float16); return model(inputs_embeds=X,attention_mask=am).logits
def plant(P,K,lr,steps,clip,tseed):
    E,am,pos,tgt=batch(P,K); ar=torch.arange(len(P),device=DEV)
    g=torch.Generator(device=DEV).manual_seed(tseed)
    sd=nn.Parameter(torch.randn(K,H,generator=g,device=DEV,dtype=torch.float32)*0.1); opt=torch.optim.AdamW([sd],lr=lr)
    for s in range(steps):
        for gr in opt.param_groups: gr['lr']=lr*0.5*(1+math.cos(math.pi*s/steps))
        lg=run(sd,E,am)[ar,pos]; loss=F.cross_entropy(lg.float(),tgt)
        opt.zero_grad(); loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_([sd],clip)
        opt.step()
    return sd.detach().to(torch.float16),E,am,pos,tgt,loss.item()
def rec(sd,E,am,pos,tgt): ar=torch.arange(len(pos),device=DEV); return int((run(sd,E,am)[ar,pos].argmax(-1)==tgt).sum())
def qi4(sd): qm=7; sc=(sd.abs().max()/qm).clamp_min(1e-8); return ((sd.float()/sc).round().clamp(-qm,qm)*sc).to(torch.float16)
log(f"pool {len(POOL)} | START")
P20=facts(20,1)
for tag,lr,steps,clip in [("lr0.1/800/clip1+cos",0.1,800,1.0),("lr0.05/1200/clip1+cos",0.05,1200,1.0)]:
    a=plant(P20,1,lr,steps,clip,111); b=plant(P20,1,lr,steps,clip,222); k4=plant(P20,4,lr,steps,clip,111)
    log(f"[{tag:22}] K1 repA {rec(*a[:5])}/20 repB {rec(*b[:5])}/20 (loss {a[5]:.3f}/{b[5]:.3f}) | K4 fp16 {rec(*k4[:5])}/20 int4 {rec(qi4(k4[0]),*k4[1:5])}/20")
log("DONE_STAB")
