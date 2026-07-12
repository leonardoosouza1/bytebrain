#!/usr/bin/env python3
"""WS10 — ÓRGÃO-BYTE v2: denoiser APRENDIDO (generaliza além do léxico?).

O órgão do WS1 (Levenshtein no léxico) recuperou typo 38→77%, mas por construção SÓ
conserta palavras QUE ESTÃO no léxico. HIPÓTESE v2: um modelo byte-a-byte PEQUENO,
treinado em pares (palavra-com-typo → palavra-limpa), aprende a REGRA de restauração e
conserta palavras QUE NUNCA VIU (held-out) — o que o léxico não pode fazer por definição.

DESENHO: GRU encoder-decoder em BYTES (~1.6M params), treinado em 12k palavras (PT+EN),
4 tipos de typo. HELD-OUT ESTRITO: 400 palavras + 30 nomes de países EXCLUÍDOS do treino.
Baselines: identidade (deixar o typo) = 0%; léxico-de-treino (não tem as held-out) = ~0%.
Métrica: restauração EXATA na held-out. Meta: ≥60% (regra aprendida, não memorização)."""
import torch, torch.nn as nn, os, re, time, random
HERE=os.path.dirname(os.path.abspath(__file__))
DEV="cuda"
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

log(f"\n{'='*72}\n# WS10 — ÓRGÃO-BYTE v2 (denoiser aprendido, held-out estrito) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()

# ---------------- dados: palavras PT (corpus real) + EN ----------------
PT=os.path.join(HERE,"../data/pt_corpus.txt")
words=set()
if os.path.exists(PT):
    raw=open(PT,"rb").read(3_000_000).decode("utf-8","ignore").lower()
    for w in re.findall(r"[a-zà-ú]{4,12}",raw):
        words.add(w)
        if len(words)>=14000: break
EN="the of capital city country located continent currency people always never question answer language history science number example government president mountain river north south east west republic kingdom island market street water light night morning".split()
words.update(EN)
COUNTRIES=["france","japan","germany","italy","spain","russia","egypt","canada","peru","greece",
 "portugal","austria","norway","poland","turkey","brazil","china","thailand","sweden","ireland",
 "kazakhstan","myanmar","bhutan","eritrea","mongolia","suriname","brunei","malawi","botswana","bolivia"]
words-=set(COUNTRIES)                       # países NUNCA vistos no treino (held-out real)
words=sorted(words)
rng=random.Random(7); rng.shuffle(words)
held_words=words[:400]; train_words=words[400:]
log(f"treino {len(train_words)} palavras · held-out {len(held_words)} palavras + {len(COUNTRIES)} países excluídos")

KEY={"a":"s","b":"v","c":"x","d":"s","e":"r","f":"g","g":"h","h":"j","i":"o","j":"k","k":"l","l":"k",
     "m":"n","n":"m","o":"p","p":"o","q":"w","r":"t","s":"d","t":"y","u":"i","v":"b","w":"e","x":"z","y":"t","z":"x"}
def typo(w,kind,r):
    i=r.randrange(1,max(2,len(w)-1))
    if kind==0 and len(w)>3: return w[:i]+w[i+1]+w[i]+w[i+2:]
    if kind==1: return w[:i]+w[i+1:]
    if kind==2: return w[:i]+KEY.get(w[i],w[i])+w[i+1:]
    return w[:i]+w[i]+w[i:]

MAX=14; SOS,EOS,PAD=1,2,0
def enc(s): return [min(253,ord(c))+3 if ord(c)<253 else 3 for c in s[:MAX]]
def pad(x,n): return x+[PAD]*(n-len(x))

class Denoiser(nn.Module):
    def __init__(s,h=256,e=64):
        super().__init__()
        s.emb=nn.Embedding(259,e); s.enc=nn.GRU(e,h,batch_first=True,bidirectional=True)
        s.dec=nn.GRU(e,h,batch_first=True); s.bridge=nn.Linear(2*h,h); s.out=nn.Linear(h,259)
    def forward(s,x,y_in):
        _,hn=s.enc(s.emb(x)); h=s.bridge(torch.cat([hn[0],hn[1]],-1)).unsqueeze(0)
        o,_=s.dec(s.emb(y_in),h); return s.out(o)
    @torch.no_grad()
    def fix(s,word):
        x=torch.tensor([pad(enc(word),MAX)],device=DEV)
        _,hn=s.enc(s.emb(x)); h=s.bridge(torch.cat([hn[0],hn[1]],-1)).unsqueeze(0)
        cur=torch.tensor([[SOS]],device=DEV); out=[]
        for _ in range(MAX):
            o,h=s.dec(s.emb(cur),h); nt=int(s.out(o[0,-1]).argmax())
            if nt==EOS: break
            out.append(nt); cur=torch.tensor([[nt]],device=DEV)
        return "".join(chr(t-3) for t in out if t>=3)

model=Denoiser().to(DEV)
opt=torch.optim.Adam(model.parameters(),lr=2e-3)
crit=nn.CrossEntropyLoss(ignore_index=PAD)
nparams=sum(p.numel() for p in model.parameters())
log(f"denoiser GRU bytes: {nparams/1e6:.1f}M params")

# treino: cada época = amostra de pares (typo, limpa) + 20% identidade (limpa→limpa: não estragar)
B=512; STEPS=2400
r=random.Random(11)
model.train()
for step in range(STEPS):
    xs=[]; yin=[]; yout=[]
    for _ in range(B):
        w=train_words[r.randrange(len(train_words))]
        src = w if r.random()<0.2 else typo(w,r.randrange(4),r)
        t=enc(w)
        xs.append(pad(enc(src),MAX)); yin.append(pad([SOS]+t,MAX+1)); yout.append(pad(t+[EOS],MAX+1))
    x=torch.tensor(xs,device=DEV); yi=torch.tensor(yin,device=DEV); yo=torch.tensor(yout,device=DEV)
    logits=model(x,yi); loss=crit(logits.reshape(-1,259),yo.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1)%600==0: log(f"  passo {step+1}/{STEPS} · loss {loss.item():.3f} · {time.time()-t0:.0f}s")

# ---------------- avaliação held-out ESTRITA ----------------
model.eval()
def eval_set(name,ws,seeds=(1,2,3)):
    tot=fixed=identity_broken=0
    for seed in seeds:
        rr=random.Random(seed)
        for w in ws:
            t=typo(w,rr.randrange(4),rr)
            if t==w: continue
            tot+=1; fixed+= (model.fix(t)==w)
    # controle: palavra LIMPA não pode ser estragada
    clean_ok=sum(model.fix(w)==w for w in ws)
    log(f"  {name:<28} restauração {fixed}/{tot} = {fixed/tot:.0%} · limpas preservadas {clean_ok}/{len(ws)} = {clean_ok/len(ws):.0%}")
    return fixed/tot
a=eval_set("held-out 400 palavras",held_words)
b=eval_set("held-out 30 PAÍSES (nunca vistos)",COUNTRIES)
log(f"  baseline identidade (deixar typo): 0% por definição · léxico-de-treino nas held-out: ~0% (não as contém)")
ok = a>=0.6
log(f"VEREDITO WS10: denoiser aprendido restaura {a:.0%} de palavras NUNCA VISTAS "
    f"({'REGRA APRENDIDA — órgão-byte v2 viável p/ IARA-3' if ok else 'abaixo de 60% — memorização/limite, registrar'}) · países {b:.0%}")
log(f"wall {(time.time()-t0)/60:.1f} min")
torch.save(model.state_dict(), os.path.join(HERE,"byte_denoiser_v2.pt"))
log("pesos salvos: bytebrain/research/byte_denoiser_v2.pt")
