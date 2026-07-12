#!/usr/bin/env python3
"""WS7 — UNIVERSALIDADE: o verificador (concorda→confia) é da ARQUITETURA ou do Instruct?

Roda o mesmo harness (85 capitais, geração aberta, território de escrita profunda) em
TRÊS geradores: Instruct (referência), Coder (fine-tune de outro domínio) e IARA-mini
(o esculpido do WS6, se existir). Se a separação concorda→alto / discorda→baixo replicar,
o cérebro acoplado é UNIVERSAL em cima de qualquer gerador da família — tese-mãe fortalecida."""
import torch, os, re, time, gc, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
BASE=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

CAPITAL={
 "Afghanistan":"Kabul","Albania":"Tirana","Algeria":"Algiers","Argentina":"Buenos Aires","Australia":"Canberra",
 "Austria":"Vienna","Bangladesh":"Dhaka","Belgium":"Brussels","Bolivia":"Sucre","Brazil":"Brasilia",
 "Bulgaria":"Sofia","Cambodia":"Phnom Penh","Cameroon":"Yaounde","Canada":"Ottawa","Chile":"Santiago",
 "China":"Beijing","Colombia":"Bogota","Croatia":"Zagreb","Cuba":"Havana","Czechia":"Prague",
 "Denmark":"Copenhagen","Ecuador":"Quito","Egypt":"Cairo","Ethiopia":"Addis Ababa","Finland":"Helsinki",
 "France":"Paris","Germany":"Berlin","Ghana":"Accra","Greece":"Athens","Hungary":"Budapest",
 "Iceland":"Reykjavik","India":"New Delhi","Indonesia":"Jakarta","Iran":"Tehran","Iraq":"Baghdad",
 "Ireland":"Dublin","Israel":"Jerusalem","Italy":"Rome","Japan":"Tokyo","Jordan":"Amman",
 "Kazakhstan":"Astana","Kenya":"Nairobi","Laos":"Vientiane","Lebanon":"Beirut","Libya":"Tripoli",
 "Malaysia":"Kuala Lumpur","Mexico":"Mexico City","Mongolia":"Ulaanbaatar","Morocco":"Rabat","Nepal":"Kathmandu",
 "Netherlands":"Amsterdam","Nigeria":"Abuja","Norway":"Oslo","Pakistan":"Islamabad","Paraguay":"Asuncion",
 "Peru":"Lima","Philippines":"Manila","Poland":"Warsaw","Portugal":"Lisbon","Qatar":"Doha",
 "Romania":"Bucharest","Senegal":"Dakar","Serbia":"Belgrade","Slovakia":"Bratislava","Spain":"Madrid",
 "Sweden":"Stockholm","Switzerland":"Bern","Syria":"Damascus","Thailand":"Bangkok","Tunisia":"Tunis",
 "Uganda":"Kampala","Ukraine":"Kyiv","Uruguay":"Montevideo","Uzbekistan":"Tashkent","Venezuela":"Caracas",
 "Vietnam":"Hanoi","Zimbabwe":"Harare","Angola":"Luanda","Armenia":"Yerevan","Belarus":"Minsk",
 "Estonia":"Tallinn","Latvia":"Riga","Lithuania":"Vilnius","Slovenia":"Ljubljana","Yemen":"Sanaa",
}
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(g,s): return norm(g) in norm(s) or norm(s).startswith(norm(g)[:max(3,len(norm(g))//2)])

def run_verifier(model, tok, label):
    NL=model.config.num_hidden_layers; DEEP=list(range(NL-12,NL))
    E=model.get_output_embeddings().weight.detach(); norm_w=model.model.norm.weight.detach()
    writes={}; handles=[]
    for L in DEEP:
        handles.append(model.model.layers[L].mlp.down_proj.register_forward_hook(
            (lambda L:(lambda m,i,o: writes.__setitem__(L,o.detach()[0,-1])))(L)))
    def fid(w): return tok.encode(" "+w,add_special_tokens=False)[0]
    cands=list({fid(a) for a in CAPITAL.values()}); cset=set(cands)
    @torch.no_grad()
    def gen(prompt,n=8):
        ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        logits=model(ids).logits[0,-1]; snap={L:writes[L].clone() for L in DEEP}
        first=int(logits.argmax()); out=[first]; cur=torch.cat([ids,torch.tensor([[first]],device=DEV)],1)
        for _ in range(n-1):
            nt=int(model(cur).logits[0,-1].argmax())
            if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
            out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
        return tok.decode(out).strip(), first, snap
    R=[]
    for c,cap in CAPITAL.items():
        g,first,snap=gen(f"The capital of {c} is")
        w=(sum(snap[L] for L in DEEP)*norm_w)
        terr=cands[int((E[cands]@w).argmax())]
        R.append(dict(ok=match(cap,g), agree=(terr==first and first in cset), memhit=(terr==fid(cap))))
    for h in handles: h.remove()
    a=[r for r in R if r["agree"]]; d=[r for r in R if not r["agree"]]
    acc=lambda rs: sum(r["ok"] for r in rs)/len(rs) if rs else float("nan")
    mem=sum(r["memhit"] for r in R)/len(R)
    log(f"  {label:<12} gerador {acc(R):.0%} · memória-neurônio {mem:.0%} · CONCORDA({len(a)})→{acc(a):.0%} · DISCORDA({len(d)})→{acc(d):.0%} · gap {acc(a)-acc(d):+.0%}")
    return acc(a)-acc(d)

log(f"\n{'='*72}\n# WS7 — UNIVERSALIDADE DO VERIFICADOR — {time.strftime('%H:%M')}\n{'='*72}")
gaps={}
for name,path in [("Instruct",f"{BASE}/Qwen2.5-1.5B-Instruct"),("Coder",f"{BASE}/Qwen2.5-Coder-1.5B"),("Math",f"{BASE}/Qwen2.5-Math-1.5B")]:
    tok=AutoTokenizer.from_pretrained(path)
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16).to(DEV).eval()
    gaps[name]=run_verifier(m,tok,name)
    del m; gc.collect(); torch.cuda.empty_cache()
MINI=f"{BASE}/iara-mini-v01"
if os.path.exists(os.path.join(MINI,"model.safetensors")):
    sys.path.insert(0,HERE)
    from iara_mini_loader import load_iara_mini
    tok=AutoTokenizer.from_pretrained(MINI)
    m=load_iara_mini(MINI,DEV)
    gaps["IARA-mini"]=run_verifier(m,tok,"IARA-mini")
    del m; gc.collect(); torch.cuda.empty_cache()
else:
    log("  IARA-mini ainda não existe (WS6 pendente) — rodar depois")
uni=sum(1 for g in gaps.values() if g>0.3)
log(f"VEREDITO WS7: separação replicou em {uni}/{len(gaps)} geradores — "
    f"{'UNIVERSAL (arquitetura, não acidente do Instruct)' if uni==len(gaps) else 'parcial — analisar quem falhou'}")
