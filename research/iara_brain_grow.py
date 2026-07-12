#!/usr/bin/env python3
"""IARA BRAIN GROW — cérebro que se AUTO-EXPANDE vivendo (choque+contraste+uso), sem pipeline de ensino.

A ideia do Leonardo, embutida no runtime. O 3B é o SUBSTRATO que vai sendo DRENADO pro grafo:
  CHOQUE NEUTRO   perceive(ent): estimula o substrato, lê o co-disparo, CONTRASTA vs a linha-base de
                  repouso (a lei do garimpo) → associações específicas de graça (Brazil→Amazon/Portuguese).
  CHOQUE DIRIGIDO learn_fact(ent,rel): sob demanda, 1 estímulo "The {rel} of {ent} is" (2 fraseados
                  concordam = cristaliza; discordam = incerto/abstém) → aresta-fato. Não é Q&A em lote.
  LAPIDA (uso)    Hebbian: sinapse fortalece no reuso; conceito NASCE por co-ativação.
Começa com 0 arestas. Vive um fluxo de perguntas. O grafo CRESCE sozinho, on-demand. Mede honesto:
cresceu? fatos certos (gold só p/ pontuar)? associações fazem sentido? custo (tempo/espaço)?"""
import torch, os, re, time, json, unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md"); MODEL=MOD+"/Qwen2.5-3B-Instruct"
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def norm(s): return re.sub(r"[^a-z0-9]","",strip_acc(s).lower())
def first_word(s):
    m=re.match(r"\s*([A-Za-z][A-Za-zÀ-ÿ]{1,18})",s); return m.group(1) if m else None

VOCAB=("Brazil Argentina Chile Peru Colombia Bolivia Paraguay Uruguay France Germany Spain Portugal Italy "
 "Poland Greece Sweden Norway Russia Japan China India Thailand Vietnam SouthKorea Turkey Iran Egypt Nigeria "
 "Kenya Morocco Ethiopia Canada Mexico Cuba Australia").split()
GOLD={"Peru":dict(capital="Lima",language="Spanish"),"France":dict(capital="Paris",language="French"),
 "Japan":dict(capital="Tokyo",language="Japanese"),"Egypt":dict(capital="Cairo",language="Arabic"),
 "Brazil":dict(capital="Bras",language="Portuguese"),"Germany":dict(capital="Berlin",language="German"),
 "China":dict(capital="Beijing",language="Chinese"),"Canada":dict(capital="Ottawa",language="English"),
 "Australia":dict(capital="Canberra",language="English"),"Portugal":dict(capital="Lisbon",language="Portuguese")}

class Brain:
    def __init__(self):
        self.tok=AutoTokenizer.from_pretrained(MODEL)
        self.m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(DEV).eval()
        NL=self.m.config.num_hidden_layers; self.INT=self.m.config.intermediate_size
        self.E=self.m.get_output_embeddings().weight.detach(); self.Et=self.E.t().contiguous()
        self.nw=self.m.model.norm.weight.detach(); self.DEEP=list(range(NL-10,NL))
        self.CAP={}
        for L in self.DEEP:
            self.m.model.layers[L].mlp.down_proj.register_forward_pre_hook(self._mk(L))
        # linha-base = o "país TÍPICO": média do template sobre países diversos, p/ os neurônios de
        # template E de "ser-um-país" cancelarem no contraste, sobrando só o específico da entidade (WS26).
        SEED=["Italy","India","Nigeria","Mexico","Sweden","Thailand","Chile","Turkey","Kenya","Poland","Spain","Vietnam"]
        self.base={L:torch.stack([self._shock(f"Facts about {s}:")[L] for s in SEED]).mean(0) for L in self.DEEP}
        self.G={}; self.SRC={}; self.ASSOC={}; self.W={}; self.born={}; self.dec={}; self.perceived=set()
        self.vec={}; self.growth=[]
    def _mk(self,L):
        def h(mod,inp): self.CAP[L]=inp[0][0,-1].detach().float()
        return h
    @torch.no_grad()
    def _shock(self,p):
        ids=self.tok(p,return_tensors="pt").input_ids.to(DEV); self.m(ids)
        return {L:self.CAP[L].clone() for L in self.DEEP}
    @torch.no_grad()
    def _gen(self,p,n=4):
        ids=self.tok(p,return_tensors="pt").input_ids.to(DEV)
        o=self.m.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
    def _decode(self,L,i):
        if (L,i) in self.dec: return self.dec[(L,i)]
        dp=self.m.model.layers[L].mlp.down_proj.weight.detach()
        v=(dp[:,i]*self.nw).to(self.E.dtype); logit=v@self.Et
        top=torch.topk(logit,4).indices
        ws=[self.tok.convert_ids_to_tokens([int(x)])[0].replace("Ġ","").replace("Ċ","") for x in top]
        clean=[w for w in ws if re.fullmatch(r"[A-Za-z][A-Za-z]{2,}",w)]
        self.dec[(L,i)]=clean[0] if clean else None; return self.dec[(L,i)]
    @torch.no_grad()
    def perceive(self,ent):
        """CHOQUE NEUTRO + CONTRASTE → associações. Leitura AGREGADA: soma os valores contrastados
        e projeta no vocabulário (junta a evidência de todos os neurônios, menos ruído que 1 a 1)."""
        a=self._shock(f"Facts about {ent}:")
        h=torch.zeros(self.E.shape[1],device=DEV,dtype=torch.float32)
        self.vec[ent]={}
        for L in self.DEEP:
            con=(a[L]-self.base[L])                                   # contraste
            dp=self.m.model.layers[L].mlp.down_proj.weight.detach().float()
            h+= dp @ con                                             # Σ_i contraste_i · valor_i
            self.vec[ent][L]=con                                     # guarda p/ correlação entidade↔entidade
        logit=(h*self.nw.float())@self.Et.float()
        top=torch.topk(logit,24).indices
        cs=[]
        for x in top:
            w=self.tok.convert_ids_to_tokens([int(x)])[0].replace("Ġ","").replace("Ċ","")
            if re.fullmatch(r"[A-Za-z]{4,}",w) and norm(w)!=norm(ent) and norm(w) not in {norm(z) for z in cs}:
                cs.append(w)
            if len(cs)>=8: break
        self.ASSOC[ent]=cs; self.perceived.add(ent)
        return cs
    def learn_fact(self,ent,rel):
        """CHOQUE DIRIGIDO sob demanda: 2 fraseados concordam = cristaliza (guarda contra alucinação)."""
        P={"capital":["The capital of {} is","The capital city of {} is"],
           "language":["The main language of {} is","People in {} mostly speak"],
           "currency":["The currency of {} is the","In {} people pay with the"]}.get(rel,
           [f"The {rel} of {{}} is",f"{{}}'s {rel} is"])
        a=first_word(self._gen(P[0].format(ent))); b=first_word(self._gen(P[1].format(ent)))
        if a and b and norm(a)==norm(b):
            self.G[(ent,rel)]=a; self.SRC[(ent,rel)]="alta"; return a,"alta"
        if a:                                                        # discordou: guarda como INCERTO (etiqueta honesta)
            self.G[(ent,rel)]=a; self.SRC[(ent,rel)]="incerto"; return a,"incerto"
        return None,"abstém"
    def _strengthen(self,ent,rel):
        self.W[(ent,rel)]=self.W.get((ent,rel),0)+1
        caps=[k for k in self.G if k[1]=="capital"]
        if rel=="capital" and len(caps)>=5 and "capitais" not in self.born:
            self.born["capitais"]={e:self.G[(e,'capital')] for (e,r) in caps}
    def answer(self,q):
        t=time.perf_counter(); ql=norm(q); trace=[]; grew=False
        ent=next((c for c in VOCAB if norm(c) in ql),None)
        rel=next((r for r in ["capital","language","currency"] if r in ql),None)
        if ent is None:
            return dict(ans="não sei (entidade fora do vocabulário perceptível)",conf="abstém",trace=trace,
                        ms=(time.perf_counter()-t)*1e3,grew=False)
        if ent not in self.perceived:
            assoc=self.perceive(ent); grew=True; trace.append(f"choque neutro → percebe {ent} → assoc {assoc[:4]}")
        if rel:
            conf="alta"
            if (ent,rel) not in self.G:
                v,conf=self.learn_fact(ent,rel); grew=True
                trace.append(f"choque dirigido → aprende {ent}|{rel} = {v} ({conf}) [cristaliza]")
                if v is None:
                    return dict(ans="não sei",conf="abstém",trace=trace,ms=(time.perf_counter()-t)*1e3,grew=grew)
            else:
                conf=self.SRC.get((ent,rel),"alta")
            v=self.G.get((ent,rel)); self._strengthen(ent,rel)
            trace.append(f"grafo[{ent}|{rel}]={v} (força {self.W[(ent,rel)]})")
            return dict(ans=v,conf=conf,trace=trace,ms=(time.perf_counter()-t)*1e3,grew=grew)
        return dict(ans=f"conheço {ent}: {self.ASSOC.get(ent,[])[:5]}",conf="assoc",trace=trace,
                    ms=(time.perf_counter()-t)*1e3,grew=grew)

if __name__=="__main__":
    log(f"\n{'='*72}\n# IARA BRAIN GROW — cérebro que se AUTO-EXPANDE vivendo (sem pipeline de ensino) — {time.strftime('%H:%M')}\n{'='*72}")
    t0=time.time(); B=Brain()
    log(f"cérebro nasce com {len(B.G)} arestas e {len(B.ASSOC)} associações (VAZIO). substrato 3B pronto p/ ser drenado. {time.time()-t0:.0f}s")
    STREAM=["capital of Peru?","language of Peru?","capital of France?","capital of Japan?","language of France?",
     "capital of Egypt?","capital of Germany?","capital of Brazil?","language of Japan?","capital of China?",
     "capital of Canada?","capital of Australia?","language of Brazil?","capital of Portugal?","capital of Peru?"]
    log(f"\n## VIVENDO {len(STREAM)} interações — o grafo cresce sozinho (0 → ?)")
    for i,q in enumerate(STREAM):
        r=B.answer(q); mark="+" if r["grew"] else "·"
        B.growth.append(len(B.G))
        if i<8 or r["grew"]: log(f"  {mark} '{q}' -> {r['ans']} [{r['conf']}, {r['ms']:.0f}ms] · grafo={len(B.G)}")
    log(f"  curva de crescimento (arestas após cada interação): {B.growth}")
    tot=cor=0
    for e,g in GOLD.items():
        for rel,gv in g.items():
            if (e,rel) in B.G:
                tot+=1; cor+= norm(gv)[:4] in norm(B.G[(e,rel)]) or norm(B.G[(e,rel)]) in norm(gv)
    log(f"\n## MEDIDA (honesta)")
    log(f"  grafo cresceu 0 -> {len(B.G)} fatos + {sum(len(v) for v in B.ASSOC.values())} associações, SEM batch de ensino")
    log(f"  fatos APRENDIDOS corretos (gold): {cor}/{tot} = {cor/max(1,tot):.0%}")
    log(f"  associações auto-formadas (exemplos):")
    for e in ["Brazil","France","Japan"]:
        if e in B.ASSOC: log(f"    {e:8} -> {B.ASSOC[e][:6]}")
    log(f"  conceito nascido por co-ativação: {list(B.born)} ({len(B.born.get('capitais',{}))} capitais no bundle)")
    reuse=[k for k,w in B.W.items() if w>1]
    log(f"  lapidado no uso (força>1): {reuse[:5]}")
    kb=(len(B.G)*8 + sum(len(v) for v in B.ASSOC.values())*8)/1024
    log(f"\n## VEREDITO iara_brain_grow")
    log(f"  AUTO-EXPANDE vivendo: 0 -> {len(B.G)} fatos on-demand, choque+contraste, {cor/max(1,tot):.0%} corretos, ~{kb:.1f}KB")
    log(f"  associações de graça (choque neutro) + fato sob demanda (choque dirigido) + uso lapida — SEM pipeline de ensino")
    json.dump(dict(facts=len(B.G),assoc=sum(len(v) for v in B.ASSOC.values()),acc=f"{cor}/{tot}",
        growth=B.growth,born=len(B.born.get('capitais',{})),kb=round(kb,1)),open(os.path.join(HERE,"iara_brain_grow.json"),"w"),indent=1)
    json.dump({"graph":{f"{e}|{r}":v for (e,r),v in B.G.items()},"assoc":B.ASSOC},open(os.path.join(HERE,"iara_grown_graph.json"),"w"))
    log(f"wall {(time.time()-t0)/60:.1f}min · grafo salvo em iara_grown_graph.json")
