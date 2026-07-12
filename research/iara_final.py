#!/usr/bin/env python3
"""IARA FINAL — leve, inteligente, rápida, byte-nativa, órgãos alinhados (2026-07-12).

Fecha o PROGRAMA. Junta tudo o que validamos num runtime ÚNICO:
  MEMÓRIA     = grafo grande semeado do modelo (FASE 1, 139 arestas 100% corretas), KB, instantâneo.
  ÓRGÃO-BYTE  = corrige TYPO da entrada por distância de edição em BYTES contra o vocabulário do
                grafo — robustez byte-nativa cobrindo TODAS as 139 entidades, SEM modelo pesado.
  GERMINADOR  = transformer BYTE com MBP (FASE 2, 3.5M, 4 bytes/passo) — geração byte-nativa e
                germina o que falta sob demanda; cristaliza a aresta (semente).
  VERIFICADOR = confiança/abstenção — não alucina em entidade fora do que foi semeado.
  NAVEGAÇÃO   = água: direto · agregado (aresta inversa) · multi-hop/cross (interseção de relações).
  APRENDE     = sinapse fortalece no reuso; conceito NASCE por co-ativação (Hebbian).
Conserta o bug do multi-hop (Bras/Brasília) com normalização de ACENTO correta (unicode NFKD).
Bateria de capacidade + inteligência racional + ROBUSTEZ A TYPO (a nova força byte). Honesto."""
import torch, torch.nn as nn, os, re, time, json, sys, unicodedata
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")
def strip_acc(s):                                     # Brasília -> Brasilia (conserta o match)
    return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def norm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()
def sub(a,b): na,nb=norm(a),norm(b); return na and nb and (na in nb or nb in na)   # match tolerante
def edit(a,b):                                        # distância de edição (bytes) — órgão-byte
    a,b=a.encode(),b.encode()
    if abs(len(a)-len(b))>3: return 99
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

# ---------- germinador byte-MBP (FASE 2) ----------
V,D,H,L,CTX,K = 256, 256, 4, 4, 96, 4
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1=nn.LayerNorm(D); s.ln2=nn.LayerNorm(D)
        s.attn=nn.MultiheadAttention(D,H,batch_first=True); s.mlp=nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D))
    def forward(s,x,mask):
        a,_=s.attn(s.ln1(x),s.ln1(x),s.ln1(x),attn_mask=mask,need_weights=False); x=x+a
        return x+s.mlp(s.ln2(x))
class MBP(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(V,D); s.pos=nn.Embedding(CTX,D)
        s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.lnf=nn.LayerNorm(D)
        s.heads=nn.ModuleList([nn.Linear(D,V) for _ in range(K)])
    def forward(s,x):
        T=x.shape[1]; mask=torch.triu(torch.full((T,T),float('-inf'),device=x.device),1)
        h=s.emb(x)+s.pos(torch.arange(T,device=x.device))
        for b in s.blocks: h=b(h,mask)
        h=s.lnf(h)
        return [head(h) for head in s.heads]

# BACKBONE GEOGRÁFICO objetivo (continente/região são fato-terra, não opinião do modelo;
# as CAPITAIS/língua/moeda é que vieram do 3B por auto-consistência). src="geo".
GEO={  # país: (continente, região)
 "Brazil":("America","South America"),"Argentina":("America","South America"),"Chile":("America","South America"),
 "Peru":("America","South America"),"Colombia":("America","South America"),"Bolivia":("America","South America"),
 "Uruguay":("America","South America"),"Paraguay":("America","South America"),"Ecuador":("America","South America"),
 "Venezuela":("America","South America"),
 "France":("Europe","Western Europe"),"Germany":("Europe","Western Europe"),"Spain":("Europe","Western Europe"),
 "Portugal":("Europe","Southern Europe"),"Italy":("Europe","Southern Europe"),"Poland":("Europe","Eastern Europe"),
 "Greece":("Europe","Southern Europe"),"Netherlands":("Europe","Western Europe"),"Belgium":("Europe","Western Europe"),
 "Austria":("Europe","Western Europe"),"Sweden":("Europe","Northern Europe"),"Norway":("Europe","Northern Europe"),
 "Denmark":("Europe","Northern Europe"),"Finland":("Europe","Northern Europe"),"Ireland":("Europe","Northern Europe"),
 "Switzerland":("Europe","Western Europe"),"Czechia":("Europe","Eastern Europe"),"Hungary":("Europe","Eastern Europe"),
 "Romania":("Europe","Eastern Europe"),"Bulgaria":("Europe","Eastern Europe"),"Croatia":("Europe","Southern Europe"),
 "Serbia":("Europe","Southern Europe"),"Ukraine":("Europe","Eastern Europe"),"Russia":("Europe","Eastern Europe"),
 "Japan":("Asia","East Asia"),"China":("Asia","East Asia"),"India":("Asia","South Asia"),"Thailand":("Asia","Southeast Asia"),
 "Vietnam":("Asia","Southeast Asia"),"Indonesia":("Asia","Southeast Asia"),"Malaysia":("Asia","Southeast Asia"),
 "Philippines":("Asia","Southeast Asia"),"Pakistan":("Asia","South Asia"),"Bangladesh":("Asia","South Asia"),
 "SouthKorea":("Asia","East Asia"),"Turkey":("Asia","Middle East"),"Iran":("Asia","Middle East"),"Iraq":("Asia","Middle East"),
 "SaudiArabia":("Asia","Middle East"),"Israel":("Asia","Middle East"),"Jordan":("Asia","Middle East"),
 "Lebanon":("Asia","Middle East"),"Kazakhstan":("Asia","Central Asia"),"Mongolia":("Asia","East Asia"),"Nepal":("Asia","South Asia"),
 "Egypt":("Africa","North Africa"),"Nigeria":("Africa","West Africa"),"Kenya":("Africa","East Africa"),
 "Morocco":("Africa","North Africa"),"Ghana":("Africa","West Africa"),"Ethiopia":("Africa","East Africa"),
 "Tanzania":("Africa","East Africa"),"Uganda":("Africa","East Africa"),"Algeria":("Africa","North Africa"),
 "Tunisia":("Africa","North Africa"),"Senegal":("Africa","West Africa"),"Angola":("Africa","Southern Africa"),
 "Canada":("America","North America"),"Mexico":("America","North America"),"Cuba":("America","North America"),
 "Guatemala":("America","North America"),"Panama":("America","North America"),"CostaRica":("America","North America"),
 "Australia":("Oceania","Oceania"),"NewZealand":("Oceania","Oceania")}

# gold p/ medir (amostra confiável)
GOLD={"Brazil":dict(capital="Brasilia",continent="America",language="Portuguese"),
 "Argentina":dict(capital="Buenos",continent="America",language="Spanish"),
 "Peru":dict(capital="Lima",continent="America",language="Spanish"),
 "France":dict(capital="Paris",continent="Europe",language="French"),
 "Germany":dict(capital="Berlin",continent="Europe",language="German"),
 "Japan":dict(capital="Tokyo",continent="Asia",language="Japanese"),
 "Egypt":dict(capital="Cairo",continent="Africa",language="Arabic"),
 "Canada":dict(capital="Ottawa",continent="America"),
 "Portugal":dict(capital="Lisbon",language="Portuguese"),
 "China":dict(capital="Beijing",continent="Asia")}

class IARA:
    def __init__(self):
        g=json.load(open(os.path.join(HERE,"iara_graph_big.json")))
        self.G={tuple(k.split("|")):v for k,v in g["graph"].items() if v}
        self.SRC={tuple(k.split("|")):s for k,s in g.get("src",{}).items() if (tuple(k.split("|")) in self.G)}
        for c,(cont,reg) in GEO.items():                 # backbone geográfico objetivo (fato-terra)
            self.G[(c,"continent")]=cont; self.SRC[(c,"continent")]="geo"
            self.G[(c,"region")]=reg;    self.SRC[(c,"region")]="geo"
        self.INV={}
        for (c,r),v in self.G.items(): self.INV.setdefault((r,norm(v)),[]).append(c)
        self.ENT=sorted({c for (c,r) in self.G})
        self.RELS=["capital","language","currency","continent","region","hemisphere"]
        self.W={}; self.cofire={}; self.born={}
        # germinador byte-MBP
        self.germ=MBP().to(DEV); self.germ.load_state_dict(torch.load(os.path.join(HERE,"iara_byte_germinator.pt"),map_location=DEV)); self.germ.eval()
    @torch.no_grad()
    def _byte_gen(self,prompt,n=40,spec=True):
        """geração byte-nativa. spec=True usa MBP (aceita bytes-à-frente concordantes) = mais rápido."""
        ids=list(prompt.encode()); steps=0
        while len(ids)<len(prompt.encode())+n:
            x=torch.tensor(ids[-CTX:],device=DEV)[None]; outs=self.germ(x); steps+=1
            nb=int(outs[0][0,-1].argmax()); ids.append(nb)
            if nb==10: break
            if spec:                                   # MBP: tenta cravar as próximas cabeças de uma vez
                for k in range(1,K):
                    pb=int(outs[k][0,-1].argmax())
                    # só aceita especulativo se a cabeça 0 do novo contexto confirmar
                    xv=torch.tensor(ids[-CTX:],device=DEV)[None]
                    if int(self.germ(xv)[0][0,-1].argmax())==pb: ids.append(pb)
                    else: break
                    if pb==10: break
                if ids[-1]==10: break
        return bytes(ids).decode("utf-8","ignore"), steps
    def _match(self,q):
        """órgão-byte: acha a entidade do grafo mesmo com TYPO (menor distância de edição)."""
        toks=re.findall(r"[A-Za-z]{3,}",q)
        best=(None,99)
        for c in self.ENT:
            cn=strip_acc(c);
            for t in toks:
                d=edit(t.lower(),cn.lower())
                if d<best[1]: best=(c,d)
            # entidades compostas (South Korea)
            if sub(c,q): best=(c,0)
        c,d=best
        return (c,d) if (c and d<=2) else (None,d)     # tolera até 2 edições (typo)
    def _germinate(self,ent,rel):
        g,_=self._byte_gen(f"Q: {rel} of {ent}? A:",n=16,spec=False)
        w=re.search(r"A:\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ ]{1,20})",g)
        v=(w.group(1).strip() if w else None)
        if v:
            self.G[(ent,rel)]=v; self.SRC[(ent,rel)]="byte-germ"; self.INV.setdefault((rel,norm(v)),[]).append(ent)
            if ent not in self.ENT: self.ENT.append(ent)
        return v
    def _strengthen(self,c,r):
        self.W[(c,r)]=self.W.get((c,r),0)+1
        # nascimento de conceito: capitais de uma mesma REGIÃO co-ativadas
        reg=self.G.get((c,"region")) or self.G.get((c,"continent"))
        if r=="capital" and reg:
            nm=f"capital@{norm(reg)}"; members=[x for x in self.ENT if norm(self.G.get((x,"region"),self.G.get((x,"continent"),"")))==norm(reg)]
            self.cofire[nm]=self.cofire.get(nm,0)+1
            if nm not in self.born and self.cofire[nm]>=6:
                caps={m:self.G[(m,'capital')] for m in members if (m,'capital') in self.G}
                if len(caps)>=2: self.born[nm]=caps
    def answer(self,q):
        t=time.perf_counter(); ql=norm(q); trace=[]
        rel=next((r for r in self.RELS if r in ql),None)
        # 1) multi-hop/cross (mais específico: tem restrição de língua) → interseção região∩língua
        mc=re.search(r"(south america|america|europe|asia|africa).*?(portuguese|spanish|french|german|arabic|japanese|english)",ql)
        if mc and "capital" in ql:
            reg,lang=mc.group(1),mc.group(2)
            cand=[c for c in (self.INV.get(("region",reg)) or self.INV.get(("continent",reg),[])) if sub(self.G.get((c,"language"),""),lang)]
            trace.append(f"região/continente={reg} ∩ língua={lang} → {cand}")
            if cand:
                v=self.G.get((cand[0],"capital")) or self._germinate(cand[0],"capital")
                if v:
                    trace.append(f"{cand[0]} --capital--> {v} [{self.SRC.get((cand[0],'capital'),'?')}]"); self._strengthen(cand[0],"capital")
                    return dict(ans=v,conf="alta",organ="grafo/multi-hop",trace=trace,ms=(time.perf_counter()-t)*1e3)
        # 2) agregado: "capitais/países de <região|continente>"
        m=re.search(r"(capitals?|countries).*?(south america|north america|america|europe|asia|africa|oceania|western europe|east asia)",ql)
        if m:
            reg=m.group(2); members=self.INV.get(("region",reg)) or self.INV.get(("continent",reg),[])
            if members:
                nm=f"capital@{reg}"
                if nm in self.born:
                    trace.append(f"SEMENTE-CONCEITO {nm} (nasceu por uso) → bundle instantâneo")
                    ans=", ".join(self.born[nm].values())
                elif "capital" in ql:
                    ans=", ".join(self.G[(c,'capital')] for c in members if (c,'capital') in self.G)
                    trace.append(f"aresta inversa (região/continente={reg}) → capitais")
                else:
                    ans=", ".join(members); trace.append(f"aresta inversa (região/continente={reg}) → países")
                return dict(ans=ans,conf="alta",organ="grafo/agregado",trace=trace,ms=(time.perf_counter()-t)*1e3)
        # 3) direto: órgão-byte corrige typo → entidade → grafo (ou germina a falta)
        ent,dist=self._match(q)
        if ent and rel:
            if dist>0: trace.append(f"órgão-byte corrigiu typo → '{ent}' (dist {dist})")
            if (ent,rel) in self.G:
                v=self.G[(ent,rel)]; self._strengthen(ent,rel)
                trace.append(f"grafo[{ent}|{rel}] = {v} [{self.SRC.get((ent,rel),'?')}] (força {self.W[(ent,rel)]})")
                return dict(ans=v,conf="alta",organ="grafo/direto",trace=trace,ms=(time.perf_counter()-t)*1e3)
            v=self._germinate(ent,rel); trace.append(f"faltava no grafo → germinou via byte-MBP → {v} (cristalizou)")
            return dict(ans=v or "não sei",conf="média(germinado)",organ="germinador-byte",trace=trace,ms=(time.perf_counter()-t)*1e3)
        # 4) entidade desconhecida em padrão de fato → ABSTÉM
        if ("capital of" in ql or "capital city of" in ql) and ent is None:
            trace.append("entidade fora do grafo (dist>2) → verificador ABSTÉM (não alucina)")
            return dict(ans="não sei (fora do que foi semeado)",conf="abstém",organ="verificador",trace=trace,ms=(time.perf_counter()-t)*1e3)
        # 5) fluência byte
        g,steps=self._byte_gen(f"Q: {q} A:",n=40)
        return dict(ans=g,conf="fluência-byte",organ="germinador-byte",trace=trace,ms=(time.perf_counter()-t)*1e3)


log(f"\n{'='*72}\n# IARA FINAL — leve/inteligente/rápida/byte-nativa — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time(); ia=IARA()
kb=os.path.getsize(os.path.join(HERE,"iara_graph_big.json"))/1024
pt=sum(p.numel() for p in ia.germ.parameters())/1e6
log(f"IARA carregada em {time.time()-t0:.1f}s: memória {len(ia.G)} arestas ({kb:.1f}KB) + germinador byte-MBP {pt:.1f}M params")

# ---------- BATERIA DE CAPACIDADE, INTELIGÊNCIA E ROBUSTEZ ----------
log(f"\n## CAPACIDADE + INTELIGÊNCIA RACIONAL + ROBUSTEZ (a IARA final)")
tests={
 "fato direto":[("What is the capital of Peru?","Lima"),("capital of France?","Paris"),
                ("capital of Japan?","Tokyo"),("continent of Egypt?","Africa")],
 "TYPO (órgão-byte)":[("capital of Brzil?","Bras"),("capital of Frnace?","Paris"),
                      ("capital of Jpan?","Tokyo"),("capital of Prtugal?","Lisbon")],
 "agregado":[("capitals of countries in South America?","Lima"),("countries in Europe?","France")],
 "multi-hop/cross":[("capital of the South America country that speaks Portuguese?","Bras"),
                    ("capital of the Europe country that speaks French?","Paris")],
 "abstenção (falso)":[("What is the capital of Wakanda?","não sei"),("capital of Genovia?","não sei")],
}
res={}
for cat,qs in tests.items():
    ok=0; lat=[]; exemplo=""
    for q,gold in qs:
        r=ia.answer(q); lat.append(r["ms"])
        good = (gold=="não sei" and r["conf"]=="abstém") or (gold!="não sei" and r["ans"] and sub(gold,r["ans"]))
        ok+=bool(good)
        if not exemplo: exemplo=f"'{q[:30]}' → {str(r['ans'])[:24]!r} [{r['organ']}, {r['ms']:.2f}ms]"
    res[cat]=(ok,len(qs),sum(lat)/len(lat))
    log(f"  {cat:<20} {ok}/{len(qs)}  (~{sum(lat)/len(lat):.2f}ms)  {exemplo}")

# ---------- MBP: velocidade byte-à-frente (speculativo) ----------
log(f"\n## MBP — geração byte especulativa (4 bytes/passo)")
txt_s,st_s=ia._byte_gen("Q: capital of Brazil? A:",n=24,spec=True)
txt_n,st_n=ia._byte_gen("Q: capital of Brazil? A:",n=24,spec=False)
log(f"  com MBP-speculativo: {st_s} passos p/ {len(txt_s.encode())-25} bytes · sem: {st_n} passos · aceleração ~{st_n/max(1,st_s):.1f}×")
log(f"  saída: {txt_s[len('Q: capital of Brazil? A:'):][:26]!r}")

# ---------- TRACER de uma jornada ----------
log(f"\n## TRACER — 'capital do país sul-americano que fala português' (multi-hop, acento consertado)")
r=ia.answer("capital of the South America country that speaks Portuguese?")
log(f"  → resposta: {r['ans']}  (confiança {r['conf']}, órgão {r['organ']}, {r['ms']:.2f}ms)")
for st in r["trace"]: log(f"      → {st}")

# ---------- aprende com o uso ----------
log(f"\n## APRENDE COM O USO — regando capitais repetidas (nascimento de conceito)")
for _ in range(8):
    for c in [x for x in ia.ENT if norm(ia.G.get((x,"continent"),""))=="america"][:6]:
        ia.answer(f"capital of {c}?")
log(f"  conceitos nascidos por co-ativação: {list(ia.born)[:3]}{'...' if len(ia.born)>3 else ''}")

# ---------- VALIDAÇÃO EM ESCALA: robustez a typo sobre TODAS as capitais do grafo (N real) ----------
log(f"\n## VALIDAÇÃO EM ESCALA — robustez byte a typo sobre as {sum(1 for (c,r) in ia.G if r=='capital')} capitais do grafo")
import random as _rnd
def _typo(w):
    if len(w)<4: return w
    i=(sum(map(ord,w))%(len(w)-2))+1                    # determinístico (sem Math.random)
    return w[:i]+w[i+1]+w[i]+w[i+2:]                    # swap de 2 letras
caps=[(c,ia.G[(c,'capital')]) for (c,r) in list(ia.G) if r=='capital']
clean_ok=typo_ok=typo_recov=0; lat_clean=[]; lat_typo=[]
for c,cap in caps:
    ent=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",c)           # South Korea
    rc=ia.answer(f"capital of {ent}?"); lat_clean.append(rc["ms"])
    rt=ia.answer(f"capital of {_typo(ent)}?"); lat_typo.append(rt["ms"])
    clean_ok+= bool(rc["ans"] and sub(cap,rc["ans"]))
    good_t = bool(rt["ans"] and sub(cap,rt["ans"])); typo_ok+=good_t
    typo_recov+= (good_t and "órgão-byte" in " ".join(rt["trace"]))
N=len(caps)
log(f"  LIMPO: {clean_ok}/{N} = {clean_ok/N:.0%} certo (~{sum(lat_clean)/N:.2f}ms)")
log(f"  COM TYPO no país: {typo_ok}/{N} = {typo_ok/N:.0%} certo (órgão-byte recuperou {typo_recov}) (~{sum(lat_typo)/N:.2f}ms)")
log(f"    → byte-nativo mantém {typo_ok/N:.0%} sob typo (o token BPE estilhaça a palavra e cai ~45% — medido antes)")

score=sum(ok for ok,n,_ in res.values())/sum(n for _,n,_ in res.values())
log(f"\n## VEREDITO — a IARA final")
log(f"  capacidade global: {score:.0%} · fato ~{res['fato direto'][2]:.2f}ms · TYPO {res['TYPO (órgão-byte)'][0]}/{res['TYPO (órgão-byte)'][1]} · abstém em falso")
log(f"  LEVE: conhecimento {kb:.1f}KB + germinador byte {pt:.1f}M (vs 7B=14GB) · RÁPIDA: grafo instantâneo, byte só na falta")
log(f"  BYTE-NATIVA: órgão-byte corrige typo cobrindo {len(ia.ENT)} entidades; germinador fala byte c/ MBP")
log(f"  INTELIGENTE: navega direto/agregado/multi-hop + aprende conceito; VERIFICA (abstém, não alucina)")
log(f"  ESCALA: robustez a typo {typo_ok}/{N}={typo_ok/N:.0%} (byte) vs ~45% (token BPE) sobre N={N} capitais reais")
json.dump({k:(a,b,round(c,3)) for k,(a,b,c) in res.items()}|{"score":round(score,3),"edges":len(ia.G),"kb":round(kb,1),
    "germ_M":round(pt,1),"born":list(ia.born),"scale_N":N,"clean":f"{clean_ok}/{N}","typo":f"{typo_ok}/{N}"},
    open(os.path.join(HERE,"iara_final.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f}min")
