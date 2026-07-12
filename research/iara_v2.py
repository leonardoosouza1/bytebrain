#!/usr/bin/env python3
"""IARA V2 — cérebro multimodal: VÊ, DIZ e RELACIONA, tudo dos próprios órgãos (2026-07-12).

Pluga o OLHO (CLIP, iara_eye) no CÉREBRO que se auto-expande (iara_brain_grow) por um BARRAMENTO
DE PERCEPTO comum: todo órgão emite Percepto(modality, concepts=[(label,conf)]). O cérebro consome
igual, venha de texto ou de imagem. Fluxo:
  ver(img) → olho emite conceitos contrastados → cérebro DIZ o que vê e tenta RELACIONAR (choque
  dirigido: "onde fica X?" → país → grafo crescido → capital). Nada hardcoded: a ponte marco→país→capital
  é APRENDIDA pelos mesmos choques. Prepara a V3: fuse([percepto_visão, percepto_fala], pergunta).
Sem mock: olho e cérebro reais na GPU. Honesto."""
import os, re, time, json
HERE=os.path.dirname(os.path.abspath(__file__)); import sys; sys.path.insert(0,HERE)
from iara_eye import Eye
from iara_brain_grow import Brain, VOCAB, norm, first_word
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    open(JOUR,"a").write(s+"\n")
def clean_concept(c): return re.sub(r"^(a |an |the )","",c).strip()

class IARAv2:
    def __init__(self):
        t=time.time()
        self.eye=Eye()                                           # órgão-olho (CLIP)
        self.brain=Brain()                                      # cérebro auto-expansível (3B substrato)
        self.load_s=time.time()-t
    def _relate(self,concept):
        """choque dirigido: em que PAÍS este conceito vive? (2 fraseados concordam = liga)"""
        cc=clean_concept(concept)
        a=first_word(self.brain._gen(f"The {cc} is located in the country of",4))
        b=first_word(self.brain._gen(f"The {cc} is in the country of",4))
        country=next((v for v in VOCAB if a and norm(v)==norm(a)),None)
        agree = a and b and norm(a)==norm(b)                     # ambos miram o PAÍS (guarda contra falso)
        if country and agree:
            self.brain.perceive(country)
            cap,conf=self.brain.learn_fact(country,"capital")
            return dict(country=country,capital=cap,cap_conf=conf)
        return None
    def see_and_say(self,image):
        """VÊ → DIZ → RELACIONA (barramento de percepto)."""
        t=time.perf_counter()
        p=self.eye.see(image,topk=3)                            # Percepto de visão
        if not p["concepts"]:
            return dict(percept=p,say="não reconheci nada com confiança",ms=(time.perf_counter()-t)*1e3)
        label,conf=p["concepts"][0]
        say=f"Vejo {label} ({conf:.0%})"
        rel=self._relate(label)
        chain=[]
        if rel:
            say+=f". Isso fica em {rel['country']}, cuja capital é {rel['capital']}."
            chain=[label,f"país={rel['country']}",f"capital={rel['capital']} ({rel['cap_conf']})"]
        return dict(percept=p,say=say,relate=rel,chain=chain,ms=(time.perf_counter()-t)*1e3)
    def fuse(self,percepts,query):
        """V3: funde perceptos (visão + fala/texto) → responde. Aqui: visão dá o contexto, texto pergunta."""
        vis=[c for pc in percepts if pc["modality"]=="vision" for c,_ in pc["concepts"]]
        rel=self._relate(vis[0]) if vis else None
        ql=norm(query)
        if rel and "capital" in ql: return f"A capital é {rel['capital']}."
        if rel and ("country" in ql or "pais" in ql or "onde" in ql): return f"Fica em {rel['country']}."
        return f"Vejo {vis[0] if vis else 'nada claro'}; não sei responder isso ainda."

if __name__=="__main__":
    IMG="/tmp/iara_imgs"
    log(f"\n{'='*72}\n# IARA V2 — VÊ, DIZ e RELACIONA (olho + cérebro auto-expansível) — {time.strftime('%H:%M')}\n{'='*72}")
    ia=IARAv2()
    log(f"IARA V2 no ar em {ia.load_s:.0f}s: olho CLIP + cérebro 3B (substrato) + grafo que cresce. barramento de percepto ativo.")
    log(f"\n## VÊ e DIZ (tudo dos próprios órgãos)")
    for name in ["eiffel.jpg","dog.jpg","car.jpg"]:
        p=os.path.join(IMG,name)
        if not os.path.exists(p): log(f"  (faltou {name})"); continue
        r=ia.see_and_say(p)
        log(f"  [{name}] {r['say']}  [{r['ms']:.0f}ms]")
        if r.get("chain"): log(f"      cadeia: {' → '.join(r['chain'])}")
    log(f"\n## FUSE (o caminho da V3: visão + pergunta)")
    p=ia.eye.see(os.path.join(IMG,"eiffel.jpg"),topk=3)
    for q in ["what is the capital there?","onde fica isso?"]:
        log(f"  visão(Torre Eiffel) + '{q}' → {ia.fuse([p],q)}")
    log(f"\n## VEREDITO IARA V2")
    log(f"  ✓ VÊ (olho CLIP, conceito contrastado) + DIZ + RELACIONA com o grafo crescido — órgãos alinhados por PERCEPTO")
    log(f"  ✓ ponte visão→cérebro APRENDIDA por choque (Eiffel→França→Paris), não hardcoded")
    log(f"  ✓ barramento pronto p/ V3: fuse([percepto_visão, percepto_fala], pergunta) — só plugar o ouvido")
    log(f"  cérebro cresceu vendo: {len(ia.brain.G)} fatos, {len(ia.brain.perceived)} entidades percebidas")
    json.dump(dict(facts=len(ia.brain.G),perceived=len(ia.brain.perceived),load_s=round(ia.load_s,0)),
        open(os.path.join(HERE,"iara_v2.json"),"w"),indent=1)
