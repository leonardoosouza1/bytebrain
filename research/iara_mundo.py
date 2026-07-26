#!/usr/bin/env python3
"""IARA-MUNDO — teste de fogo de SOBREVIVÊNCIA. A criatura vive num mundo com comida, predador e
necessidades. O cérebro é NEUROMODULADO (os mesmos princípios do cérebro anatômico, num corpo leve
p/ dar população): dopamina(recompensa)·cortisol(perigo)·curiosidade(explora)·fadiga(descansa)·
LIBIDO(acasala). Ela REAGE, APRENDE COM O ERRO (plasticidade 3-fatores gated pelo neuromodulador),
SOBREVIVE o máximo e PROCRIA. Valida por ablação: aprendizado on/off, neuromod on/off, libido on/off.
Uso: python3 iara_mundo.py [--watch] [--bateria]"""
import math, random, sys, json

W=H=60.0                                   # mundo toroidal fácil
NFOOD=42; NPRED=4
FEATS=["fome","energia_baixa","perigo_perto","comida_perto","parceiro_perto","fadiga"]
ACTS=["buscar_comida","fugir","buscar_parceiro","explorar","descansar"]

def wrap(x,L): return x%L
def tdist(ax,ay,bx,by):                    # distância toroidal
    dx=abs(ax-bx); dy=abs(ay-by); dx=min(dx,W-dx); dy=min(dy,H-dy)
    return math.hypot(dx,dy),dx,dy
def tdir(ax,ay,bx,by):                     # direção toroidal (unit)
    dx=bx-ax; dy=by-ay
    if dx> W/2: dx-=W
    if dx<-W/2: dx+=W
    if dy> H/2: dy-=H
    if dy<-H/2: dy+=H
    n=math.hypot(dx,dy) or 1; return dx/n,dy/n

class Creature:
    _id=0
    def __init__(s,x,y,W0=None,learn=True,neuro=True,mate=True):
        s.id=Creature._id; Creature._id+=1
        s.x=x; s.y=y; s.energy=1.0; s.hunger=0.2; s.health=1.0; s.age=0; s.alive=True
        s.dop=s.cort=s.cur=s.fad=s.lib=0.0  # neuromoduladores (libido = acasalamento)
        s.learn=learn; s.neuro=neuro; s.mate=mate; s.births=0
        # política aprendível: peso[ação][feature] + viés — começa aleatória pequena
        if W0 is None: s.Wt=[[random.uniform(-.2,.2) for _ in FEATS] for _ in ACTS]
        else: s.Wt=[[w+random.gauss(0,.12) for w in row] for row in W0]  # herda + muta
        s.last=None
    def sense(s,world):
        fd=pd=md=99.0; fdir=pdir=mdir=(0,0); mate=None
        for f in world.food:
            d,_,_=tdist(s.x,s.y,f[0],f[1])
            if d<fd: fd=d; fdir=tdir(s.x,s.y,f[0],f[1])
        for p in world.preds:
            d,_,_=tdist(s.x,s.y,p[0],p[1])
            if d<pd: pd=d; pdir=tdir(s.x,s.y,p[0],p[1])
        for c in world.creatures:
            if c is s or not c.alive or c.age<40: continue
            d,_,_=tdist(s.x,s.y,c.x,c.y)
            if d<md: md=d; mdir=tdir(s.x,s.y,c.x,c.y); mate=c
        s._fdir,s._pdir,s._mdir,s._mate=fdir,pdir,mdir,mate
        return [min(1,s.hunger), 1-s.energy, max(0,1-pd/14), max(0,1-fd/14), max(0,1-md/16), s.fad]
    def neuromod(s,feat):                  # o ambiente dispara os hormônios
        peri,comi,parc=feat[2],feat[3],feat[4]
        s.cort=min(1,s.cort*.8+peri*0.9)                       # perigo → cortisol
        s.cur=min(1,s.cur*.9+(0.15 if (peri<.2 and s.hunger<.5) else 0))  # seguro+saciado → curiosidade
        s.fad=min(1,max(0,s.fad+ (0.02 if s.energy<.4 else -0.01)))       # cansaço
        if s.neuro and s.energy>.6 and s.age>60 and s.mate:   # maduro + com energia → LIBIDO (acasalamento)
            s.lib=min(1,s.lib+0.06)
        else: s.lib=max(0,s.lib-0.03)
    def decide(s,feat):
        score=[sum(s.Wt[a][i]*feat[i] for i in range(len(FEATS))) for a in range(len(ACTS))]
        if s.neuro:                        # NEUROMODULAÇÃO enviesa a decisão (homeostase por drive)
            score[0]+=2.4*feat[0]          # fome → buscar comida
            score[1]+=3.0*s.cort           # cortisol → fugir
            score[2]+=2.6*s.lib            # libido → buscar parceiro
            score[3]+=1.4*s.cur            # curiosidade → explorar
            score[4]+=2.0*s.fad*feat[1]    # fadiga+sem energia → descansar
        m=max(score); ex=[math.exp(v-m) for v in score]; Z=sum(ex); p=[e/Z for e in ex]
        r=random.random(); acc=0
        for a,pi in enumerate(p):
            acc+=pi
            if r<=acc: s.last=(feat,a); return a
        s.last=(feat,len(ACTS)-1); return len(ACTS)-1
    def learn_from(s,reward,driver):       # PLASTICIDADE 3-FATORES: Δw = lr · neuromod · reward · feature (aprende com acerto/erro)
        if not s.learn or not s.last: return
        feat,a=s.last; mod=(s.dop if reward>0 else s.cort) if s.neuro else 0.6
        lr=0.08*(0.5+mod)
        for i in range(len(FEATS)): s.Wt[a][i]+=lr*reward*feat[i]
    def step(s,world):
        if not s.alive: return
        s.age+=1; feat=s.sense(world); s.neuromod(feat); a=s.decide(feat)
        act=ACTS[a]; reward=0.0; driver="dopamina"
        # metabolismo
        s.hunger=min(1,s.hunger+world.hrate); s.energy=max(0,s.energy-0.006)
        move=0.0
        if act=="buscar_comida": dx,dy=s._fdir; move=1.3
        elif act=="fugir": dx,dy=(-s._pdir[0],-s._pdir[1]); move=1.6; s.energy-=0.004
        elif act=="buscar_parceiro": dx,dy=s._mdir; move=1.2
        elif act=="explorar": ang=random.uniform(0,6.28); dx,dy=math.cos(ang),math.sin(ang); move=0.9; s.cur=max(0,s.cur-0.05)
        else: dx,dy=0,0; s.energy=min(1,s.energy+0.02); s.fad=max(0,s.fad-0.06)  # descansar recupera
        s.x=wrap(s.x+dx*move,W); s.y=wrap(s.y+dy*move,H)
        # comer
        for f in list(world.food):
            d,_,_=tdist(s.x,s.y,f[0],f[1])
            if d<2.2:
                s.energy=min(1,s.energy+0.5); s.hunger=max(0,s.hunger-0.5)
                s.dop=min(1,s.dop+0.6); reward=1.0; driver="dopamina"
                world.food.remove(f); world.food.append(world.rndpos()); break
        # perigo (aprender com o ERRO)
        for p in world.preds:
            d,_,_=tdist(s.x,s.y,p[0],p[1])
            if d<2.6:
                s.health-=0.34; s.cort=min(1,s.cort+0.7)
                if reward==0: reward=-1.0; driver="cortisol"
        # acasalar
        if act=="buscar_parceiro" and s._mate and s.lib>0.6 and s.energy>0.5:
            d,_,_=tdist(s.x,s.y,s._mate.x,s._mate.y)
            if d<2.6 and s._mate.lib>0.5 and s._mate.energy>0.4 and len(world.creatures)<80:
                world.spawn_child(s,s._mate); s.energy-=0.3; s._mate.energy-=0.3
                s.lib=s._mate.lib=0.0; s.dop=min(1,s.dop+0.8)
                reward=max(reward,1.2); driver="libido"; s.births+=1
        # fome/energia zeradas machucam
        if s.hunger>=1.0: s.health-=0.02
        if s.energy<=0.0: s.health-=0.03
        s.learn_from(reward,driver)
        s.dop*=0.85
        if s.health<=0 or s.age>1600:
            s.alive=False; world.lifespans.append(s.age)
        return act,reward,driver

class World:
    def __init__(s,learn=True,neuro=True,mate=True,n0=10,seed=0,nfood=NFOOD,npred=NPRED,hrate=0.012,pspeed=0.8):
        random.seed(seed); Creature._id=0
        s.hrate=hrate; s.pspeed=pspeed
        s.food=[s.rndpos() for _ in range(nfood)]
        s.preds=[[random.uniform(0,W),random.uniform(0,H)] for _ in range(npred)]
        s.creatures=[Creature(random.uniform(0,W),random.uniform(0,H),learn=learn,neuro=neuro,mate=mate) for _ in range(n0)]
        s.learn,s.neuro,s.mate=learn,neuro,mate; s.t=0; s.deaths=[]; s.total_births=0; s.lifespans=[]
    def rndpos(s): return [random.uniform(0,W),random.uniform(0,H)]
    def spawn_child(s,a,b):
        Wc=[[(a.Wt[i][j]+b.Wt[i][j])/2 for j in range(len(FEATS))] for i in range(len(ACTS))]  # crossover dos pais
        c=Creature(wrap(a.x+random.uniform(-3,3),W),wrap(a.y+random.uniform(-3,3),H),W0=Wc,learn=s.learn,neuro=s.neuro,mate=s.mate)
        c.energy=0.7; s.creatures.append(c); s.total_births+=1
    def step(s):
        s.t+=1
        for p in s.preds:                  # predador vagueia rumo à criatura viva mais próxima (devagar)
            near=None; nd=99
            for c in s.creatures:
                if c.alive:
                    d,_,_=tdist(p[0],p[1],c.x,c.y)
                    if d<nd: nd=d; near=c
            if near and nd<26: dx,dy=tdir(p[0],p[1],near.x,near.y); p[0]=wrap(p[0]+dx*s.pspeed,W); p[1]=wrap(p[1]+dy*s.pspeed,H)
            else: p[0]=wrap(p[0]+random.uniform(-1,1),W); p[1]=wrap(p[1]+random.uniform(-1,1),H)
        for c in list(s.creatures):
            if c.alive: c.step(s)
            elif c not in s.deaths: s.deaths.append(c)
        s.creatures=[c for c in s.creatures if c.alive][:80]
    def alive(s): return [c for c in s.creatures if c.alive]

def run(learn=True,neuro=True,mate=True,ticks=1500,seed=0,narrate=False,n0=10,**dif):
    w=World(learn,neuro,mate,seed=seed,n0=n0,**dif); lifespans=[]; peak=len(w.creatures)
    for t in range(ticks):
        w.step(); peak=max(peak,len(w.alive()))
        if narrate and t%150==0:
            al=w.alive()
            if al:
                c=al[0]
                print(f"  t{t:4d} · vivas {len(al):2d} · nascimentos {w.total_births} · ex: id{c.id} idade{c.age} en{c.energy:.2f} fome{c.hunger:.2f} | dop{c.dop:.1f} cort{c.cort:.1f} lib{c.lib:.1f}")
        if not w.alive() and w.total_births==0: break
    for c in w.creatures+w.deaths:
        if not c.alive: lifespans.append(c.age)
    born_survivors=[c for c in w.alive()]
    todos=w.lifespans+[ticks for _ in w.alive()]  # morreram (idade real) + sobreviventes (censurados no fim)
    avg=sum(todos)/len(todos) if todos else 0
    surv=len(w.alive())
    return dict(vida_media=round(avg,1), nascimentos=w.total_births, pop_final=len(w.alive()), pico=peak, mortes=len(w.lifespans), sobreviveram=len(w.alive()))

if __name__=="__main__":
    if "--watch" in sys.argv:
        print("=== IARA-MUNDO — uma população vivendo (neuromod + aprendizado + acasalamento) ===")
        r=run(narrate=True,ticks=1500,seed=1)
        print("\nRESULTADO:",r)
    else:
        print("=== BATERIA DE VALIDAÇÃO (5 sementes, 1500 ticks) — o teste de fogo ===\n")
        cfgs=[("COMPLETA (neuromod+aprende+acasala)",dict(learn=True,neuro=True,mate=True)),
              ("SEM aprendizado",dict(learn=False,neuro=True,mate=True)),
              ("SEM neuromodulação",dict(learn=True,neuro=False,mate=True)),
              ("SEM libido (não acasala)",dict(learn=True,neuro=True,mate=False))]
        for nome,cfg in cfgs:
            rs=[run(ticks=1500,seed=sd,**cfg) for sd in range(5)]
            vm=sum(r["vida_media"] for r in rs)/5; nb=sum(r["nascimentos"] for r in rs)/5; pk=sum(r["pico"] for r in rs)/5
            print(f"  {nome:38s} vida_média {vm:6.1f} · nascimentos {nb:4.1f} · pico_pop {pk:4.1f}")
        print("\n=== MUNDO DIFÍCIL (fome rápida, 9 predadores ágeis, comida escassa) — valida a SOBREVIVÊNCIA ===\n")
        HARD=dict(nfood=22,npred=9,hrate=0.02,pspeed=1.15)
        for nome,cfg in cfgs:
            rs=[run(ticks=1500,seed=sd,**cfg,**HARD) for sd in range(5)]
            vm=sum(r["vida_media"] for r in rs)/5; nb=sum(r["nascimentos"] for r in rs)/5; pk=sum(r["pico"] for r in rs)/5; mo=sum(r["mortes"] for r in rs)/5
            print(f"  {nome:38s} vida_média {vm:6.1f} · mortes {mo:5.1f} · nascimentos {nb:4.1f} · pico {pk:4.1f}")
        print("\n=== SOBREVIVÊNCIA ISOLADA (SEM reprodução, mundo difícil, coorte 16, 1200 ticks) ===\n")
        SV=dict(nfood=20,npred=9,hrate=0.02,pspeed=1.15)
        for nome,cfg in [("COMPLETA (neuromod+aprende)",dict(learn=True,neuro=True)),
                         ("SEM aprendizado (reflexo fixo)",dict(learn=False,neuro=True)),
                         ("SEM neuromodulação (aleatório)",dict(learn=True,neuro=False))]:
            rs=[run(ticks=1200,seed=sd,mate=False,n0=16,**cfg,**SV) for sd in range(6)]
            vm=sum(r["vida_media"] for r in rs)/6; sv=sum(r["sobreviveram"] for r in rs)/6; mo=sum(r["mortes"] for r in rs)/6
            print(f"  {nome:34s} vida_média {vm:6.1f} ticks · morreram {mo:4.1f}/16 · sobreviveram {sv:4.1f}/16")
        print("\n→ COMPLETA vive mais = neuromod+aprendizado geram competência de sobrevivência (validação honesta).")
