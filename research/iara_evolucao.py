#!/usr/bin/env python3
"""IARA-EVOLUÇÃO — neuroevolução geracional que FORÇA uma habilidade a emergir.
Mundo com DISCRIMINAÇÃO: metade da comida é VENENO; um SINAL (feature) a distingue. Reflexo que come
tudo morre; evoluir/aprender a LER o sinal e evitar veneno = skill que dá vantagem. Seleciona os
melhores por fitness → crossover+mutação → próxima geração. Mede a skill EMERGIR e valida por ablação.
Uso: python3 iara_evolucao.py            (bateria: com/sem evolução, aprendizado, neuromod)
     python3 iara_evolucao.py --curva    (curva de fitness/veneno por geração da config completa)"""
import math, random, sys
W=H=50.0
FEATS=["fome","energia_baixa","perigo","comida_boa_perto","veneno_perto","fadiga"]  # sensores SEPARADOS: bom vs veneno
ACTS=["buscar_comida","evitar_comida","fugir","explorar","descansar"]  # evitar_comida = recusar (skill de discriminar veneno)

def wrap(x,L): return x%L
def tdist(ax,ay,bx,by):
    dx=abs(ax-bx); dy=abs(ay-by); dx=min(dx,W-dx); dy=min(dy,H-dy); return math.hypot(dx,dy)
def tdir(ax,ay,bx,by):
    dx=bx-ax; dy=by-ay
    if dx>W/2:dx-=W
    if dx<-W/2:dx+=W
    if dy>H/2:dy-=H
    if dy<-H/2:dy+=H
    n=math.hypot(dx,dy) or 1; return dx/n,dy/n

class Bug:
    def __init__(s,genome,learn,neuro):
        s.Wt=[row[:] for row in genome]; s.learn=learn; s.neuro=neuro
        s.x=random.uniform(0,W); s.y=random.uniform(0,H)
        s.energy=1.0; s.hunger=0.2; s.health=1.0; s.age=0; s.alive=True
        s.dop=s.cort=s.cur=s.fad=0.0; s.bom=0; s.veneno=0; s.last=None
    def sense(s,world):
        gd=bd=pd=99; gdir=bdir=pdir=(0,0)
        for f in world.food:
            d=tdist(s.x,s.y,f[0],f[1])
            if f[2] and d<bd: bd=d; bdir=tdir(s.x,s.y,f[0],f[1])             # veneno mais perto
            if (not f[2]) and d<gd: gd=d; gdir=tdir(s.x,s.y,f[0],f[1])       # comida BOA mais perto
        for p in world.preds:
            d=tdist(s.x,s.y,p[0],p[1])
            if d<pd: pd=d; pdir=tdir(s.x,s.y,p[0],p[1])
        s._gdir,s._bdir,s._pdir=gdir,bdir,pdir
        return [min(1,s.hunger),1-s.energy,max(0,1-pd/12),max(0,1-gd/12),max(0,1-bd/10),s.fad]
    def neuromod(s,f):
        if s.neuro:
            s.cort=min(1,s.cort*.8+f[2]*0.9); s.fad=min(1,max(0,s.fad+(0.02 if s.energy<.4 else -0.01)))
    def decide(s,f):
        sc=[sum(s.Wt[a][i]*f[i] for i in range(len(FEATS))) for a in range(len(ACTS))]
        if s.neuro:
            sc[0]+=2.2*f[0]; sc[2]+=2.8*s.cort; sc[3]+=1.2*s.cur; sc[4]+=1.6*s.fad*f[1]  # fome→comer · cortisol→fugir · curiosidade→explora · fadiga→descansa
        m=max(sc); ex=[math.exp(v-m) for v in sc]; Z=sum(ex); p=[e/Z for e in ex]
        r=random.random(); acc=0
        for a,pi in enumerate(p):
            acc+=pi
            if r<=acc: s.last=(f,a); return a
        s.last=(f,0); return 0
    def learn_from(s,rew):
        if not s.learn or not s.last: return
        f,a=s.last; mod=(s.dop if rew>0 else s.cort) if s.neuro else 0.6
        lr=0.10*(0.5+mod)
        for i in range(len(FEATS)): s.Wt[a][i]+=lr*rew*f[i]
    def step(s,world):
        if not s.alive: return
        s.age+=1; f=s.sense(world); s.neuromod(f); a=s.decide(f); rew=0.0
        s.hunger=min(1,s.hunger+world.hrate); s.energy=max(0,s.energy-0.006)
        if a==0: dx,dy=s._gdir; mv=1.3                                      # buscar comida BOA (usa o sinal)
        elif a==1: dx,dy=(-s._bdir[0],-s._bdir[1]); mv=1.1                  # EVITAR o veneno (afasta do sinal ruim)
        elif a==2: dx,dy=(-s._pdir[0],-s._pdir[1]); mv=1.6                  # fugir do predador
        elif a==3: ang=random.uniform(0,6.28); dx,dy=math.cos(ang),math.sin(ang); mv=1.0; s.cur=max(0,s.cur-.05)
        else: dx,dy=0,0; mv=0; s.energy=min(1,s.energy+.02); s.fad=max(0,s.fad-.06)
        s.x=wrap(s.x+dx*mv,W); s.y=wrap(s.y+dy*mv,H)
        for fo in list(world.food):
            if tdist(s.x,s.y,fo[0],fo[1])<2.2:
                if fo[2]:                                    # VENENO
                    s.health-=0.5; s.cort=min(1,s.cort+.7); rew=-1.2; s.veneno+=1
                else:                                        # comida boa
                    s.energy=min(1,s.energy+.5); s.hunger=max(0,s.hunger-.5); s.dop=min(1,s.dop+.6); rew=1.0; s.bom+=1
                world.food.remove(fo); world.food.append(world.spawn_food()); break
        for p in world.preds:
            if tdist(s.x,s.y,p[0],p[1])<2.5:
                s.health-=0.3; s.cort=min(1,s.cort+.6)
                if rew==0: rew=-1.0
        if s.hunger>=1: s.health-=.02
        s.learn_from(rew); s.dop*=.85
        if s.health<=0 or s.energy<=0 or s.age>900: s.alive=False; world.deaths.append(s.age)

class World:
    def __init__(s,seed,nfood=26,npred=4,hrate=0.016,pfrac=0.5):
        random.seed(seed); s.hrate=hrate; s.pfrac=pfrac
        s.food=[s.spawn_food() for _ in range(nfood)]
        s.preds=[[random.uniform(0,W),random.uniform(0,H)] for _ in range(npred)]
        s.deaths=[]
    def spawn_food(s): return [random.uniform(0,W),random.uniform(0,H),1 if random.random()<s.pfrac else 0]  # 1=veneno
    def run_cohort(s,bugs,ticks):
        for t in range(ticks):
            for p in s.preds:
                near=min((b for b in bugs if b.alive),key=lambda b:tdist(p[0],p[1],b.x,b.y),default=None)
                if near and tdist(p[0],p[1],near.x,near.y)<20: dx,dy=tdir(p[0],p[1],near.x,near.y); p[0]=wrap(p[0]+dx*.9,W); p[1]=wrap(p[1]+dy*.9,H)
            for b in bugs:
                if b.alive: b.step(s)
            if not any(b.alive for b in bugs): break

def rand_genome(): return [[random.uniform(-.2,.2) for _ in FEATS] for _ in ACTS]
def crossover(a,b):
    return [[(a[i][j] if random.random()<.5 else b[i][j])+random.gauss(0,.07) for j in range(len(FEATS))] for i in range(len(ACTS))]
def fitness(b): return b.bom*3 - b.veneno*4 + b.age*0.02        # come bom, evita veneno, vive

def evolve(gens=25,pop=24,ticks=700,learn=True,neuro=True,mutate=True,seed=0,log=False):
    random.seed(seed)
    genomes=[rand_genome() for _ in range(pop)]
    hist=[]
    for g in range(gens):
        bugs=[Bug(genomes[i],learn,neuro) for i in range(pop)]
        w=World(seed*1000+g); w.run_cohort(bugs,ticks)
        scored=sorted(bugs,key=fitness,reverse=True)
        fits=[fitness(b) for b in bugs]; avgf=sum(fits)/pop
        vr=sum(b.veneno for b in bugs)/max(1,sum(b.veneno+b.bom for b in bugs))   # taxa de veneno comido
        hist.append((round(avgf,1),round(vr,2)))
        if log: print(f"  gen {g:2d} · fitness_médio {avgf:6.1f} · taxa_veneno {vr:.0%} · top_fit {round(fitness(scored[0]),1)}")
        if not mutate:                                          # SEM evolução: reembaralha genomas aleatórios (sem herdar os bons)
            genomes=[rand_genome() for _ in range(pop)]; continue
        elite=[b.Wt for b in scored[:max(2,pop//4)]]           # seleção: top 25% procriam
        genomes=elite[:2]+[crossover(random.choice(elite),random.choice(elite)) for _ in range(pop-2)]
    return hist

if __name__=="__main__":
    if "--curva" in sys.argv:
        print("=== CURVA DE EVOLUÇÃO (config completa) — a skill de discriminar veneno emergindo ===\n")
        evolve(gens=25,pop=24,ticks=700,seed=1,log=True)
    else:
        print("=== BATERIA DE EVOLUÇÃO (3 sementes, 25 gerações) — forçando a skill ===")
        print("mundo: metade da comida é VENENO; sobreviver exige LER o sinal e evitar\n")
        cfgs=[("COMPLETA (evolui+aprende+neuromod)",dict(learn=True,neuro=True,mutate=True)),
              ("SEM evolução (só aprende na vida)",dict(learn=True,neuro=True,mutate=False)),
              ("SEM aprendizado (só evolui)",dict(learn=False,neuro=True,mutate=True)),
              ("SEM neuromodulação",dict(learn=True,neuro=False,mutate=True))]
        import statistics as st
        for nome,cfg in cfgs:
            hs=[evolve(gens=30,pop=28,ticks=700,seed=sd,**cfg) for sd in range(4)]
            ini=lambda h:st.mean(x[0] for x in h[:5]); fim=lambda h:st.mean(x[0] for x in h[-5:])   # média 5 primeiras vs 5 últimas gerações (menos ruído)
            vini=lambda h:st.mean(x[1] for x in h[:5]); vfim=lambda h:st.mean(x[1] for x in h[-5:])
            f0=st.mean(ini(h) for h in hs); fN=st.mean(fim(h) for h in hs)
            v0=st.mean(vini(h) for h in hs); vN=st.mean(vfim(h) for h in hs)
            print(f"  {nome:36s} fitness {f0:5.1f}→{fN:5.1f} · veneno_comido {v0:.0%}→{vN:.0%}")
        print("\n→ se a COMPLETA sobe fitness e DERRUBA a taxa de veneno ao longo das gerações, a skill EVOLUIU (honesto).")
