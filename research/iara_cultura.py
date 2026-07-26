#!/usr/bin/env python3
"""IARA-CULTURA — tool-use EMERGENTE + aprendizado social. Nada de hardcode da regra.
Mundo: comida ESCASSA e disputada; quando 2 seres chegam na mesma comida, LUTAM — vence o de maior
poder = base + bônus se estiver segurando um GALHO. O ser NÃO sabe que galho ajuda; descobre pela
recompensa (venceu/perdeu) com CRÉDITO TEMPORAL (traço de elegibilidade → credita 'pegar galho' feito
ANTES da luta). Skill emergente esperada: rival perto + sem galho → BUSCAR GALHO → lutar (plano meio-fim).
E APRENDIZADO SOCIAL: quem VÊ outro vencer segurando galho aumenta a própria tendência de buscar galho
(cultura). Valida: a preparação (lutar COM galho) emerge? o aprendizado social espalha mais rápido?
Uso: python3 iara_cultura.py            (bateria: individual vs +social vs sem-nada)
     python3 iara_cultura.py --curva     (curva: % de lutas com galho ao longo do tempo)
     python3 iara_cultura.py --semente    (1 'sabedor' numa pop ingênua — a cultura se espalha?)"""
import math, random, sys
W=H=44.0
FEATS=["fome","energia_baixa","rival_perto","comida_perto","tem_galho","galho_perto"]
ACTS=["buscar_comida","buscar_galho","lutar","explorar"]
BRANCH_BONUS=0.9                     # o galho dá vantagem REAL na luta (o ser tem que DESCOBRIR isso)

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

class Being:
    _id=0
    def __init__(s,learn=True,social=True,W0=None):
        s.id=Being._id; Being._id+=1
        s.learn=learn; s.social=social
        s.Wt=[[random.uniform(-.2,.2) for _ in FEATS] for _ in ACTS] if W0 is None else [r[:] for r in W0]
        s.x=random.uniform(0,W); s.y=random.uniform(0,H)
        s.energy=1.0; s.hunger=0.2; s.health=1.0; s.age=0; s.alive=True
        s.has_branch=0; s.elig=[[0.0]*len(FEATS) for _ in ACTS]  # traço de elegibilidade (crédito temporal)
        s.wins=0; s.fights=0; s.fights_armed=0; s.last=None
    def sense(s,world):
        gd=cd=rd=bd=99; gdir=cdir=rdir=bdir=(0,0); rival=None
        for f in world.food:
            d=tdist(s.x,s.y,f[0],f[1])
            if d<cd: cd=d; cdir=tdir(s.x,s.y,f[0],f[1])
        for b in world.branches:
            d=tdist(s.x,s.y,b[0],b[1])
            if d<bd: bd=d; bdir=tdir(s.x,s.y,b[0],b[1])
        for o in world.beings:
            if o is s or not o.alive: continue
            d=tdist(s.x,s.y,o.x,o.y)
            if d<rd: rd=d; rdir=tdir(s.x,s.y,o.x,o.y); rival=o
        s._cdir,s._bdir,s._rdir,s._rival=cdir,bdir,rdir,rival
        return [min(1,s.hunger),1-s.energy,max(0,1-rd/12),max(0,1-cd/12),float(s.has_branch),max(0,1-bd/12)]
    def decide(s,f):
        sc=[sum(s.Wt[a][i]*f[i] for i in range(len(FEATS))) for a in range(len(ACTS))]
        sc[0]+=1.8*f[0]                                    # fome → buscar comida (drive básico)
        m=max(sc); ex=[math.exp(v-m) for v in sc]; Z=sum(ex); p=[e/Z for e in ex]
        r=random.random(); acc=0
        for a,pi in enumerate(p):
            acc+=pi
            if r<=acc: a_=a; break
        else: a_=0
        s.last=(f,a_)
        for i in range(len(FEATS)):                        # decai o traço e marca a ação de agora
            for aa in range(len(ACTS)): s.elig[aa][i]*=0.9
            s.elig[a_][i]=f[i]
        return a_
    def reward(s,r):                                       # recompensa credita o que fez recentemente (traço) → conecta 'pegar galho' à vitória
        if not s.learn: return
        lr=0.09
        for a in range(len(ACTS)):
            for i in range(len(FEATS)): s.Wt[a][i]+=lr*r*s.elig[a][i]
    def observe(s,winner):                                # APRENDIZADO SOCIAL: viu outro vencer com galho → imita a preparação
        if not s.social or not winner.alive: return
        if winner.has_branch:                             # o vencedor estava armado → aumenta minha tendência de buscar galho quando há rival
            s.Wt[1][2]+=0.05                               # ação 'buscar_galho' × feature 'rival_perto'
            s.Wt[1][5]+=0.03                               # × 'galho_perto'
    def step(s,world):
        if not s.alive: return
        s.age+=1; f=s.sense(world); a=s.decide(f)
        s.hunger=min(1,s.hunger+0.014); s.energy=max(0,s.energy-0.006)
        if a==0: dx,dy=s._cdir; mv=1.3
        elif a==1: dx,dy=s._bdir; mv=1.2
        elif a==2: dx,dy=s._rdir; mv=1.3                  # vai pro rival lutar
        else: ang=random.uniform(0,6.28); dx,dy=math.cos(ang),math.sin(ang); mv=1.0
        s.x=wrap(s.x+dx*mv,W); s.y=wrap(s.y+dy*mv,H)
        # pegar galho
        for b in list(world.branches):
            if tdist(s.x,s.y,b[0],b[1])<2.0 and not s.has_branch:
                s.has_branch=1; world.branches.remove(b); break
        # comer
        for fo in list(world.food):
            if tdist(s.x,s.y,fo[0],fo[1])<2.0:
                s.energy=min(1,s.energy+.5); s.hunger=max(0,s.hunger-.5); s.reward(0.5)
                world.food.remove(fo); world.food.append(world.rnd()); break
        # LUTA: se ação=lutar e rival adjacente → resolve pelo poder (base + galho)
        if a==2 and s._rival and tdist(s.x,s.y,s._rival.x,s._rival.y)<2.6:
            o=s._rival; s.fights+=1; o.fights+=1
            if s.has_branch: s.fights_armed+=1
            if o.has_branch: o.fights_armed+=1
            ps=random.uniform(0,.4)+BRANCH_BONUS*s.has_branch
            po=random.uniform(0,.4)+BRANCH_BONUS*o.has_branch
            win,lose=(s,o) if ps>=po else (o,s)
            win.wins+=1; win.energy=min(1,win.energy+.35); win.hunger=max(0,win.hunger-.35); win.reward(1.0)
            lose.health-=.35; lose.reward(-1.0); lose.has_branch=0
            for w2 in world.beings:                        # quem está PERTO observa o resultado (aprendizado social)
                if w2 is not win and w2.alive and tdist(w2.x,w2.y,win.x,win.y)<10: w2.observe(win)
        if s.hunger>=1: s.health-=.02
        if s.energy<=0: s.health-=.03
        if s.health<=0 or s.age>1400: s.alive=False; world.dead.append(s)

class World:
    def __init__(s,seed,learn=True,social=True,n0=18,nfood=10,nbranch=6,seed_knower=False):
        random.seed(seed); Being._id=0
        s.food=[s.rnd() for _ in range(nfood)]; s.branches=[s.rnd() for _ in range(nbranch)]
        s.beings=[Being(learn,social) for _ in range(n0)]; s.dead=[]
        if seed_knower:                                    # 1 'sabedor': já sabe buscar galho quando há rival (semente da cultura)
            k=s.beings[0]; k.Wt[1][2]=3.0; k.Wt[1][5]=2.0
    def rnd(s): return [random.uniform(0,W),random.uniform(0,H)]
    def run(s,ticks):
        armed_hist=[]
        for t in range(ticks):
            for b in s.beings:
                if b.alive: b.step(s)
            if t%60==0 and len(s.branches)<6: s.branches.append(s.rnd())
            if t%100==0:
                al=[b for b in s.beings if b.alive]
                tf=sum(b.fights for b in s.beings); ta=sum(b.fights_armed for b in s.beings)
                armed_hist.append(round(ta/max(1,tf),2))
        return armed_hist
    def stats(s):
        tf=sum(b.fights for b in s.beings); ta=sum(b.fights_armed for b in s.beings)
        return dict(taxa_luta_armada=round(ta/max(1,tf),2), lutas=tf, vivos=len([b for b in s.beings if b.alive]))

def run(learn=True,social=True,ticks=1400,seed=0,seed_knower=False):
    w=World(seed,learn,social,seed_knower=seed_knower); hist=w.run(ticks); return w.stats(),hist

if __name__=="__main__":
    if "--curva" in sys.argv:
        print("=== CURVA — % de lutas COM galho ao longo do tempo (a preparação emergindo) ===\n")
        for nome,soc in [("com aprendizado social",True),("só individual",False)]:
            _,h=run(social=soc,seed=1,ticks=1600)
            print(f"  {nome:24s}: "+" ".join(f"{x:.0%}" for x in h))
    elif "--semente" in sys.argv:
        print("=== 1 SABEDOR numa população ingênua — a cultura (buscar galho) se espalha por observação? ===\n")
        for nome,soc in [("COM aprendizado social",True),("SEM (só individual)",False)]:
            rs=[run(social=soc,seed=sd,seed_knower=True,ticks=1400)[0] for sd in range(5)]
            ar=sum(r["taxa_luta_armada"] for r in rs)/5
            print(f"  {nome:24s}: taxa_luta_armada final {ar:.0%}")
        print("\n→ se COM social espalha mais a luta-armada que SEM, a cultura transmitiu (honesto).")
    else:
        print("=== BATERIA: tool-use emergente + cultura (5 sementes, 1400 ticks) ===")
        print("galho dá +dano na luta; o ser tem que DESCOBRIR e se preparar\n")
        for nome,cfg in [("COMPLETO (aprende + social)",dict(learn=True,social=True)),
                         ("SÓ individual (sem social)",dict(learn=True,social=False)),
                         ("SEM aprendizado (baseline)",dict(learn=False,social=False))]:
            rs=[run(seed=sd,ticks=1400,**cfg)[0] for sd in range(5)]
            ar=sum(r["taxa_luta_armada"] for r in rs)/5; lu=sum(r["lutas"] for r in rs)/5
            print(f"  {nome:30s} lutas_com_galho {ar:.0%} · lutas/run {lu:.0f}")
        print("\n→ se COMPLETO luta mais COM galho que o baseline, a skill de tool-use EMERGIU (honesto).")
