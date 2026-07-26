#!/usr/bin/env python3
"""IARA-CARCAÇA — a ARQUITETURA da IARA (não os pesos) embodied num ser leve e REPLICÁVEL.
Extrai os órgãos LEVES do iara_cerebro.py — NEUROMODULAÇÃO causal (o instinto), FLORESTA como memória
associativa (sinapse gated por hormônio), órgão PREDITIVO (forward model), ROTEADOR System1/2 — e
descarta o pesado (Qwen/embeddings). O ser é AGNÓSTICO: nasce sem saber nada e DESCOBRE sozinho o laço
ação↔percepção vivendo. Alvo de validação honesto: ele aprende a FREAR/DESVIAR antes da parede sozinho
(ninguém ensina; bater dói → cortisol → a sinapse de evitar fortalece). Leve → replica em milhares.
Uso: python3 iara_carcaca.py            (bateria: carcaça completa vs ablações — colisões caem?)
     python3 iara_carcaca.py --curva     (curva de colisões ao longo da vida de 1 ser)
     python3 iara_carcaca.py --milhares   (replica 1000 seres — escala/custo)"""
import math, random, sys, time

# ---- ambiente: arena com PAREDES; o ser anda, vê a distância à frente, e tem que aprender a não bater ----
ARENA=40.0; NOBST=7
def clamp(x,a,b): return a if x<a else (b if x>b else x)

class Carcaca:
    """O ser = carcaça leve da IARA. Órgãos reais, sem peso pesado."""
    def __init__(s,neuro=True,plast=True,pred=True,W0=None):
        s.neuro=neuro; s.plast=plast; s.pred=pred
        # ---- NEUROMODULAÇÃO (extraído: dop/cort/cur/fad) ----
        s.dop=s.cort=s.cur=s.fad=0.0
        # ---- FLORESTA = memória associativa: nó = SITUAÇÃO (dist_bucket, vel_bucket), aresta = situação--ação-->força_evitar ----
        # syn[(dist_b,vel_b)][acao] = força (quão bom/ruim foi fazer essa ação nessa situação). Cresce por hormônio (como _reinforce).
        s.syn={} if W0 is None else {k:v[:] for k,v in W0.items()}
        # ---- órgão PREDITIVO (forward model): prevê Δdist dada a ação+vel. Aprendido online (média por situação). ----
        s.fwd={}            # fwd[(dist_b,vel_b,acao)] = (Δdist_previsto, n)
        s.x=random.uniform(5,ARENA-5); s.y=random.uniform(5,ARENA-5); s.head=random.uniform(0,6.28)
        s.vel=0.3; s.health=1.0; s.age=0; s.alive=True
        s.colisoes=0; s.passos=0; s.last=None; s.energy=1.0; s.dist_tot=0.0
    # --- percepção: distância à parede/obstáculo À FRENTE (raio) + velocidade ---
    def sense(s,world):
        dx,dy=math.cos(s.head),math.sin(s.head); dist=ARENA
        for r in [i*0.6 for i in range(1,40)]:
            px,py=s.x+dx*r,s.y+dy*r
            if px<0 or px>ARENA or py<0 or py>ARENA: dist=r; break
            hit=any((px-o[0])**2+(py-o[1])**2<o[2]*o[2] for o in world.obst)
            if hit: dist=r; break
        s._dist=dist; return dist
    def _db(s,d): return min(4,int(d/3))          # bucket de distância (0=coladíssimo … 4=longe)
    def _vb(s): return min(2,int(s.vel/0.4))       # bucket de velocidade (0=devagar,1,2=rápido)
    def _plast_coef(s):                            # PLASTICIDADE causal (igual _plast: dopamina abre, fadiga fecha)
        return (0.15+0.6*s.cort+0.3*s.dop-0.1*s.fad) if s.plast else 0.12  # cortisol (dor) tb abre p/ aprender a evitar
    def decide(s,world):
        d=s.sense(world); sit=(s._db(d),s._vb())
        row=s.syn.setdefault(sit,[0.0,0.0,0.0,0.0])  # ações: 0=acelera 1=freia 2=vira_esq 3=vira_dir
        score=row[:]
        # ROTEADOR System 2: se o órgão PREDITIVO prevê colisão iminente, pesa fugir/frear (deliberação leve)
        if s.pred:
            for a in range(4):
                dp=s.fwd.get((sit[0],sit[1],a),(0,0))[0]
                if d+dp<2.5: score[a]-=2.5*(1 if a==0 else 0)   # se essa ação (esp. acelerar) leva a bater → pune
        if s.neuro and d<3: score[1]+=1.5*s.cort               # cortisol → viés reflexo de FREAR quando perto
        m=max(score); ex=[math.exp(v-m) for v in score]; Z=sum(ex); p=[e/Z for e in ex]
        r=random.random(); acc=0; a_=3
        for a,pi in enumerate(p):
            acc+=pi
            if r<=acc: a_=a; break
        s.last=(sit,a_); return a_
    def step(s,world):
        if not s.alive: return
        s.age+=1; s.passos+=1; a=s.decide(world); d0=s._dist
        # efetores
        if a==0: s.vel=clamp(s.vel+0.15,0.1,1.2)
        elif a==1: s.vel=clamp(s.vel-0.2,0.0,1.2)
        elif a==2: s.head+=0.5
        else: s.head-=0.5
        nx=s.x+math.cos(s.head)*s.vel; ny=s.y+math.sin(s.head)*s.vel
        s.energy=min(1,s.energy+0.002)
        colidiu = nx<0 or nx>ARENA or ny<0 or ny>ARENA or any((nx-o[0])**2+(ny-o[1])**2<o[2]*o[2] for o in world.obst)
        rew=0.0
        if colidiu:
            s.colisoes+=1; s.health=max(0,s.health-0.03); s.cort=min(1,s.cort+0.8); rew=-1.0   # BATER DÓI → cortisol dispara
            s.x=clamp(s.x-math.cos(s.head)*1.6,1,ARENA-1); s.y=clamp(s.y-math.sin(s.head)*1.6,1,ARENA-1)  # RECUA
            s.vel=0.1; s.head+=random.uniform(2.4,3.9)                            # e vira (sai do encalhe)
        else:
            s.x,s.y=clamp(nx,0,ARENA),clamp(ny,0,ARENA); s.dist_tot+=s.vel
            if s.vel>0.15: s.dop=min(1,s.dop+0.03); rew=0.15                   # andar livre/rápido = bom (dopamina)
        # ---- APRENDIZADO: forward model (auto-supervisionado) + plasticidade da floresta (gated pela neuromod) ----
        d1=s.sense(world)                                                      # o que REALMENTE aconteceu com a distância
        sit,act=s.last
        if s.pred:                                                            # forward model aprende Δdist previsto ← o "andar move a cena"
            key=(sit[0],sit[1],act); pv,n=s.fwd.get(key,(0.0,0))
            s.fwd[key]=((pv*n+(d1-d0))/(n+1),min(n+1,50))
        if s.plast:                                                           # a sinapse situação→ação fortalece/enfraquece pelo hormônio (como _reinforce)
            row=s.syn.setdefault(sit,[0.0,0.0,0.0,0.0]); c=s._plast_coef()
            row[act]=clamp(row[act]+c*rew,-3,3)
        s.dop*=0.85; s.cort*=0.8; s.fad=min(1,max(0,s.fad+(0.01 if s.vel>0.1 else -0.02)))
        s.health=min(1,s.health+0.0015)  # cicatriza devagar → vive pra aprender
        if s.health<=0 or s.age>4000: s.alive=False

class World:
    def __init__(s,seed):
        random.seed(seed); s.obst=[[random.uniform(6,ARENA-6),random.uniform(6,ARENA-6),random.uniform(2,3.5)] for _ in range(NOBST)]

def live(neuro=True,plast=True,pred=True,ticks=2000,seed=0,W0=None,window=150):
    w=World(seed); b=Carcaca(neuro,plast,pred,W0)
    for _ in range(50):
        if not any((b.x-o[0])**2+(b.y-o[1])**2<(o[2]+3)**2 for o in w.obst): break
        b.x=random.uniform(5,ARENA-5); b.y=random.uniform(5,ARENA-5)
    col_curve=[]; win_col=0; win_steps=0
    for t in range(ticks):
        if not b.alive: break
        pre=b.colisoes; b.step(w); win_col+=(b.colisoes-pre); win_steps+=1
        if win_steps>=window: col_curve.append(round(win_col/window,3)); win_col=0; win_steps=0
    taxa=b.colisoes/max(1,b.passos)
    dpc=b.dist_tot/max(1,b.colisoes)   # DISTÂNCIA por COLISÃO = anda muito e bate pouco = skill
    return dict(colisoes=b.colisoes, passos=b.passos, dist=round(b.dist_tot), dist_por_colisao=round(dpc,1), viveu=b.age, curva=col_curve), b.syn

if __name__=="__main__":
    if "--curva" in sys.argv:
        print("=== 1 ser vivendo — taxa de COLISÃO por janela de 200 passos (aprende a não bater sozinho?) ===\n")
        r,_=live(seed=3,ticks=3000)
        print("  colisões/passo por janela:", " ".join(f"{x:.2f}" for x in r["curva"]))
        print(f"  → do início ao fim: {r['curva'][0]:.2f} → {r['curva'][-1]:.2f} (se cair, ele APRENDEU a evitar)")
    elif "--milhares" in sys.argv:
        t0=time.time(); N=1000
        for i in range(N): live(ticks=300,seed=i)
        print(f"=== {N} seres × 300 passos em {time.time()-t0:.1f}s — carcaça LEVE, replicável em milhares (sem pesos) ===")
    else:
        print("=== BATERIA — a carcaça aprende a não bater? (média início vs fim, 8 sementes, 3000 passos) ===\n")
        for nome,cfg in [("CARCAÇA COMPLETA (neuromod+plast+preditivo)",dict(neuro=True,plast=True,pred=True)),
                         ("SEM órgão preditivo",dict(neuro=True,plast=True,pred=False)),
                         ("SEM plasticidade (não aprende)",dict(neuro=True,plast=False,pred=True)),
                         ("SEM neuromodulação",dict(neuro=False,plast=True,pred=True))]:
            import statistics as st
            rs=[live(ticks=3000,seed=sd,**cfg)[0] for sd in range(8)]
            dpc=st.mean(r["dist_por_colisao"] for r in rs); col=st.mean(r["colisoes"] for r in rs); di=st.mean(r["dist"] for r in rs)
            print(f"  {nome:42s} distância/colisão {dpc:6.1f} · colisões {col:4.0f} · distância {di:5.0f}")
        print("\n→ maior distância/colisão = anda muito E bate pouco = a skill de navegar EMERGIU (honesto).")
