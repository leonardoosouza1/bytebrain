#!/usr/bin/env python3
"""IARA-ÁGUA × CARCAÇA (v2, com a FÍSICA VALIDADA) — seres-átomo SOBREVIVEM devorando o resíduo de uma imagem.
Junta as gemas que venceram (ver GEMAS_IARA.md):
 · amp em FORMA FECHADA (⟨r,ψ⟩/‖ψ‖²) — o ser BEBE o coeficiente ótimo (sem isso EXPLODE, foi o bug da v1);
 · relaxação GAUSS-SEIDEL — cada átomo refita o resíduo DEPOIS dos outros (harmonização, sem duplo-cômputo);
 · LEI DO METABOLISMO (AIC) — quem explica menos que met_c morre → a população de equilíbrio EMERGE;
 · CARCAÇA IARA (neuromod) esculpe a GEOMETRIA (freq/pos/fase) rumo a mais comida; a amp é sempre forma-fechada.
Estabilizar = a soma dos vivos reconstrói a imagem = HARMONIA. Headless: PSNR sobe = harmoniza.
Uso: python3 iara_agua_carcaca.py [--watch]"""
import numpy as np, sys, time
np.random.seed(0)
H=W=48
def alvo():
    y,x=np.mgrid[0:H,0:W].astype(np.float64)
    img =0.6*np.exp(-(((x-16)**2+(y-16)**2)/60))
    img+=0.5*np.exp(-(((x-34)**2+(y-30)**2)/40))
    img+=0.25*(x/W)+0.2*((np.sin(x*0.9)*np.sin(y*0.9))>0)+0.15*(x>36)
    return np.clip(img,0,1).astype(np.float64)
TARGET=alvo()
YY,XX=np.mgrid[0:H,0:W].astype(np.float64)

def gabor(cx,cy,freq,theta,phase,sigma):
    xr=(XX-cx)*np.cos(theta)+(YY-cy)*np.sin(theta); yr=-(XX-cx)*np.sin(theta)+(YY-cy)*np.cos(theta)
    return (np.exp(-(xr*xr+yr*yr)/(2*sigma*sigma))*np.cos(2*np.pi*freq*xr+phase))

class Atomo:
    _id=0
    def __init__(s,cx,cy,sigma=6.0):
        s.id=Atomo._id; Atomo._id+=1
        s.cx=cx; s.cy=cy; s.sigma=sigma
        s.freq=np.random.uniform(0.03,0.35); s.theta=np.random.uniform(0,np.pi); s.phase=np.random.uniform(0,2*np.pi)
        s.amp=0.0; s.g=gabor(cx,cy,s.freq,s.theta,s.phase,sigma); s.food=0.0
        s.dop=s.cort=s.cur=0.0
    def params(s): return (s.cx,s.cy,s.freq,s.theta,s.phase,s.sigma)
    def _fit(s,resid,g):                              # amp FORMA-FECHADA + energia explicada (comida)
        gn=float((g*g).sum())+1e-9
        amp=float((g*resid).sum())/gn
        food=amp*amp*gn                              # = energia realmente removida do resíduo
        return amp,food
    def tune(s,resid):                               # a CARCAÇA esculpe a GEOMETRIA rumo a mais comida (neuromod gate)
        base_amp,base_food=s._fit(resid,s.g); best=base_food; bg=s.g; bp=s.params()
        step=0.5 if s.dop>0.3 else 1.5               # confiante→passo fino; faminto→explora
        for _ in range(3):
            f2=float(np.clip(s.freq+np.random.uniform(-.04,.04)*step,0.02,0.45))
            t2=(s.theta+np.random.uniform(-.4,.4)*step)%np.pi
            ph=(s.phase+np.random.uniform(-.6,.6)*step)%(2*np.pi)
            cx=float(np.clip(s.cx+np.random.uniform(-1.5,1.5)*step,2,W-2)); cy=float(np.clip(s.cy+np.random.uniform(-1.5,1.5)*step,2,H-2))
            g=gabor(cx,cy,f2,t2,ph,s.sigma); _,fd=s._fit(resid,g)
            if fd>best: best=fd; bg=g; bp=(cx,cy,f2,t2,ph,s.sigma)
        s.cx,s.cy,s.freq,s.theta,s.phase,s.sigma=bp; s.g=bg
        s.amp,s.food=s._fit(resid,s.g)
        if s.food>base_food+1e-9: s.dop=min(1,s.dop+0.4); s.cur=max(0,s.cur-0.05)   # melhorou = dopamina
        else: s.cort=min(1,s.cort+0.3); s.cur=min(1,s.cur+0.1)                        # estagnou = cortisol/curiosidade
        s.dop*=0.85; s.cort*=0.8
        return s.amp*s.g

def psnr(a,b):
    mse=float(((a-b)**2).mean()); return 99.0 if mse<1e-9 else 10*np.log10(1.0/mse)

def run(epocas=140,n0=6,cap=110,met_c=0.02,narrate=False,watch=False):
    atoms=[Atomo(np.random.uniform(4,W-4),np.random.uniform(4,H-4)) for _ in range(n0)]
    recon=np.zeros((H,W)); hist=[]
    for a in atoms: recon+=a.amp*a.g
    for ep in range(epocas):
        for a in atoms:                              # GAUSS-SEIDEL: cada um refita o resíduo DEPOIS dos outros
            recon-=a.amp*a.g
            resid=TARGET-recon
            contrib=a.tune(resid)                    # esculpe geometria (sobrevivência) + amp forma-fechada
            recon+=contrib
        # LEI DO METABOLISMO: quem explica menos que met_c morre; devolve o patch ao resíduo
        keep=[]
        for a in atoms:
            if a.food>=met_c: keep.append(a)
            else: recon-=a.amp*a.g
        atoms=keep
        # CHUVA ∝ resíduo²: coloniza o maior buraco não-explicado (a pirâmide emerge, grosso→fino)
        resid=TARGET-recon; hist.append(psnr(TARGET,recon))
        if len(atoms)<cap:
            r2=np.abs(resid).copy(); r2[:2]=0;r2[-2:]=0;r2[:,:2]=0;r2[:,-2:]=0
            cy,cx=np.unravel_index(int(np.argmax(r2)),r2.shape)
            if r2[cy,cx]>0.03:
                sig=float(np.random.choice([3.0,6.0,10.0]))
                na=Atomo(float(cx)+np.random.uniform(-2,2),float(cy)+np.random.uniform(-2,2),sigma=sig)
                na.amp,na.food=na._fit(TARGET-recon,na.g); recon+=na.amp*na.g; atoms.append(na)
        if narrate and ep%14==0:
            fr=np.array([a.freq for a in atoms])
            print(f"  época {ep:3d} · vivos {len(atoms):3d} · PSNR {hist[-1]:6.2f} dB · freq {fr.min():.2f}–{fr.max():.2f}")
    if watch:
        try:
            from PIL import Image
            comp=np.concatenate([TARGET,np.clip(recon,0,1)],axis=1)
            Image.fromarray((comp*255).astype(np.uint8)).resize((W*10,H*5),Image.NEAREST).save("reconstrucao.png")
            print("  [reconstrucao.png: alvo | reconstruído pelos seres]")
        except Exception as e: print("  (PIL:",e,")")
    return atoms,hist,recon

if __name__=="__main__":
    print("=== IARA-ÁGUA v2: seres-átomo SOBREVIVEM devorando o resíduo (física validada: forma-fechada+Gauss-Seidel+metabolismo) ===\n")
    t0=time.time(); atoms,hist,recon=run(narrate=True,watch="--watch" in sys.argv)
    print(f"\nPSNR: {hist[0]:.2f} → {max(hist):.2f} dB — a imagem EMERGIU da sobrevivência dos seres · {time.time()-t0:.1f}s")
    fr=np.array([a.freq for a in atoms]); sg=np.array([a.sigma for a in atoms])
    print(f"população final: {len(atoms)} átomos (emergiu da lei do metabolismo) · freq {fr.min():.2f}–{fr.max():.2f} · escalas σ {sorted(set(sg.round()))} = PIRÂMIDE")
    print("\nCÉREBRO/RELAÇÃO de cada ser (os melhor-nutridos): freq+pos distintas → juntos TILAM a imagem:")
    for a in sorted(atoms,key=lambda a:-a.food)[:8]:
        print(f"  átomo#{a.id:3d} freq {a.freq:.2f} σ{a.sigma:.0f} @({a.cx:4.1f},{a.cy:4.1f}) amp {a.amp:+.2f} comida {a.food:.3f} | dop {a.dop:.1f} cort {a.cort:.1f}")
