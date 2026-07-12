#!/usr/bin/env python3
"""SURVIVAL BRAIN — testa a tese num domínio NÃO-linguístico: SOBREVIVÊNCIA (Leonardo 2026-07-07, "imitar um
cérebro mais simples, tipo Universe"). Cérebro imitado = C. elegans (quimiotaxia bilateral, ~pura sobrevivência)
= a linhagem do CfC do Universe (integrador leaky de tempo contínuo, i8/ternário). A criatura tem 2 sensores de
cheiro (esq/dir), precisa achar comida pra não morrer de fome. Cérebro evoluído por GA (como a genética do Universe).
VALIDA em 3 substratos: CfC-float | CfC-ternário (genética Universe) | reativo-sem-memória. Fitness = comida comida.
Pop-paralelo em numpy. CPU."""
import numpy as np, sys, time
rng = np.random.default_rng(0); t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

# ---- MUNDO + CRIATURA (pop-paralelo: P criaturas simuladas juntas) ----
I, H, O = 4, 6, 2            # entrada [cheiro_esq, cheiro_dir, energia, bias]; hidden CfC; saída [virar, acelerar]
NF, WORLD = 14, 1.0         # pellets de comida; mundo toroidal [0,1)²
SIG = 0.13                  # alcance do cheiro
EAT_R, FOOD_E, METAB, MOVE_C, MAXE = 0.045, 34.0, 7.0, 6.0, 130.0
DT, TSTEPS = 0.05, 500

def genome_size(): return H*I + O*H + H + 1     # W_in, W_out, decay(H), gain
def unpack(G):                                  # G: [P, genome_size] -> tensores
    P = G.shape[0]; i = 0
    Win = G[:, i:i+H*I].reshape(P,H,I); i += H*I
    Wout = G[:, i:i+O*H].reshape(P,O,H); i += O*H
    decay = 0.5 + 4.5*_sig(G[:, i:i+H]); i += H   # decay ∈ (0.5,5)
    gain = 0.5 + 2.0*_sig(G[:, i:i+1]); i += 1
    return Win, Wout, decay, gain
def _sig(x): return 1/(1+np.exp(-x))

def run_episode(G, mode="cfc", ternary=False, seed=1, bits=False, flicker=0.0):
    """Simula P criaturas. mode cfc=leaky memory; reactive=sem memória. ternary=pesos {-1,0,1}. bits=percepção 2-bit.
    flicker>0 = sensor INTERMITENTE (blackout aleatório) → tarefa que EXIGE memória pra atravessar o escuro."""
    P = G.shape[0]; r = np.random.default_rng(seed)
    Win, Wout, decay, gain = unpack(G)
    if ternary:                                 # genética do Universe: pesos ternários (sinal com limiar)
        Win = _tern(Win); Wout = _tern(Wout)
    pos = r.random((P,2)); head = r.random(P)*2*np.pi
    energy = np.full(P, MAXE*0.6); h = np.zeros((P,H)); eaten = np.zeros(P); alive = np.ones(P, bool)
    food = r.random((P,NF,2)); falive = np.ones((P,NF), bool)
    lifespan = np.zeros(P)
    for t in range(TSTEPS):
        # --- SENSORES: cheiro bilateral (quimiotaxia C. elegans) ---
        sL = pos + 0.03*np.stack([np.cos(head+0.6), np.sin(head+0.6)],1)   # antena esquerda
        sR = pos + 0.03*np.stack([np.cos(head-0.6), np.sin(head-0.6)],1)   # antena direita
        smellL = _smell(sL, food, falive); smellR = _smell(sR, food, falive)
        if flicker:                             # blackout sensorial: perde o cheiro em parte dos passos
            blk = r.random(P) < flicker; smellL = smellL*~blk; smellR = smellR*~blk
        en = energy/MAXE
        if bits:                                # percepção 2-bit (como o Draft Buffer do Universe)
            smellL = np.round(np.clip(smellL,0,1)*3)/3; smellR = np.round(np.clip(smellR,0,1)*3)/3; en = np.round(en*3)/3
        inp = np.stack([smellL, smellR, en, np.ones(P)],1)                 # [P,I]
        # --- CÉREBRO ---
        drive = np.tanh(np.einsum('phi,pi->ph', Win, inp)) * gain          # [P,H]
        if mode == "cfc": h = h*np.exp(-decay*DT) + drive*DT               # integrador leaky (tempo contínuo)
        else:             h = drive                                        # reativo: sem memória
        out = np.tanh(np.einsum('poh,ph->po', Wout, h))                    # [P,O] em (-1,1)
        turn, thrust = out[:,0], (out[:,1]+1)/2                            # thrust ∈ (0,1)
        # --- AGE NO MUNDO ---
        head = (head + turn*3.0*DT) % (2*np.pi)
        step = thrust[:,None]*np.stack([np.cos(head), np.sin(head)],1)*0.5*DT
        pos = (pos + step*alive[:,None]) % WORLD
        energy -= (METAB + MOVE_C*thrust)*DT * alive
        # --- COMER (comida dentro do raio) ---
        d = np.linalg.norm((pos[:,None,:]-food+0.5)%WORLD-0.5, axis=2)     # dist toroidal [P,NF]
        hit = (d < EAT_R) & falive & alive[:,None]
        got = hit.any(1)
        for p in np.where(got)[0]:
            j = np.argmax(hit[p]); falive[p,j]=False; food[p,j]=r.random(2); falive[p,j]=True  # respawn
            energy[p]=min(MAXE, energy[p]+FOOD_E); eaten[p]+=1
        alive &= energy > 0; lifespan += alive
    return eaten + lifespan/TSTEPS*0.5          # fitness: comida + bônus de sobrevivência

def _smell(p, food, falive):                    # concentração de cheiro no ponto p [P,2]
    d2 = (((p[:,None,:]-food+0.5)%WORLD-0.5)**2).sum(2)                    # [P,NF]
    return (np.exp(-d2/SIG**2)*falive).sum(1)
def _tern(W):                                   # ternariza: {-1,0,1} com limiar = 0.5*média|W| (estilo BitLinear)
    thr = 0.5*np.abs(W).mean(axis=(-1,-2), keepdims=True) + 1e-6
    return np.sign(W)*(np.abs(W) > thr)

def evolve(mode="cfc", ternary=False, bits=False, flicker=0.0, P=64, GEN=40, tag=""):
    G = rng.standard_normal((P, genome_size()))*0.8
    seeds = [11, 23, 37]                         # 3 layouts de comida (fitness robusto, anti-sorte)
    hist = []
    for g in range(GEN):
        fit = np.mean([run_episode(G, mode, ternary, s, bits, flicker) for s in seeds], 0)
        order = np.argsort(-fit); G = G[order]; fit = fit[order]; hist.append(fit[0])
        elite = G[:P//4]                                                   # top 25% reproduz
        kids = elite[rng.integers(0, len(elite), P-len(elite))] + rng.standard_normal((P-len(elite), G.shape[1]))*0.25
        G = np.vstack([elite, kids])
        if g % 10 == 0 or g == GEN-1: log(f"  {tag:16} gen {g:2d}  best {fit[0]:5.1f}  mean {fit.mean():5.1f}")
    # comportamento final do campeão: come e sobrevive?
    champ = G[:1]; final = np.mean([run_episode(champ, mode, ternary, s, bits, flicker)[0] for s in [101,102,103,104]])
    return hist, final

if __name__ == "__main__":
    if "--memtask" in sys.argv:
        # TAREFA QUE EXIGE MEMÓRIA: sensor intermitente (blackout 55%). Hipótese: agora o CfC bate o reativo.
        log("MEM-TASK — sensor INTERMITENTE (blackout 55%): a memória tem que atravessar o escuro\n")
        cf_h, cf = evolve(mode="cfc", flicker=0.55, tag="CfC (memória)")
        rc_h, rc = evolve(mode="reactive", flicker=0.55, tag="reativo (s/mem)")
        log(f"\n=== SOB SENSOR INTERMITENTE (OOD, mundos novos) ===")
        log(f"  CfC (memória)   {cf:5.1f} comidas")
        log(f"  reativo(s/mem)  {rc:5.1f} comidas")
        log(f"\nVEREDITO: quando a tarefa EXIGE memória, CfC/reativo = {cf/max(rc,0.1):.2f}× "
            f"({'memória PAGA' if cf>rc*1.1 else 'memória não ajudou'})")
        log("DONE"); sys.exit(0)
    log(f"SURVIVAL — imitando C. elegans (quimiotaxia) | cérebro CfC leaky (linhagem Universe) | genoma {genome_size()} pesos\n")
    # baseline: cérebros ALEATÓRIOS (sem evolução) — quanta comida pega no acaso?
    base = np.mean([run_episode(rng.standard_normal((200, genome_size()))*0.8, "cfc", False, s) for s in [11,23,37]],0)
    log(f"BASELINE (200 cérebros aleatórios, sem evolução): fitness médio {base.mean():.2f}, melhor {base.max():.1f}\n")
    res = {}
    for tag, kw in [("CfC-float", dict(mode="cfc")), ("CfC-ternário", dict(mode="cfc", ternary=True)),
                    ("reativo(s/mem)", dict(mode="reactive")), ("CfC-percep-2bit", dict(mode="cfc", bits=True))]:
        hist, final = evolve(tag=tag, **kw); res[tag] = (hist[-1], final)
        log(f"  → {tag:16} evoluído: best {hist[-1]:.1f} | campeão OOD(4 mundos novos) {final:.1f} comidas\n")
    log("=== RESUMO (fitness evoluído | campeão em mundos NOVOS) ===")
    for k,(h,f) in res.items(): log(f"  {k:16} {h:6.1f} | OOD {f:5.1f}")
    b = res["CfC-float"][1]
    log(f"\nVEREDITO: evolução vs acaso = {res['CfC-float'][0]:.1f} vs {base.mean():.1f} (>{res['CfC-float'][0]/max(base.mean(),0.1):.0f}×) | "
        f"ternário/float = {res['CfC-ternário'][1]/max(b,0.1):.0%} | memória importa? cfc/reativo = {b/max(res['reativo(s/mem)'][1],0.1):.1f}×")
    log("DONE")
