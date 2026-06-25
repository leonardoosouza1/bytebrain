"""
ByteBrain — OVERNIGHT LOOP autonomo. A cada ciclo: reconstroi o corpus LIMPO (que cresce com
agents+nlm+wiki), continua o treino do checkpoint, mede wtrans, salva o melhor por wtrans.
Roda ~10h. Robusto (try/except). De manha: corpus grande + modelo treinado + grafico do wtrans.
"""
import os, re, math, time, json, glob, hashlib
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV = "cuda"
BASE = os.path.dirname(os.path.abspath(__file__))
CKDIR = f"{BASE}/overnight_ck"; os.makedirs(CKDIR, exist_ok=True)
TOTAL_H = 10.0; CYCLE_MIN = 25; LB = 256; BATCH = 56
D, NL, NH = 512, 8, 8     # ~25M: zona eficiente RDNA2; corpus 4.4MB+ suporta sem overfit; guards (dropout/wd/epoch-cap/auto-revert) ativos
PT_SW = re.compile(r"\b(que|não|para|uma|com|de|do|da|em|os|as|ele|ela|mais|como|foi|são|seu|sua|por|isso|este|esta|quando|porque|também|entre|sobre|ser|ter|sem)\b", re.I)
EN_SW = re.compile(r"\b(the|and|of|with|is|are|this|that|for|from|was|were|which|their|have|will)\b", re.I)
ACC = set("ãõáéíóúâêôàçÃÕÁÉÍÓÚÂÊÔÀÇ")
CODE = re.compile(r"[{}<>]|;\s|=>|::|def |function |import |return |```|##| \| |https?://|www\.|R\$|MODEL:|USER:|\bself\b|\bconst\b|\bvoid\b")


def good(p):
    p = p.strip()
    if not (120 <= len(p) <= 3000): return False
    w = re.findall(r"\S+", p)
    if len(w) < 20 or CODE.search(p): return False
    pt = len(PT_SW.findall(p)); en = len(EN_SW.findall(p))
    if pt < len(w)*0.07 or en > pt: return False
    if sum(c in ACC for c in p)/len(p) < 0.004: return False
    if sum(c.isdigit() or c in "|/\\[]{}*#=+`~_" for c in p)/len(p) > 0.12: return False
    if p.count(". ")+p.count("? ")+p.count("! ") < 2: return False
    return True


def build_corpus():
    files = glob.glob(f"{BASE}/data/agents/*.txt") + glob.glob(f"{BASE}/data/raw/*.txt") + [f"{BASE}/data/multiscript/pt.txt"]
    seen = set(); kept = []
    for fp in files:
        try: txt = open(fp, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        for para in re.split(r"\n\s*\n", txt):
            para = re.sub(r"[ \t]+", " ", para).strip()
            if good(para):
                h = hashlib.md5(para[:120].encode()).hexdigest()
                if h not in seen: seen.add(h); kept.append(para)
    blob = "\n\n".join(kept)
    open(f"{BASE}/data/pt_clean.txt", "w", encoding="utf-8").write(blob)
    return blob


def wtrans_stats(blob):
    Wd = re.findall(r"[a-zàáâãéêíóôõúüç]+", blob.lower()); WUNI = Counter(Wd); WBI = Counter(zip(Wd, Wd[1:])); WV = len(WUNI)
    def wt(t):
        w = re.findall(r"[a-zàáâãéêíóôõúüç]+", t.lower())
        if len(w) < 3: return 12.0
        return float(np.mean([-math.log((WBI.get((w[i], w[i+1]), 0)+0.05)/(WUNI.get(w[i], 0)+0.05*WV)) for i in range(len(w)-1)]))
    return wt


class Blk(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.l1 = nn.LayerNorm(d); s.l2 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3*d); s.pr = nn.Linear(d, d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Dropout(0.15), nn.Linear(4*d, d)); s.do = nn.Dropout(0.15)
    def forward(s, x):
        Bn, L, Dd = x.shape; h = s.l1(x); q = s.qkv(h).view(Bn, L, 3, s.nh, Dd//s.nh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q[0], q[1], q[2], is_causal=True, dropout_p=0.1 if s.training else 0.0); x = x + s.do(s.pr(a.transpose(1, 2).reshape(Bn, L, Dd))); return x + s.do(s.mlp(s.l2(x)))


class GPT(nn.Module):
    def __init__(s, d=D, nl=NL, nh=NH):
        super().__init__(); s.t = nn.Embedding(256, d); s.p = nn.Embedding(LB, d); s.b = nn.ModuleList([Blk(d, nh) for _ in range(nl)]); s.f = nn.LayerNorm(d); s.o = nn.Linear(d, 256)
    def forward(s, x):
        h = s.t(x)+s.p(torch.arange(x.size(1), device=x.device))[None]
        for b in s.b: h = b(h)
        return s.o(s.f(h))


def batch(a, bs, L=160):
    ix = np.random.randint(0, len(a)-L-1, bs); return torch.from_numpy(np.stack([a[i:i+L] for i in ix]).astype(np.int64)).to(DEV)


def gen(g, n=300, temp=0.6, top_p=0.85, rep=1.4):
    # top-p (nucleus) + penalidade de repeticao nos ultimos 48 bytes: mata o colapso "e e e" e estende o span coerente
    x = torch.tensor([list("O ".encode())], device=DEV)
    with torch.no_grad():
        for _ in range(n):
            lo = g(x[:, -LB:])[0, -1].clone()
            for b in set(x[0, -48:].tolist()): lo[b] /= rep
            p = F.softmax(lo/temp, -1); sp, si = torch.sort(p, descending=True); cum = torch.cumsum(sp, 0)
            keep = cum <= top_p; keep[0] = True; sp = sp*keep; sp = sp/sp.sum()
            x = torch.cat([x, si[torch.multinomial(sp, 1)].view(1, 1)], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def main():
    g = GPT().to(DEV); opt = torch.optim.AdamW(g.parameters(), 2.5e-4, weight_decay=0.1)
    if os.path.exists(f"{CKDIR}/loop.pt"):
        try: g.load_state_dict(torch.load(f"{CKDIR}/loop.pt", map_location=DEV)); print("checkpoint carregado", flush=True)
        except Exception: pass
    P = sum(p.numel() for p in g.parameters()); print(f"OVERNIGHT LOOP | gerador {P/1e6:.0f}M | {TOTAL_H}h, ciclos de {CYCLE_MIN}min", flush=True)
    metrics = {"ciclos": []}; best = 99.0; best_val = 99.0; overfit = 0; t0 = time.time(); cyc = 0
    while time.time()-t0 < TOTAL_H*3600:
        try:
            blob = build_corpus(); B = np.frombuffer(blob.encode("utf-8"), np.uint8)
            if len(B) < 5000: time.sleep(60); continue
            cut = int(len(B)*0.97); TR, VA = B[:cut], B[cut:]; wt = wtrans_stats(blob)
            cyc += 1; print(f"\n--- ciclo {cyc} | corpus {len(B)/1e6:.2f}MB | {(time.time()-t0)/3600:.1f}h ---", flush=True)
            g.train(); ce = time.time(); step = 0
            max_steps = max(150, int(2.0*len(TR)/(BATCH*160)))   # <=2 epocas/ciclo: ANTI-OVERFIT (cresce com o corpus)
            while step < max_steps and time.time()-ce < CYCLE_MIN*60 and time.time()-t0 < TOTAL_H*3600:
                x = batch(TR, BATCH); loss = F.cross_entropy(g(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1))
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); opt.step(); step += 1
                if step % 200 == 0: print(f"    step {step}/{max_steps} bpb {loss.item()/math.log(2):.2f}", flush=True)
            g.eval(); vb = 0
            with torch.no_grad():
                for _ in range(10): x = batch(VA, 32); vb += F.cross_entropy(g(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1)).item()
            vb = vb/10/math.log(2); smp = [gen(g, 280) for _ in range(6)]; mw = float(np.mean([wt(s) for s in smp]))
            torch.save(g.state_dict(), f"{CKDIR}/loop.pt")
            # --- AUTO-VALIDACAO + AUTO-CORRECAO: nao espera ninguem ---
            lr = opt.param_groups[0]["lr"]; status = "ok"
            if vb < best_val - 0.01:                              # generaliza melhor -> novo melhor
                best_val = vb; best = mw; overfit = 0; torch.save(g.state_dict(), f"{CKDIR}/loop_best.pt")
            elif vb > best_val + 0.4:                             # val SUBINDO = overfit
                overfit += 1; status = f"val subiu ({overfit}/2)"
                if overfit >= 2 and os.path.exists(f"{CKDIR}/loop_best.pt"):
                    g.load_state_dict(torch.load(f"{CKDIR}/loop_best.pt", map_location=DEV))  # reverte
                    torch.save(g.state_dict(), f"{CKDIR}/loop.pt")                            # continua do bom
                    lr = max(lr*0.6, 4e-5); [pg.update(lr=lr) for pg in opt.param_groups]     # baixa LR
                    status = f"OVERFIT->revertido+LR={lr:.1e}"; overfit = 0
            metrics["ciclos"].append({"ciclo": cyc, "corpus_MB": round(len(B)/1e6, 2), "horas": round((time.time()-t0)/3600, 2), "val_bpb": round(vb, 3), "melhor_val": round(best_val, 3), "wtrans": round(mw, 2), "lr": round(lr, 6), "status": status, "amostra": smp[0][:400]})
            json.dump(metrics, open(f"{BASE}/overnight_loop_metrics.json", "w"), indent=2, ensure_ascii=False)
            print(f"  ciclo {cyc}: val_bpb {vb:.2f} (melhor {best_val:.2f}) | wtrans {mw:.2f} | {step} steps | STATUS: {status}", flush=True)
            print(f"  amostra: {smp[0][:200]!r}", flush=True)
            time.sleep(150)   # deixa fetchers/agents crescerem o corpus antes do proximo ciclo (anti-overfit)
        except Exception as e:
            print(f"  ERRO ciclo: {e}", flush=True); time.sleep(30)
    print(f"\n=== OVERNIGHT FIM: {cyc} ciclos, melhor wtrans {best:.2f} ===", flush=True)


if __name__ == "__main__":
    main()
