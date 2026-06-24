"""
ByteBrain — TREINO OVERNIGHT (8h) especialista PT-BR em corpus rico (nlm + Wikipedia).
40M, fp32 (rapido na RDNA2). Checkpoint + amostras + val bpb A CADA HORA -> relatorio de manha.
Loga bpb/ms-step/tok-s/VRAM. Salva melhor por val bpb. GPU.
"""
import os, math, time, json, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda"
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt"
CKDIR = "/home/leonardo/projects/LLM/bytebrain/overnight_ck"
os.makedirs(CKDIR, exist_ok=True)
LB = 256; BATCH = 48; D, NL, NH = 640, 8, 10
HOURS = 8; LOGEVERY = 100; CKEVERY = 3600  # 1h
LR, WARM = 3e-4, 500


class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, L, Dd = x.shape; h = s.ln1(x); qkv = s.qkv(h).view(B, L, 3, s.nh, Dd//s.nh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, L, Dd)); return x + s.mlp(s.ln2(x))


class GPT(nn.Module):
    def __init__(s, d, nl, nh):
        super().__init__(); s.tok = nn.Embedding(256, d); s.pos = nn.Embedding(LB, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)]); s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, 256)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))[None]
        for b in s.blocks: h = b(h)
        return s.head(s.lnf(h))


def main():
    blob = open(CORPUS, encoding="utf-8").read().encode("utf-8")
    arr = np.frombuffer(blob, np.uint8); cut = int(len(arr)*0.97)
    tr, va = arr[:cut], arr[cut:]
    print(f"corpus PT {len(arr)/1e6:.1f}MB | train {len(tr)/1e6:.1f}MB | epoca~{len(tr)//(BATCH*LB)} steps", flush=True)

    def batch(a, bs):
        ix = np.random.randint(0, len(a)-LB-1, bs)
        return torch.from_numpy(np.stack([a[i:i+LB] for i in ix]).astype(np.int64)).to(DEV)

    model = GPT(D, NL, NH).to(DEV); P = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=0.1)
    tot_steps = int(HOURS*3600/0.46)
    def lr_at(s):
        if s < WARM: return LR*s/WARM
        import math as m; return LR*(0.1 + 0.9*0.5*(1+m.cos(m.pi*(s-WARM)/max(1, tot_steps-WARM))))
    print(f"MODELO {P/1e6:.0f}M (d{D} L{NL} h{NH}) | fp32 batch {BATCH} | alvo {HOURS}h ~{tot_steps} steps", flush=True)

    def val_bpb():
        model.eval(); tot = 0; n = 0
        with torch.no_grad():
            for _ in range(20):
                x = batch(va, 32); lo = model(x[:, :-1]); tot += F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1)).item(); n += 1
        model.train(); return tot/n/math.log(2)

    def gen(seed, n=200, temp=0.8):
        model.eval(); x = torch.tensor([list(seed.encode("utf-8"))], device=DEV)
        with torch.no_grad():
            for _ in range(n):
                lo = model(x[:, -LB:]); p = F.softmax(lo[0, -1]/temp, -1); x = torch.cat([x, torch.multinomial(p, 1)[None]], 1)
        model.train(); return bytes(x[0].tolist()).decode("utf-8", "ignore")

    metrics = {"params_M": round(P/1e6, 1), "corpus_MB": round(len(arr)/1e6, 1), "horas": {}}
    seeds = ["O Brasil ", "A historia ", "A inteligencia artificial ", "Em 2026, ", "O projeto "]
    model.train(); t0 = time.time(); step = 0; tstep = 0; last_ck = 0; best = 9e9
    while time.time()-t0 < HOURS*3600:
        for g in opt.param_groups: g["lr"] = lr_at(step)
        s0 = time.time(); x = batch(tr, BATCH); lo = model(x[:, :-1]); loss = F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; tstep += time.time()-s0
        if step % LOGEVERY == 0:
            sps = tstep/step; pk = torch.cuda.max_memory_allocated()/1e9
            print(f"  step {step} | bpb {loss.item()/math.log(2):.3f} | lr {lr_at(step):.1e} | {sps*1000:.0f}ms | {BATCH*255/sps:.0f}tok/s | VRAM pico {pk:.1f} | {(time.time()-t0)/3600:.2f}h", flush=True)
        if time.time()-t0 - last_ck >= CKEVERY:
            last_ck = time.time()-t0; h = round(last_ck/3600, 1); vb = val_bpb()
            torch.save(model.state_dict(), f"{CKDIR}/overnight_h{h}.pt")
            if vb < best: best = vb; torch.save(model.state_dict(), f"{CKDIR}/overnight_best.pt")
            sm = {s: gen(s, 180) for s in seeds}
            metrics["horas"][str(h)] = {"val_bpb": round(vb, 3), "step": step, "amostras": sm}
            json.dump(metrics, open("/home/leonardo/projects/LLM/bytebrain/overnight_metrics.json", "w"), indent=2, ensure_ascii=False)
            print(f"\n===== CHECKPOINT {h}h | val bpb {vb:.3f} (melhor {best:.3f}) | step {step} =====", flush=True)
            for s, t in sm.items(): print(f"  [{s!r}] {t[:140]!r}", flush=True)
            print("", flush=True)
    vb = val_bpb(); torch.save(model.state_dict(), f"{CKDIR}/overnight_final.pt")
    metrics["final"] = {"val_bpb": round(vb, 3), "step": step, "melhor_val_bpb": round(best, 3),
                        "amostras": {s: gen(s, 220) for s in seeds}}
    json.dump(metrics, open("/home/leonardo/projects/LLM/bytebrain/overnight_metrics.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n========== OVERNIGHT FIM: {step} steps, val bpb {vb:.3f} (melhor {best:.3f}) ==========", flush=True)
    for s, t in metrics["final"]["amostras"].items(): print(f"  [{s!r}] {t[:200]!r}", flush=True)


if __name__ == "__main__":
    main()
