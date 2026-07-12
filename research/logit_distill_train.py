#!/usr/bin/env python3
"""LOGIT-LEVEL distillation — STAGE 2: train ByteBrain to MATCH Qwen's next-byte
distribution (KL divergence), not just reproduce its text. Starts from the fluent
ckpt_coh_scale and absorbs Qwen's soft distribution (the 'dark knowledge'). Resumable."""
import torch, numpy as np, torch.nn.functional as F, os, time, sys
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant

DEV = "cuda"
B = np.load("data/ld_bytes.npy")                   # [N] uint8
T = np.load("data/ld_targets.npy")                 # [N,256] float16 (Qwen next-byte dist)
N = len(B)
CTX, BATCH, STEPS, LR = 512, 16, 1500, 1e-4   # fewer steps (bigger clean corpus) → avoid overfit
CK, OUT = "ckpt_coh_scale/ckpt_best.pt", "ckpt_ld"
os.makedirs(OUT, exist_ok=True)
c = torch.load(CK, map_location=DEV, weights_only=False); cf = c["config"]; set_act_quant(0)
m = GraphByteGPT(cf["dim"], cf["layers"], cf["heads"], cf["ctx"],
                 topk=cf.get("topk", 0), mem=cf.get("mem", 0), topic=cf.get("topic", 0)).to(DEV)
m.load_state_dict(c["model"]); m.train()
opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=0.05)
scaler = torch.amp.GradScaler("cuda")

def batch():
    ix = np.random.randint(0, N - CTX - 1, BATCH)
    x = np.stack([B[i:i + CTX] for i in ix]).astype(np.int64)
    t = np.stack([T[i:i + CTX] for i in ix]).astype(np.float32)
    return torch.tensor(x, device=DEV), torch.tensor(t, device=DEV)

start = 0
if os.path.exists(f"{OUT}/ckpt.pt"):
    s = torch.load(f"{OUT}/ckpt.pt", map_location=DEV, weights_only=False)
    m.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); start = s["step"]
    print(f"RESUME from {start}", flush=True)
print(f"LOGIT-DISTILL | {N} bytes | KL→Qwen byte-dist | init {CK}", flush=True)
t0 = time.time()
for step in range(start + 1, STEPS + 1):
    x, tgt = batch()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = m(x)                              # [B,L,256], predicts next byte at each pos
    loss = F.kl_div(F.log_softmax(logits.float(), -1).reshape(-1, 256),
                    tgt.reshape(-1, 256), reduction="batchmean")
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
    if step % 100 == 0:
        print(f"step {step} | kl {loss.item():.4f} | {(time.time()-t0)/60:.1f}m", flush=True)
    if step % 500 == 0:
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step, "config": cf}, f"{OUT}/ckpt.pt")
torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": STEPS, "best": 0.0, "config": cf}, f"{OUT}/ckpt_best.pt")
open(f"{OUT}/DONE", "w").close()
print("DONE logit_distill_train", flush=True)
