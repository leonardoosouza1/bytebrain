#!/usr/bin/env python3
"""LOGIT-LEVEL byte-marginalized distillation — STAGE 1: precompute Qwen byte targets.
Qwen predicts next TOKEN (vocab 151936); ByteBrain predicts next BYTE (256). We convert
Qwen's next-token distribution into a next-BYTE distribution by marginalizing tokens by
their FIRST byte. Soft target at token boundaries (where Qwen decides = where knowledge
lives); one-hot within a token (deterministic spelling). Saves byte seq + [N,256] targets.
Run: --sanity  (verify the marginalization) | (no arg) full corpus precompute. GPU."""
import torch, json, numpy as np, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).to(DEV).eval()

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
b2u = bytes_to_unicode(); u2b = {u: b for b, u in b2u.items()}
vocab = json.load(open(f"{M}/vocab.json")); inv = {v: k for k, v in vocab.items()}

EMB = m.get_input_embeddings().weight.shape[0]
fb = np.full(EMB, 256, dtype=np.int64)                     # bin 256 = invalid/special token
for s, tid in vocab.items():
    if tid < EMB and len(s) > 0 and s[0] in u2b: fb[tid] = u2b[s[0]]
fb_t = torch.tensor(fb, device=DEV)

@torch.no_grad()
def qwen_byte_targets(text):
    ids = tok(text, add_special_tokens=False).input_ids
    blens = [len(inv[i]) for i in ids]                     # bytes per token (1 char = 1 byte in BPE space)
    bseq = text.encode("utf-8")
    if sum(blens) != len(bseq): return None, None          # skip if byte-misaligned (rare special chars)
    probs = torch.softmax(m(torch.tensor([ids], device=DEV)).logits[0].float(), -1)  # [T,vocab] P(next tok)
    bd = torch.zeros(probs.shape[0], 257, device=DEV)
    bd.index_add_(1, fb_t, probs)                          # marginalize tokens → first-byte bins
    bd = bd[:, :256]; bd = bd / bd.sum(1, keepdim=True).clamp_min(1e-9)   # [T,256] next-first-byte dist
    bd = bd.cpu().numpy()
    N = len(bseq); targ = np.zeros((N, 256), dtype=np.float16); pos = 0
    for k, L in enumerate(blens):
        s, e = pos, pos + L
        for j in range(s, e - 1):                          # within token: deterministic next byte
            targ[j, bseq[j + 1]] = 1.0
        if e - 1 < N - 1:                                  # boundary byte: soft = Qwen's marginalized dist
            targ[e - 1] = bd[k].astype(np.float16)
        pos = e
    return np.frombuffer(bseq, dtype=np.uint8).copy(), targ

if __name__ == "__main__":
    if "--sanity" in sys.argv:
        for t in ["A capital do Brasil é a cidade de Brasília.",
                  "A água é composta por hidrogênio e oxigênio.", "2 mais 2 é igual a 4."]:
            bseq, targ = qwen_byte_targets(t)
            print(f"\n{t!r}")
            for i in range(len(bseq) - 1):
                row = targ[i]
                if (row > 0.02).sum() > 1:                 # a soft (boundary) position
                    top = row.argsort()[::-1][:4]
                    ctx = bytes(bseq[max(0, i - 14):i + 1]).decode("utf-8", "replace")
                    alts = "  ".join(f"{(chr(b) if 32<=b<127 else '·'+str(b))}={row[b]:.2f}" for b in top)
                    print(f"   …{ctx!r:18} → {alts}")
        sys.exit()
    # full corpus: chunk by lines/paragraphs to bound memory
    raw = open("data/pt_big.txt", encoding="utf-8", errors="replace").read(1_400_000)   # CLEAN Wikipedia (pt_overnight = project notes!)
    chunks, cur = [], ""
    for line in raw.splitlines():
        cur += line + "\n"
        if len(cur.encode()) > 350: chunks.append(cur); cur = ""
        if len(chunks) >= 2200: break
    all_b, all_t = [], []
    for i, c in enumerate(chunks):
        b, t = qwen_byte_targets(c)
        if b is not None: all_b.append(b); all_t.append(t)
        if i % 50 == 0: print(f"  chunk {i}/{len(chunks)} | bytes so far {sum(len(x) for x in all_b)}", flush=True)
    B = np.concatenate(all_b); T = np.concatenate(all_t)
    np.save("data/ld_bytes.npy", B); np.save("data/ld_targets.npy", T)
    print(f"SAVED {len(B)} bytes, targets {T.shape} → data/ld_bytes.npy, data/ld_targets.npy")
    print("DONE logit_distill_precompute")
