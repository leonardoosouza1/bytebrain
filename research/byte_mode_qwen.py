#!/usr/bin/env python3
"""Phase C — "the SAME Qwen, but in BYTE mode". Qwen's vocab has all 256 byte
tokens, so we can feed text byte-by-byte (mapping each byte to its byte-token id)
instead of via BPE merges, using the SAME weights. Measures: how much worse is
Qwen's per-byte compression when forced to read bytes (the value of the tokenizer),
and can it still generate coherently byte-by-byte. CPU."""
import torch, json, math
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

vocab = json.load(open(f"{M}/vocab.json"))
b2u = bytes_to_unicode(); u2b = {u: b for b, u in b2u.items()}
inv = {v: k for k, v in vocab.items()}
byte2id = [vocab[b2u[b]] for b in range(256)]

def bpb(ids, nbytes):
    t = torch.tensor([ids])
    with torch.no_grad():
        loss = m(t, labels=t).loss.item()
    return loss * (len(ids) - 1) / math.log(2) / nbytes

TEXTS = [
    "A capital da França é Paris, uma das cidades mais famosas da Europa.",
    "The quadratic formula solves ax^2 + bx + c = 0 for x.",
    "23 * 47 = 1081 and 100 - 37 = 63.",
]
print("# Qwen2.5-1.5B: BPE vs BYTE-mode (same weights), bits-per-byte")
print(f"{'text':36} {'BPE tok':>7} {'byte tok':>8} {'bpb BPE':>8} {'bpb byte':>9} {'ratio':>6}")
for txt in TEXTS:
    nb = len(txt.encode())
    bpe = tok(txt).input_ids
    byt = [byte2id[b] for b in txt.encode()]
    rb, rby = bpb(bpe, nb), bpb(byt, nb)
    print(f"{txt[:36]:36} {len(bpe):>7} {len(byt):>8} {rb:>8.3f} {rby:>9.3f} {rby/rb:>5.2f}x", flush=True)

# byte-mode generation: feed a byte-prompt, decode tokens back to bytes
def byte_gen(prompt, n=80):
    ids = torch.tensor([[byte2id[b] for b in prompt.encode()]])
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    bs = bytearray()
    for tid in out[0].tolist():
        for c in inv.get(tid, ""):
            if c in u2b: bs.append(u2b[c])
    return bytes(bs).decode("utf-8", "replace")

for p in ["A capital da França é", "23 * 47 ="]:
    print(f"\nbyte-mode gen  {p!r}\n  → {byte_gen(p)!r}", flush=True)
print("\nDONE byte_mode_qwen", flush=True)
