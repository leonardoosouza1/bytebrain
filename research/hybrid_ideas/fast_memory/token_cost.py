"""Tokens per function with the Qwen2.5-Coder tokenizer, and the cost of re-reading them.

Loads ONLY the tokenizer (no weights, no GPU). Prints per-file stats and the FLOP cost of
re-prefilling a function vs one bank lookup.
Usage: HF_HOME=... HF_HUB_OFFLINE=1 token_cost.py OUT.json FILE [FILE ...]
"""

import ast
import json
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# Qwen2.5-0.5B config: hidden 896, 24 layers, ffn 4864, 14 heads / 2 kv heads, vocab 151936, tied.
D, L, FF, H, KV, V = 896, 24, 4864, 14, 2, 151936
HD = D // H
PER_LAYER = D * D + 2 * D * KV * HD + D * D + 3 * D * FF   # q, k+v, o, gate/up/down (biases ignored)
NONEMB = L * PER_LAYER
TOTAL = NONEMB + V * D


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = {"params_nonembedding": NONEMB, "params_total": TOTAL, "files": {}}
    all_counts = []
    for p in sys.argv[2:]:
        src = Path(p).read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src)
        fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        texts = ["\n".join(lines[f.lineno - 1:f.end_lineno]) for f in fns]
        sigs = [lines[f.lineno - 1].strip() for f in fns]
        t = time.perf_counter_ns()
        counts = [len(tok(x)["input_ids"]) for x in texts]
        enc_ns = (time.perf_counter_ns() - t) / max(1, sum(counts))
        sig_counts = [len(tok(s)["input_ids"]) for s in sigs]
        whole = len(tok(src)["input_ids"])
        all_counts += counts
        out["files"][p] = dict(n_functions=len(fns), whole_file_tokens=whole,
                               fn_tokens_mean=statistics.mean(counts) if counts else 0,
                               fn_tokens_median=statistics.median(counts) if counts else 0,
                               fn_tokens_max=max(counts, default=0),
                               signature_tokens_mean=statistics.mean(sig_counts) if sig_counts else 0,
                               tokenizer_ns_per_token=enc_ns)
        print(f"{Path(p).name:22s} fns={len(fns):4d} whole={whole:6d} tok  fn mean={out['files'][p]['fn_tokens_mean']:.0f} "
              f"median={out['files'][p]['fn_tokens_median']} max={out['files'][p]['fn_tokens_max']}  "
              f"sig mean={out['files'][p]['signature_tokens_mean']:.1f}  tokenizer {enc_ns:.0f} ns/tok")
    flops_tok_task = 2 * 0.5e9
    flops_tok_nonemb = 2 * NONEMB
    out["all_fn_tokens_mean"] = statistics.mean(all_counts)
    out["flops_per_token_2x0.5B"] = flops_tok_task
    out["flops_per_token_2xnonemb"] = flops_tok_nonemb
    print(f"params non-embedding={NONEMB/1e6:.1f}M total={TOTAL/1e6:.1f}M; FLOPs/token 2*0.5B={flops_tok_task:.2e} "
          f"2*nonemb={flops_tok_nonemb:.2e}; mean fn tokens (all files)={out['all_fn_tokens_mean']:.1f}")
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
