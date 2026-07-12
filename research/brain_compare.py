#!/usr/bin/env python3
"""Phase B — compare the 'brains' BEHAVIORALLY: how different model TYPES formulate
sentences, do arithmetic, and complete formulas; and how their MLP neurons fire
(activation sparsity = how 'spiking' the brain is). CPU (no GPU contention)."""
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models"
PROMPTS = {
 "frase":   "A capital da França é a cidade de",
 "conta":   "Cálculo passo a passo: 23 * 47 =",
 "fewshot": "2+2=4\n5+7=12\n13+8=21\n40+6=",
 "formula": "A área de um círculo de raio r é dada pela fórmula A =",
}

def run(path, name):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    nb = sum(p.numel() for p in m.parameters()) / 1e9
    print(f"\n##### {name} ({nb:.2f}B) #####", flush=True)
    for k, p in PROMPTS.items():
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=36, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).replace("\n", " ⏎ ")
        print(f"  [{k:8}] …{p[-22:]!r} → {txt!r}", flush=True)
    # brain: MLP intermediate activation on the 'conta' last token, per layer
    acts = []
    def hook(mod, inp): acts.append(inp[0][0, -1].detach())
    hs = [L.mlp.down_proj.register_forward_pre_hook(hook) for L in m.model.layers]
    with torch.no_grad():
        m(tok(PROMPTS["conta"], return_tensors="pt").input_ids)
    for h in hs: h.remove()
    A = torch.stack(acts).abs()                      # [layers, intermediate]
    spars = (A < 0.01 * A.amax()).float().mean().item()
    print(f"  brain: {A.shape[0]} layers × {A.shape[1]} MLP neurons | "
          f"activation sparsity(<1% peak)={spars:.1%} | mean|act|={A.mean():.3f} | peak={A.amax():.1f}", flush=True)
    del m; gc.collect()

for path, name in [(f"{M}/qwen2.5-1.5b", "Qwen2.5-1.5B base"),
                   (f"{M}/Qwen2.5-Math-1.5B", "Qwen2.5-Math-1.5B"),
                   (f"{M}/SmolLM2-1.7B", "SmolLM2-1.7B")]:
    try:
        run(path, name)
    except Exception as e:
        print(f"  {name} FAILED: {e}", flush=True)
print("\nDONE brain_compare", flush=True)
