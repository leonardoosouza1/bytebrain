#!/usr/bin/env python3
"""Export a ByteBrain (ByteGPT) checkpoint to safetensors + config.json so the
iara-engine (llm-lab) can load and serve it natively on the GPU.

ByteGPT is a plain decoder-only transformer:
  vocab 256 · learned positional embeddings · LayerNorm · full MHA · GELU MLP.
(No RoPE / RMSNorm / GQA / SwiGLU — simpler than Qwen.)

Run inside the PyTorch (ROCm) container, e.g.:
  docker exec saci-brain python /work/bytebrain/export_to_safetensors.py \
      bytebrain/ckpt_big2/ckpt_best.pt --heads 12 --out bytebrain/export_bytebrain
"""
import argparse, json, os, sys
import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to ByteBrain .pt (train.py or overnight_loop format)")
    ap.add_argument("--out", default="export_bytebrain", help="output directory")
    ap.add_argument("--heads", type=int, default=0, help="num attention heads (0 = guess dim//64)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f32"])
    args = ap.parse_args()

    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    # Normalize legacy short names (overnight_loop) -> clean module names.
    def rename(k):
        if k.startswith("t."): return "tok." + k[2:]
        if k.startswith("p."): return "pos." + k[2:]
        if k.startswith("f."): return "lnf." + k[2:]
        if k.startswith("o."): return "head." + k[2:]
        if k.startswith("b."):
            rest = k[2:].replace(".l1.", ".ln1.").replace(".l2.", ".ln2.").replace(".pr.", ".proj.")
            return "blocks." + rest
        return k
    sd = {rename(k): v for k, v in sd.items()}

    # Infer architecture from tensor shapes.
    dim = sd["tok.weight"].shape[1]
    context = sd["pos.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    heads = args.heads or max(1, dim // 64)
    if dim % heads != 0:
        print(f"⚠️  dim {dim} not divisible by heads {heads}; pass --heads", file=sys.stderr)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tensors = {k: v.to(dtype).contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "model.safetensors"))

    config = {
        "architectures": ["ByteGPT"],
        "model_type": "bytegpt",
        "hidden_size": dim,
        "num_hidden_layers": n_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,          # no GQA
        "head_dim": dim // heads,
        "intermediate_size": dim * 4,          # MLP is 4x (Linear->GELU->Linear)
        "max_position_embeddings": context,
        "vocab_size": 256,
        "hidden_act": "gelu",
        "norm_type": "layernorm",              # NOT rmsnorm
        "position_embedding_type": "learned",  # NOT rope
        "tie_word_embeddings": False,
        "torch_dtype": args.dtype,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    n_params = sum(v.numel() for v in tensors.values())
    print(f"✅ exported {len(tensors)} tensors, ~{n_params/1e6:.1f}M params")
    print(f"   dim={dim} layers={n_layers} heads={heads} ctx={context} vocab=256 ({args.dtype})")
    print(f"   -> {args.out}/model.safetensors + config.json")
    print("   ByteBrain is byte-level: serve with the engine in BYTE tokenizer mode (no BPE).")


if __name__ == "__main__":
    main()
