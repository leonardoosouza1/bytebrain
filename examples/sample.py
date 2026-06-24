"""Generate text from a trained ByteBrain checkpoint.

    python -m examples.sample --checkpoint overnight_ck/loop_best.pt --prompt "O "

Pass --dim/--layers/--heads to match the checkpoint you are loading (defaults are the ~26M config).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ByteGPT, load_checkpoint  # noqa: E402
from src.sample import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Sample from a ByteBrain checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="O ")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--n", type=int, default=300, help="bytes to generate")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--rep-penalty", type=float, default=1.4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    model = ByteGPT(dim=a.dim, n_layers=a.layers, n_heads=a.heads).to(a.device)
    load_checkpoint(model, a.checkpoint, map_location=a.device)
    print(f"loaded {model.n_params / 1e6:.0f}M params from {a.checkpoint}\n")
    print(generate(
        model, prompt=a.prompt, n=a.n, temperature=a.temperature,
        top_p=a.top_p, rep_penalty=a.rep_penalty, device=a.device,
    ))


if __name__ == "__main__":
    main()
