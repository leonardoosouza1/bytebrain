"""ByteGPT — a decoder-only transformer over raw bytes.

The vocabulary is the 256 possible byte values: there is no tokenizer and no BPE merge table.
Any UTF-8 string is fed in as its byte sequence, and the model predicts the next byte. This gives
a ~600x smaller embedding table than a typical BPE model and makes out-of-vocabulary inputs
impossible by construction — every byte is in-vocabulary, in every language and script.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

CONTEXT = 256  # bytes of context (size of the positional table)


class Block(nn.Module):
    """Pre-norm transformer block: causal multi-head self-attention + GELU MLP."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.15):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        attn_drop = 0.1 if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_drop)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class ByteGPT(nn.Module):
    """Byte-level GPT. The default config (dim=512, layers=8) is ~26M parameters."""

    def __init__(self, dim: int = 512, n_layers: int = 8, n_heads: int = 8, context: int = CONTEXT,
                 use_checkpoint: bool = False):
        super().__init__()
        self.context = context
        self.use_checkpoint = use_checkpoint
        self.tok = nn.Embedding(256, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        for b in self.blocks:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
            else:
                h = b(h)
        return self.head(self.lnf(h))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _rename_legacy(key: str) -> str:
    """Map the short attribute names used by overnight_loop.py to the clean module names."""
    if key.startswith("t."):
        return "tok." + key[2:]
    if key.startswith("p."):
        return "pos." + key[2:]
    if key.startswith("f."):
        return "lnf." + key[2:]
    if key.startswith("o."):
        return "head." + key[2:]
    if key.startswith("b."):
        rest = key[2:].replace(".l1.", ".ln1.").replace(".l2.", ".ln2.").replace(".pr.", ".proj.")
        return "blocks." + rest
    return key


def load_checkpoint(model: ByteGPT, path: str, map_location: str = "cpu") -> ByteGPT:
    """Load weights from any checkpoint format this repo produces:
    train.py's full checkpoint ({"model": ..., "opt": ...}), the legacy overnight_loop state dict
    (short attribute names), or a plain state dict."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if "t.weight" in sd:  # legacy overnight_loop checkpoint -> remap to clean names
        sd = {_rename_legacy(k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model
