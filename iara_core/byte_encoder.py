import torch
import torch.nn as nn


class ByteEncoder(nn.Module):
    def __init__(self, d=128, nlayers=2, nheads=4, maxlen=192):
        super().__init__()
        self.maxlen = maxlen
        self.emb = nn.Embedding(256, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(d, nheads, d*4, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.d = d

    def forward(self, texts):
        dev = self.emb.weight.device
        seqs = [list(t.encode("utf-8")[:self.maxlen]) for t in texts]
        L = max(1, max(len(s) for s in seqs))
        x = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
        m = torch.zeros(len(seqs), L, dtype=torch.bool, device=dev)
        for i, s in enumerate(seqs):
            if s:
                x[i, :len(s)] = torch.tensor(s, device=dev)
                m[i, :len(s)] = True
        h = self.emb(x) + self.pos(torch.arange(L, device=dev))[None]
        h = self.enc(h, src_key_padding_mask=~m)
        w = m.float().unsqueeze(-1)
        return (h * w).sum(1) / w.sum(1).clamp_min(1e-6)
