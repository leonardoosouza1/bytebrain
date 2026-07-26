import torch
import torch.nn as nn
from dataclasses import dataclass
from .byte_encoder import ByteEncoder
from .cfc import CfCCell
from .memory import WaterMemory
from .organs import Router, Verifier
from .neuromod import Neuromod


@dataclass
class Out:
    state: torch.Tensor
    route_logits: torch.Tensor
    retrieved_idx: torch.Tensor
    retrieved_scores: torch.Tensor
    verify_logit: torch.Tensor
    feat: torch.Tensor


class IaraCore(nn.Module):
    """O organismo: byte → CfC(h, dt) → [roteador | água | verificador].
    Cada flag use_* liga/desliga um órgão para a BATERIA DE ABLAÇÃO."""

    def __init__(self, graph, d=128, h=192, use_cfc=True, use_neuromod=True,
                 use_water=True, use_verifier=True, k=8):
        super().__init__()
        self.use_cfc = use_cfc
        self.use_neuromod = use_neuromod
        self.use_water = use_water
        self.use_verifier = use_verifier
        self.k = k
        self.h_size = h

        self.encoder = ByteEncoder(d=d)
        self.memory = WaterMemory(graph, self.encoder, d=d) if use_water else None

        nm = 4 if use_neuromod else 0
        self.cfc = CfCCell(d + nm, h) if use_cfc else None

        d_feat = (h if use_cfc else d) + d
        self.router = Router(d_feat)
        self.verifier = Verifier(d_feat + 1) if use_verifier else None
        self.neuro = Neuromod()

    def init_state(self, B, device):
        return torch.zeros(B, self.h_size, device=device)

    def step(self, queries, dt, state=None):
        q = self.encoder(queries)
        B, dev = q.shape[0], q.device

        if self.use_cfc:
            if state is None:
                state = self.init_state(B, dev)
            x = q
            if self.use_neuromod:
                x = torch.cat([q, self.neuro.vector().to(dev).expand(B, 4)], dim=-1)
            state = self.cfc(state, x, dt.to(dev))
            ctx = state
        else:
            state = torch.zeros(B, self.h_size, device=dev)
            ctx = q

        feat = torch.cat([ctx, q], dim=-1)
        route = self.router(feat)

        if self.use_water:
            idx, sc = self.memory.retrieve(q, k=self.k)
        else:
            idx = torch.zeros(B, self.k, dtype=torch.long, device=dev)
            sc = torch.zeros(B, self.k, device=dev)

        if self.use_verifier:
            ver = self.verifier(torch.cat([feat, sc.max(-1, keepdim=True).values], dim=-1))
        else:
            ver = torch.zeros(B, 1, device=dev)

        return Out(state, route, idx, sc, ver, feat)
