import torch
import torch.nn as nn


class CfCCell(nn.Module):
    """Closed-form Continuous-time. O dt REAL entra no gate — é assim que a IARA sente o tempo.
    Portado de iara-brain/backend/src/systems/brain/cfc_cell.py (Jax/Flax)."""

    def __init__(self, input_size, hidden_size, backbone=None):
        super().__init__()
        b = backbone or hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(input_size + hidden_size, b),
            nn.Tanh()
        )
        self.ff1 = nn.Linear(b, hidden_size)
        self.ff2 = nn.Linear(b, hidden_size)
        self.gate = nn.Linear(b, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, h, x, dt, return_gate=False):
        """Forward pass with time-aware gating.

        Args:
            h: Hidden state of shape [batch, hidden_size]
            x: Input of shape [batch, input_size]
            dt: Elapsed time of shape [batch, 1]
            return_gate: If True, return (h_new, gate); else return h_new.

        Returns:
            Updated hidden state of shape [batch, hidden_size], or (h_new, gate) if return_gate=True.
        """
        z = self.backbone(torch.cat([h, x], dim=-1))
        # log1p(dt) comprime a escala (segundos..dias) e evita saturar o sigmoid.
        # dt=0 => gate=0.5 exato (neutro); dt grande => o gate se compromete.
        g = torch.sigmoid(self.gate(z) * torch.log1p(dt.clamp_min(0)))
        h_new = g * self.ff1(z) + (1 - g) * self.ff2(z)
        return (h_new, g) if return_gate else h_new
