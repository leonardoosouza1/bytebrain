import torch.nn as nn

ROUTES = ["reflexo","memoria","cortex","abster"]


class Router(nn.Module):
    """Roteador de 4 rotas: reflexo, memoria, cortex, abster."""
    def __init__(self, d_in, n_routes=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_routes))

    def forward(self, feat):
        return self.net(feat)


class Verifier(nn.Module):
    """Verificador: logit de 'eu sei' (treinado com negativos de held-out)."""
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, feat):
        return self.net(feat)
