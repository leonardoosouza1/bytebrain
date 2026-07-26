import torch
from iara_core.organs import Router, Verifier, ROUTES


def test_router_shape_e_rotas():
    r = Router(64)
    assert r(torch.randn(3,64)).shape == (3,4)
    assert ROUTES == ["reflexo","memoria","cortex","abster"]


def test_verificador_aprende_a_abster():
    """Sinal claro: feature alta => sei; baixa => não sei."""
    torch.manual_seed(0); v = Verifier(8)
    opt = torch.optim.Adam(v.parameters(), lr=0.05)
    X = torch.cat([torch.ones(64,8), -torch.ones(64,8)]); Y = torch.cat([torch.ones(64,1), torch.zeros(64,1)])
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(v(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.sigmoid(v(torch.ones(1,8))).item() > 0.8
    assert torch.sigmoid(v(-torch.ones(1,8))).item() < 0.2
