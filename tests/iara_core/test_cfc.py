import torch
from iara_core.cfc import CfCCell


def test_shape_e_estabilidade():
    c = CfCCell(16, 32)
    h = torch.zeros(4, 32)
    x = torch.randn(4, 16)
    h2 = c(h, x, torch.full((4, 1), 1.0))
    assert h2.shape == (4, 32) and torch.isfinite(h2).all()


def test_dt_muda_o_estado():
    """dt é o SINAL DO TEMPO: mesmo input, dt diferente => estado diferente."""
    torch.manual_seed(0)
    c = CfCCell(16, 32)
    h = torch.randn(2, 32)
    x = torch.randn(2, 16)
    curto = c(h, x, torch.full((2, 1), 0.01))
    longo = c(h, x, torch.full((2, 1), 100.0))
    assert (curto - longo).abs().mean() > 1e-3


def test_dt_grande_esquece_mais():
    """dt grande => gate->1 => vai pra ff1 (transição) => menos parecido com h anterior."""
    torch.manual_seed(0)
    c = CfCCell(16, 32)
    h = torch.randn(8, 32)
    x = torch.zeros(8, 16)
    sim_curto = torch.cosine_similarity(c(h, x, torch.full((8, 1), 0.01)), h, dim=-1).mean()
    sim_longo = torch.cosine_similarity(c(h, x, torch.full((8, 1), 100.0)), h, dim=-1).mean()
    assert sim_curto > sim_longo


def test_gradiente_flui():
    c = CfCCell(16, 32)
    h = torch.zeros(2, 32)
    x = torch.randn(2, 16, requires_grad=True)
    c(h, x, torch.full((2, 1), 1.0)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
