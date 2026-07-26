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


def test_dt_zero_deixa_o_gate_neutro():
    """dt=0 => log1p(0)=0 => sigmoid(0)=0.5 EXATO: sem tempo decorrido, o gate não decide nada."""
    torch.manual_seed(0); c = CfCCell(16, 32)
    _, g = c(torch.randn(4,32), torch.randn(4,16), torch.zeros(4,1), return_gate=True)
    assert torch.allclose(g, torch.full_like(g, 0.5), atol=1e-6)

def test_gate_satura_conforme_o_tempo_passa():
    """A semântica do tempo é ESTRUTURAL: quanto maior o dt, mais o gate se compromete (sai de 0.5).
    (No CfC oficial ff1 e ff2 são ambos aprendidos — 'esquecer com dt grande' só emerge do TREINO,
    não do init. O que é estrutural, e o que testamos aqui, é a saturação do gate.)"""
    torch.manual_seed(0); c = CfCCell(16, 32)
    h, x = torch.randn(8,32), torch.randn(8,16)
    d = []
    for dt in [0.01, 1.0, 100.0, 86400.0]:
        _, g = c(h, x, torch.full((8,1), dt), return_gate=True)
        d.append((g - 0.5).abs().mean().item())
    assert all(d[i] < d[i+1] for i in range(len(d)-1)), f"gate não satura monotonicamente: {d}"


def test_gradiente_flui():
    c = CfCCell(16, 32)
    h = torch.zeros(2, 32)
    x = torch.randn(2, 16, requires_grad=True)
    c(h, x, torch.full((2, 1), 1.0)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
