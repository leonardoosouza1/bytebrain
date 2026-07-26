import torch, pytest
from iara_core.data import load_graph, Graph
from iara_core.core import IaraCore

STATE = "research/iara_cerebro_state.json"


@pytest.fixture(scope="module")
def sub():
    """Subgrafo pequeno: encodar 38k frases num teste é lento demais (gap G12)."""
    return Graph(load_graph(STATE).facts[:1500])


def test_forward_e_estado_persiste(sub):
    m = IaraCore(sub, d=32, h=48)
    o1 = m.step(["capital do brasil","o que é coracao"], torch.full((2,1),1.0), None)
    o2 = m.step(["e a moeda?","quantas camaras"], torch.full((2,1),60.0), o1.state)
    assert o1.state.shape == (2,48) and not torch.allclose(o1.state, o2.state)


def test_ablacao_muda_contagem_de_params(sub):
    cheio = sum(p.numel() for p in IaraCore(sub, d=32, h=48).parameters())
    sem   = sum(p.numel() for p in IaraCore(sub, d=32, h=48, use_cfc=False).parameters())
    assert cheio > sem


def test_nucleo_e_LEVE():
    """Fidelidade à tese: o treinável é pequeno."""
    g = Graph(load_graph(STATE).facts[:1500])
    n = sum(p.numel() for p in IaraCore(g, d=128, h=192).parameters())
    assert n < 20_000_000, f"núcleo com {n/1e6:.1f}M — a tese pede leve"
