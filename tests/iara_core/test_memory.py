import torch, pytest
from iara_core.data import load_graph, Graph
from iara_core.byte_encoder import ByteEncoder
from iara_core.memory import WaterMemory

STATE = "research/iara_cerebro_state.json"

@pytest.fixture(scope="module")
def sub():
    """Subgrafo pequeno: encodar 38k frases num teste é lento demais (gap G12)."""
    return Graph(load_graph(STATE).facts[:1500])

def test_recupera_o_fato_certo(sub):
    m = WaterMemory(sub, ByteEncoder(d=64), d=64).eval()
    with torch.no_grad():
        idx, _ = m.retrieve(m.encoder([sub.facts[0].stmt]), k=5)
    assert 0 in idx[0].tolist()

def test_gradiente_flui_pela_agua(sub):
    m = WaterMemory(sub, ByteEncoder(d=32), d=32)
    m.soft_scores(m.encoder(["capital do brasil"])).sum().backward()
    assert m.encoder.emb.weight.grad is not None

def test_divide_por_grau_reduz_hub(sub):
    """A correção validada: sem ÷grau o hub domina."""
    m = WaterMemory(sub, ByteEncoder(d=32), d=32).eval()
    with torch.no_grad():
        q = m.encoder(["brasil"])
        hub_com = m.soft_scores(q, by_degree=True).argmax().item()
        hub_sem = m.soft_scores(q, by_degree=False).argmax().item()
    assert m.degree[hub_com] <= m.degree[hub_sem]
