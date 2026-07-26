import torch, pytest
from iara_core.data import load_graph, Graph
from iara_core.byte_encoder import ByteEncoder
from iara_core.memory import WaterMemory

STATE = "research/iara_cerebro_state.json"

@pytest.fixture(scope="module")
def sub():
    """Subgrafo pequeno: encodar 38k frases num teste é lento demais (gap G12)."""
    return Graph(load_graph(STATE).facts[:1500])

def test_retrieve_devolve_k_validos(sub):
    """Contrato de forma: k índices no range, scores finitos e em ordem decrescente."""
    m = WaterMemory(sub, ByteEncoder(d=64), d=64).eval()
    with torch.no_grad():
        idx, sc = m.retrieve(m.encoder(["capital do brasil"]), k=5)
    assert idx.shape == (1,5) and sc.shape == (1,5)
    assert idx.min() >= 0 and idx.max() < len(sub.facts)
    assert torch.isfinite(sc).all() and (sc[0][:-1] >= sc[0][1:]).all()

def test_encoder_poe_o_proprio_fato_acima_da_media(sub):
    """Estrutural (sobreposição de bytes): a frase EXATA pontua acima da média.
    NÃO testamos 'recupera em top-k' — isso é propriedade TREINADA. Com encoder no init as
    frases em PT-BR têm ~0.93 de cosseno entre si (medido), então o argmax é ruído.
    A qualidade real de recuperação é medida na Tarefa 9 (recall@1 / MRR, pós-treino)."""
    m = WaterMemory(sub, ByteEncoder(d=64), d=64).eval()
    with torch.no_grad():
        sim = (m.encoder([sub.facts[0].stmt]) @ m.fact_embeddings().t())[0]
    assert sim[0] > sim.mean(), f"própria {sim[0]:.3f} vs média {sim.mean():.3f}"

def test_divide_por_grau_favorece_fato_pouco_conectado(sub):
    """A gema validada (multi-hop 0%→100%): ÷grau tira a vantagem do hub, então os fatos
    POUCO conectados sobem de posição. Medido: rank 857 → 598 para um fato de grau 1."""
    m = WaterMemory(sub, ByteEncoder(d=64), d=64).eval()
    baixos = (m.degree <= m.degree.median()).nonzero().flatten()[:20]
    with torch.no_grad():
        q = m.encoder(["capital do brasil"])
        s_com, s_sem = m.soft_scores(q, by_degree=True)[0], m.soft_scores(q, by_degree=False)[0]
    rank = lambda s, i: int((s > s[i]).sum())
    r_com = sum(rank(s_com, int(i)) for i in baixos) / len(baixos)
    r_sem = sum(rank(s_sem, int(i)) for i in baixos) / len(baixos)
    assert r_com < r_sem, f"÷grau não ajudou os pouco-conectados: {r_com:.0f} vs {r_sem:.0f}"

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
