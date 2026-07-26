from iara_core.data import load_graph, split_by_entity, make_sessions

def test_carrega_grafo_real():
    g = load_graph("research/iara_cerebro_state.json")
    assert len(g.facts) > 10000
    f = g.facts[0]
    assert isinstance(f.stmt, str) and isinstance(f.ts, float)

def test_split_nao_vaza_entidade():
    g = load_graph("research/iara_cerebro_state.json")
    tr, va = split_by_entity(g, 0.15, seed=0)
    assert set(f.entity for f in tr.facts).isdisjoint(f.entity for f in va.facts)

def test_sessoes_tem_dt_variado():
    g = load_graph("research/iara_cerebro_state.json")
    ss = make_sessions(g, seed=0)
    dts = [s.dt_seconds for sess in ss for s in sess.steps]
    assert min(dts) < 60 and max(dts) > 86400   # segundos ATÉ dias
