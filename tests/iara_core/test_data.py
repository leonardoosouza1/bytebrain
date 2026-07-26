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
    # Separate within-session gaps (i > 0) from between-session gaps (i == 0)
    within_session_dts = [sess.steps[i].dt_seconds for sess in ss for i in range(1, len(sess.steps))]
    between_session_dts = [sess.steps[0].dt_seconds for sess in ss]
    # Within-session gaps should be short (< 60s)
    assert min(within_session_dts) < 60, "within-session gaps must be < 60s"
    # Between-session gaps should be long (> 86400s = 1d)
    assert max(between_session_dts) > 86400, "between-session gaps must span > 1d"
