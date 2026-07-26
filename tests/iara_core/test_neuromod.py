from iara_core.neuromod import Neuromod

def test_surpresa_sobe_dopamina():
    n = Neuromod()
    for _ in range(50): n.update(1.0, 1.0)          # regime estável
    base = n.dop
    n.update(5.0, 1.0)                               # choque
    assert n.dop > base

def test_fadiga_acumula_e_dorme():
    n = Neuromod()
    for _ in range(500): n.update(1.0, 1.0)
    assert n.fad > 0.5 and n.precisa_dormir()

def test_plasticidade_segue_formula_validada():
    n = Neuromod(); n.dop, n.fad = 1.0, 0.0
    assert abs(n.plasticity() - (0.06 + 0.55)) < 1e-6

def test_dt_longo_relaxa_hormonio():
    n = Neuromod(); n.update(5.0, 1.0); alto = n.dop
    n.update(1.0, 86400.0)                           # um dia depois
    assert n.dop < alto
