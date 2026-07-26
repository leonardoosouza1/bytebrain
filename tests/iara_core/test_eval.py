"""
Task 9: Bateria de Ablação — testes para iara_core/eval.py.

TDD: este arquivo é escrito ANTES da implementação de eval.py.

Caveats documentados aqui (não omitidos):
  1. route_acc: apenas rotas 0 (reflexo) e 1 (memoria) recebem sinal de treino.
     O campo é calculado apenas sobre essas duas classes. Se for calculado como
     4-class accuracy sobre logits não treinados, o resultado é enganoso.
  2. continuidade sem CfC: quando use_cfc=False, step() retorna zeros como state
     e a chamada duplicada (com e sem state) produz saídas idênticas — portanto
     continuidade DEVE ser exatamente 0.0 para esse config.
  3. continuidade sem water: sem water não há retrieved_scores com semântica de
     similaridade; o campo deve retornar float(nan) ou 0.0 como sentinel, e o
     caller deve saber que a métrica não é significativa nesse caso.
"""

import math
import torch
import pytest

from iara_core.data import load_graph, split_by_entity, make_sessions
from iara_core.core import IaraCore
from iara_core.eval import evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    return load_graph("research/iara_cerebro_state.json")


@pytest.fixture(scope="module")
def train_val(graph):
    return split_by_entity(graph, 0.15, seed=0)


@pytest.fixture(scope="module")
def val_sessions(train_val):
    _, va = train_val
    return make_sessions(va, seed=1, n_sessions=20)


@pytest.fixture(scope="module")
def full_model(train_val):
    tr, _ = train_val
    m = IaraCore(tr, d=32, h=48)
    return m.eval()


# ---------------------------------------------------------------------------
# Step 1 (brief verbatim): as chaves existem e são float
# ---------------------------------------------------------------------------

def test_metricas_saem(full_model, val_sessions):
    r = evaluate(full_model, val_sessions)
    for k in ("recall@1", "recall@8", "mrr", "route_acc", "continuidade"):
        assert k in r, f"chave '{k}' ausente no resultado"
        assert isinstance(r[k], float), f"r['{k}'] não é float: {type(r[k])}"


# ---------------------------------------------------------------------------
# Honestidade: route_acc deve ser calculada APENAS nas rotas supervisionadas
# (0=reflexo, 1=memoria). O campo supervised_only deve existir ou a doc string
# deve deixar claro. Verificamos aqui com um assert fraco: o valor é [0,1].
# ---------------------------------------------------------------------------

def test_route_acc_range(full_model, val_sessions):
    r = evaluate(full_model, val_sessions)
    assert 0.0 <= r["route_acc"] <= 1.0, f"route_acc fora de [0,1]: {r['route_acc']}"


# ---------------------------------------------------------------------------
# Ablação honesta: sem CfC, o estado retornado por step() são zeros
# independente do estado passado. Logo, a chamada "com estado" e "sem estado"
# são idênticas → continuidade deve ser exatamente 0.0.
#
# Raciocínio: core.py linha 62-63:
#   else:
#       state = torch.zeros(B, self.h_size, device=dev)
#       ctx = q
# O ctx (e portanto feat) não depende do state passado — é sempre q.
# Portanto out.retrieved_scores não varia entre as duas chamadas → Δ = 0.
# ---------------------------------------------------------------------------

def test_continuidade_sem_cfc_e_zero(train_val, val_sessions):
    tr, _ = train_val
    m = IaraCore(tr, d=32, h=48, use_cfc=False).eval()
    r = evaluate(m, val_sessions)
    assert r["continuidade"] == 0.0, (
        f"sem CfC a continuidade deve ser 0.0 (as duas chamadas são idênticas), "
        f"mas foi {r['continuidade']}"
    )


# ---------------------------------------------------------------------------
# Ablação sem water: continuidade não é nan nem levanta exceção; é um float.
# O valor é 0.0 porque sem water não há retrieved_scores com semântica real
# (zeros são retornados), e o evaluate deve reconhecer isso e retornar 0.0.
# ---------------------------------------------------------------------------

def test_continuidade_sem_water_e_float(train_val, val_sessions):
    tr, _ = train_val
    m = IaraCore(tr, d=32, h=48, use_water=False).eval()
    r = evaluate(m, val_sessions)
    cont = r["continuidade"]
    assert isinstance(cont, float), f"continuidade deve ser float, foi {type(cont)}"
    assert not math.isnan(cont), "continuidade não deve ser nan (use 0.0 como sentinel)"


# ---------------------------------------------------------------------------
# Smoke: recall@8 >= recall@1 (invariante básico de ranking)
# ---------------------------------------------------------------------------

def test_recall_ordering(full_model, val_sessions):
    r = evaluate(full_model, val_sessions)
    assert r["recall@8"] >= r["recall@1"], (
        f"recall@8 ({r['recall@8']:.4f}) < recall@1 ({r['recall@1']:.4f}) — bug no ranking"
    )


# ---------------------------------------------------------------------------
# Smoke: mrr está em [0, 1]
# ---------------------------------------------------------------------------

def test_mrr_range(full_model, val_sessions):
    r = evaluate(full_model, val_sessions)
    assert 0.0 <= r["mrr"] <= 1.0, f"mrr fora de [0,1]: {r['mrr']}"
