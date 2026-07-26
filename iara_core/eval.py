"""
iara_core/eval.py — Bateria de Ablação (Task 9)

evaluate(model, sessions) -> dict
    Métricas sobre o conjunto de validação. Não vaza dados de treino.
    Chaves retornadas:
        recall@1, recall@8, mrr  — ranking do Water Memory (só quando use_water=True)
        route_acc                — acurácia nas rotas SUPERVISIONADAS (0=reflexo, 1=memoria)
                                   NOTE: cortex(2) e abster(3) não recebem sinal de treino
                                   neste dataset; a métrica é calculada apenas nas classes {0,1}.
        continuidade             — Δ(score_com_estado − score_sem_estado) / n
                                   Mede se o estado da sessão ajuda o Water Memory.
                                   - sem CfC: sempre 0.0 (step() ignora state → chamadas idênticas)
                                   - sem water: sempre 0.0 (scores são zeros sintéticos)
        n                        — número de steps avaliados (debug)

ablation_battery(graph, tr, va, steps, seeds, out_path) -> dict
    Treina 5 configs × n_seeds e reporta Δ de cada órgão.
    O diretório out_path é criado se não existir (default local "out/").
"""

import json
import os
import torch

from .train import SUPERVISED_ROUTES  # frozenset({0, 1})


@torch.no_grad()
def evaluate(model, sessions, k=8):
    """Avalia o modelo sobre as sessões de validação.

    Parâmetros
    ----------
    model    : IaraCore já em modo .eval()
    sessions : lista de Session (de data.make_sessions)
    k        : top-k para retrieval

    Retorno
    -------
    dict com float para cada métrica documentada no módulo.
    """
    model.eval()
    dev = next(model.parameters()).device

    # Índice fact.key → posição na lista, apenas quando water está ativo.
    key2idx: dict = {}
    if model.use_water:
        key2idx = {f.key: i for i, f in enumerate(model.memory.graph.facts)}

    r1 = r8 = mrr = racc = 0.0
    n_retrieval = 0   # steps com gold_fact_key válido e water ativo
    n_route = 0       # steps com rota supervisionada
    com = sem = 0.0   # acúmulo para continuidade (water + cfc)
    n_continuidade = 0

    for s in sessions:
        state = None
        for st in s.steps:
            dt = torch.tensor([[st.dt_seconds]], device=dev)

            # --- Forward COM estado de sessão ---
            out = model.step([st.query], dt, state)

            # --- Retrieval metrics (apenas quando water ativo e gold existe) ---
            g = key2idx.get(st.gold_fact_key, -1) if model.use_water else -1
            if g >= 0:
                ids = out.retrieved_idx[0].tolist()
                r1 += float(ids[0] == g)
                r8 += float(g in ids)
                mrr += 1.0 / (ids.index(g) + 1) if g in ids else 0.0
                n_retrieval += 1

                # --- Continuidade: mede se o estado da sessão ajuda.
                # Só tem sentido quando CfC está ativo (state influencia feat).
                # Quando use_cfc=False, step() ignora state e retorna zeros →
                # out0 == out → Δ == 0 exato.
                # Quando use_water=False, scores são zeros sintéticos → Δ == 0.
                if model.use_cfc:
                    out0 = model.step([st.query], dt, None)
                    com += float(out.retrieved_scores[0].max())
                    sem += float(out0.retrieved_scores[0].max())
                    n_continuidade += 1

            # --- Route accuracy (apenas sobre rotas supervisionadas {0, 1}) ---
            # kind=="repeticao" → label 0 (reflexo); qualquer outro → label 1 (memoria)
            gold_route = 0 if st.kind == "repeticao" else 1
            # Restringimos o argmax às duas classes supervisionadas (índices 0 e 1)
            # para não enganar com logits de cortex/abster não treinados.
            pred_route = int(out.route_logits[0, :2].argmax().item())
            racc += float(pred_route == gold_route)
            n_route += 1

            # Avança o estado da sessão
            state = out.state

    n_route = max(n_route, 1)
    n_retrieval_safe = max(n_retrieval, 1)

    # Continuidade: se não houve nenhum step com CfC+water, retorna 0.0 (sentinel)
    continuidade = (com - sem) / max(n_continuidade, 1) if n_continuidade > 0 else 0.0

    return {
        "recall@1": r1 / n_retrieval_safe,
        "recall@8": r8 / n_retrieval_safe,
        "mrr": mrr / n_retrieval_safe,
        "route_acc": racc / n_route,
        "continuidade": float(continuidade),
        "n": n_route,
    }


def ablation_battery(graph, tr, va, steps=3000, seeds=(0, 1, 2),
                     out_path="out/ablacao.json"):
    """Treina 5 configs × n_seeds e compara cada órgão com o organismo completo.

    Parâmetros
    ----------
    graph    : Graph completo (não usado diretamente, mas passado por clareza)
    tr       : Graph de treino
    va       : Graph de validação
    steps    : número de passos de treino por config/seed
    seeds    : seeds para multi-seed (mínimo 3 para intervalos)
    out_path : caminho do JSON de saída; o diretório é criado se não existir.
               Default "out/ablacao.json" (local, não Colab-específico).
    """
    from .core import IaraCore
    from .train import train
    from .data import make_sessions

    # Garante que o diretório de saída existe independente do ambiente.
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    configs = {
        "completo":        dict(),
        "sem_cfc":         dict(use_cfc=False),
        "sem_neuromod":    dict(use_neuromod=False),
        "sem_agua":        dict(use_water=False),
        "sem_verificador": dict(use_verifier=False),
    }

    res = {}
    for name, cfg in configs.items():
        runs = []
        for sd in seeds:
            ckpt = os.path.join(out_dir, f"{name}_s{sd}.pt")
            log  = os.path.join(out_dir, f"{name}_s{sd}.json")
            m = IaraCore(tr, **cfg)
            train(m, make_sessions(tr, seed=sd), steps=steps, seed=sd,
                  ckpt_path=ckpt, log_path=log)
            runs.append(evaluate(m, make_sessions(va, seed=100 + sd)))
        # Média multi-seed para cada métrica
        res[name] = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        json.dump(res, open(out_path, "w"), indent=1)
        print(name, {k: round(v, 4) for k, v in res[name].items()})

    base = res["completo"]
    print("\n=== Δ DE CADA ÓRGÃO (completo − sem_X; positivo = órgão paga) ===")
    for name in configs:
        if name == "completo":
            continue
        print(
            f"  {name:18s} mrr {base['mrr'] - res[name]['mrr']:+.4f} · "
            f"continuidade {base['continuidade'] - res[name]['continuidade']:+.4f}"
        )
    return res
