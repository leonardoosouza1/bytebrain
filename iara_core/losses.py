import torch
import torch.nn.functional as F


def retrieval_loss_sampled(model, queries, gold_idx, n_neg=255, gen=None):
    """InfoNCE com NEGATIVOS AMOSTRADOS (gap G12).
    Encodar os 38k fatos a cada passo levaria minutos/passo. Aqui: fato-ouro + n_neg sorteados."""
    dev = gold_idx.device
    N = len(model.memory.stmts)
    B = gold_idx.shape[0]
    neg = torch.randint(0, N, (n_neg,), device=dev, generator=gen)
    cand = torch.cat([gold_idx, neg]).unique()              # ouro sempre presente
    pos_in_cand = (cand[None, :] == gold_idx[:, None]).float().argmax(-1)
    q = model.encoder(queries)                              # [B,d]
    E = model.encoder([model.memory.stmts[i] for i in cand.tolist()])  # [C,d] — só os candidatos
    logits = q @ E.t() / (q.shape[-1] ** 0.5)
    return F.cross_entropy(logits, pos_in_cand)


def route_loss(logits, gold_route):
    return F.cross_entropy(logits, gold_route)


def verify_loss(logit, acertou):
    return F.binary_cross_entropy_with_logits(logit, acertou)


def time_loss(pred_gap, true_gap):
    """A tarefa que SÓ o tempo resolve (gap G3): prever quanto tempo passou (log-escala)."""
    return F.mse_loss(pred_gap, torch.log1p(true_gap))


W_DEFAULT = {"retr": 1.0, "route": 0.5, "verify": 0.5}


def total_loss(out, batch, model, w=None):
    w = w or W_DEFAULT
    l = {}
    if model.use_water:
        l["retr"] = retrieval_loss_sampled(model, batch["queries"], batch["gold_fact_idx"])
    l["route"] = route_loss(out.route_logits, batch["gold_route"])
    if model.use_verifier:
        l["verify"] = verify_loss(out.verify_logit, batch["acertou"])
    total = sum(w[k] * v for k, v in l.items())
    return total, {k: float(v) for k, v in l.items()}
