import torch, os, tempfile, time
from iara_core.data import load_graph, Graph, split_by_entity, make_sessions
from iara_core.core import IaraCore
from iara_core.train import train, SUPERVISED_ROUTES

STATE = "research/iara_cerebro_state.json"


def _make_model_and_sessions(n_facts=1500):
    """Subgrafo pequeno para evitar encodar 38k frases nos testes (gap G12)."""
    g = Graph(load_graph(STATE).facts[:n_facts])
    tr, _ = split_by_entity(g, 0.15, seed=0)
    ss = make_sessions(tr, seed=0, n_sessions=40)
    m = IaraCore(tr, d=32, h=48)
    return m, ss, tr


def test_loss_cai_e_salva_checkpoint():
    """Treino 40 passos: loss final < loss inicial e checkpoint é salvo.

    loss_inicial is now the mean of the first 5 steps (not a single noisy
    sample), making the comparison mean-vs-mean and less fragile.
    """
    m, ss, _ = _make_model_and_sessions()
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "c.pt")
        lp = os.path.join(td, "l.json")
        r = train(m, ss, steps=40, lr=1e-3, ckpt_path=ck, log_path=lp, seed=0)
        elapsed = time.time() - t0
        print(f"\n  wall-clock: {elapsed:.1f}s  loss_inicial={r['loss_inicial']:.4f}  loss_final={r['loss_final']:.4f}")
        assert os.path.exists(ck), "checkpoint não foi salvo"
        assert r["loss_final"] < r["loss_inicial"], (
            f"loss não caiu: {r['loss_inicial']:.4f} -> {r['loss_final']:.4f}"
        )


def test_resume_continua_do_checkpoint():
    """Resume: treina N passos, interrompe, retoma — step count continua de onde parou.

    Also verifies that optimizer Adam state (momentum/variance) is non-empty
    after resume, confirming finding #1 is genuinely fixed.
    """
    m, ss, _ = _make_model_and_sessions()
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "c.pt")
        lp = os.path.join(td, "l.json")

        # Primeira corrida: 20 passos
        r1 = train(m, ss, steps=20, lr=1e-3, ckpt_path=ck, log_path=lp, seed=0)
        assert os.path.exists(ck)
        ck_data = torch.load(ck, weights_only=False)
        assert ck_data["step"] == 20, f"esperado step=20, got {ck_data['step']}"
        hist_len_after_first = len(ck_data["hist"])
        assert hist_len_after_first == 20, f"hist deveria ter 20 entradas, tem {hist_len_after_first}"

        # Optimizer state must be persisted in the checkpoint.
        assert "opt" in ck_data, "checkpoint deve conter chave 'opt' com o state do otimizador"
        assert len(ck_data["opt"]["state"]) > 0, (
            "opt['state'] está vazio — Adam nunca atualizou ou não foi salvo"
        )
        n_params_with_state = len(ck_data["opt"]["state"])

        # Segunda corrida: resume do ck e vai até 40 passos totais
        # Cria um novo modelo (simula restart de sessão Colab)
        m2, _, _ = _make_model_and_sessions()
        r2 = train(m2, ss, steps=40, lr=1e-3, ckpt_path=ck, log_path=lp, seed=0)
        ck_data2 = torch.load(ck, weights_only=False)
        assert ck_data2["step"] == 40, f"esperado step=40 após resume, got {ck_data2['step']}"
        hist_len_after_second = len(ck_data2["hist"])
        # Após resume de passo 20 até 40, hist total deve ter 40 entradas
        assert hist_len_after_second == 40, (
            f"hist deveria ter 40 entradas após resume, tem {hist_len_after_second}"
        )

        # After resume the optimizer state must still be non-empty and have the
        # same number of tracked parameter tensors.
        assert "opt" in ck_data2, "checkpoint pós-resume deve conter 'opt'"
        n_params_after_resume = len(ck_data2["opt"]["state"])
        assert n_params_after_resume == n_params_with_state, (
            f"optimizer state mudou de tamanho no resume: {n_params_with_state} -> {n_params_after_resume}"
        )
        assert n_params_after_resume > 0, "optimizer state vazio após resume"

        print(
            f"\n  resume ok: {hist_len_after_first} + 20 = {hist_len_after_second} entradas no hist"
            f"\n  optimizer state: {n_params_after_resume} param groups com estado Adam (momentum/variance)"
        )


def test_supervised_routes_constantes():
    """SUPERVISED_ROUTES deve expor exatamente {0, 1} para Task-9."""
    assert SUPERVISED_ROUTES == frozenset({0, 1}), (
        f"Expected SUPERVISED_ROUTES == {{0, 1}}, got {SUPERVISED_ROUTES}"
    )
