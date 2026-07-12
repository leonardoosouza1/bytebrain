#!/usr/bin/env python3
"""Roteador do cérebro: compara rotear por EMBEDDING-MÉDIO (atual, cru) vs ESTADO-OCULTO do MemByte (que
agora tem priors ricos). Mede separação: pergunta-alvo/parafraseada (POS) deve casar alto com seu fato;
não-relacionada (NEG) deve casar baixo. Melhor separação = mata o false-route (França→Krylon). GPU."""
import sys, time, os, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from src.model import ByteGPT
from wisdom_bridge import enc
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FMT = "P: {q}\nR:"

CK = "/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt"
ck = torch.load(CK, map_location=DEV, weights_only=False); c = ck["config"]
m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV)
m.load_state_dict(ck["model"]); m.eval()
for p in m.parameters(): p.requires_grad_(False)
log(f"MemByte-real ({m.n_params/1e6:.1f}M, val {ck.get('best_val',0):.3f} bpb)")

@torch.no_grad()
def emb_mean(q):
    ids = torch.tensor([enc(FMT.format(q=q))], device=DEV)
    return m.tok(ids).detach()[0].mean(0)
@torch.no_grad()
def hidden_mean(q):  # roda os blocos + lnf, média das posições
    ids = torch.tensor([enc(FMT.format(q=q))], device=DEV)
    pos = torch.arange(ids.size(1), device=DEV); h = m.tok(ids) + m.pos(pos)[None]
    for b in m.blocks: h = b(h)
    return m.lnf(h)[0].mean(0)

# perguntas-ALVO armazenadas
STORED = ["Qual o codigo do cofre da IARA?", "Qual o planeta natal do Zephyr?",
          "Quem e o guardiao da torre de Vaelis?", "Qual a senha do laboratorio 7?",
          "Qual a cor do dragao de Miro?", "Qual o mineral do templo?"]
# parafraseadas (POS held-out — mesma pergunta, forma diferente)
PARA = ["Qual e o codigo do cofre da IARA?", "De qual planeta o Zephyr e natural?",
        "Quem guarda a torre de Vaelis?", "Qual a senha do laboratorio numero 7?",
        "De que cor e o dragao do Miro?", "Que mineral fica no templo?"]
# NÃO-relacionadas (NEG — não podem casar)
NEG = ["Qual a capital da Franca?", "Quanto e 12 mais 15?", "O que e fotossintese?",
       "Quem foi Napoleao?", "Qual a capital do Japao?", "Explique a gravidade.",
       "Qual a formula da agua?", "Quem pintou a Mona Lisa?"]

def evaluate(rep, name):
    bank = torch.stack([rep(q) for q in STORED])
    def top(q):
        qe = rep(q); s = F.cosine_similarity(qe[None], bank, 1); o = torch.argsort(s, descending=True)
        return int(o[0]), float(s[o[0]]), float(s[o[0]]-s[o[1]])
    # POS: parafraseadas devem rotear pro índice certo
    pos_ok = 0; pos_best = []; pos_mar = []
    for i, q in enumerate(PARA):
        j, best, mar = top(q); pos_ok += (j == i); pos_best.append(best); pos_mar.append(mar)
    # NEG: máx similaridade (quanto menor, melhor separação)
    neg_best = [top(q)[1] for q in NEG]; neg_mar = [top(q)[2] for q in NEG]
    sep_abs = min(pos_best) - max(neg_best); sep_mar = min(pos_mar) - max(neg_mar)
    log(f"[{name}] rota-parafraseada {pos_ok}/{len(PARA)} | POS sim[min {min(pos_best):.2f}] NEG sim[max {max(neg_best):.2f}] "
        f"sep_abs {sep_abs:+.2f} | POS mar[min {min(pos_mar):.2f}] NEG mar[max {max(neg_mar):.2f}] sep_mar {sep_mar:+.2f}")
    return sep_abs, sep_mar

log("== comparando representações de roteamento ==")
evaluate(emb_mean, "embedding-médio")
evaluate(hidden_mean, "estado-oculto ")
log(f"DONE ({time.time()-t0:.0f}s)")
