#!/usr/bin/env python3
"""IARA BRAIN — cérebro de ÓRGÃOS COMPLEMENTARES (ideia do Leonardo: o 8M não é fraco, é ARMAZÉM; separar
armazenar de responder, como o cérebro humano hipocampo↔córtex / System 1↔2).

  MEMÓRIA (hipocampo, sabedoria):  byte-model 8M congelado + sementes de byte. Guarda barato o que ninguém
                                   sabe (fatos privados), cresce sozinho, minúsculo. NÃO articula.
  CÓRTEX  (respondedor, ágil):     modelo pequeno fluente. Não precisa SABER — só ARTICULAR o que a memória
                                   entrega, e raciocinar rápido no que já domina.

Prova honesta (fatos PRIVADOS que nenhum modelo pode saber → qualquer acerto vem DA memória):
  - córtex sozinho  -> erra/aluciná (não tem o fato)
  - memória sozinha -> tem o fato mas gagueja (não conversa)
  - CÉREBRO         -> memória recupera + córtex articula = certo E fluente
GPU. Roda:  python3 iara_brain.py
"""
import sys, time, math, os, json, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from wisdom_bridge import load_byte_model, ByteSeed, enc, dec, quant
from src.model import ByteGPT
DEV = "cuda"; t0 = time.time()

def load_membyte_real():
    """MemByte-real (11M, treinado em Wikipedia PT, priors amplos → recall limpo p/ qualquer palavra)."""
    ck = "/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt"
    if not os.path.exists(ck): return None
    o = torch.load(ck, map_location=DEV, weights_only=False); c = o["config"]
    m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV)
    m.load_state_dict(o["model"]); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
FMT = "P: {q}\nR:"

# ── ÓRGÃO 1: MEMÓRIA (byte-model + sementes) ────────────────────────────────────────────────────
class Memory:
    def __init__(self):
        self.bb = load_membyte_real() or load_byte_model(trained=True)  # vaso especializado; fallback 8M
        self.bs = ByteSeed(self.bb)
        self.lib = []  # (q_emb, seed, fact)
    @torch.no_grad()
    def _qrep(self, q):  # ESTADO-OCULTO do armazém (priors ricos separam perguntas melhor que embedding-médio; mem_router)
        ids = torch.tensor([enc(FMT.format(q=q))], device=DEV)
        pos = torch.arange(ids.size(1), device=DEV); h = self.bb.tok(ids) + self.bb.pos(pos)[None]
        for b in self.bb.blocks: h = b(h)
        return self.bb.lnf(h)[0].mean(0)
    def store(self, q, fact):
        seed, _ = self.bs.plant(FMT.format(q=q), " " + fact + "\n", K=8, steps=400)  # terminador \n = recall limpo
        self.lib.append((self._qrep(q), quant(seed, 4), fact))  # semente int4 = 1.5KB
    @torch.no_grad()
    def recall(self, q, margin=0.02):
        """roteia por ESTADO-OCULTO + gate por MARGEM (melhor-2º): parafraseado casa MUITO mais que o resto
        (mem_router: POS margem≥0.03 vs NEG≤0.01). Recupera até a fronteira \\n = LIMPO. None se margem baixa."""
        if not self.lib: return None
        qe = self._qrep(q)
        sims = sorted(((float(F.cosine_similarity(qe, e, 0)), i) for i, (e, _, _) in enumerate(self.lib)), reverse=True)
        best, j = sims[0]; second = sims[1][0] if len(sims) > 1 else 0.0
        if best - second < margin: return None
        return self.bs.recall(self.lib[j][1], FMT.format(q=q), n=20, stop_at=10).strip()  # recall LIMPO

# ── ÓRGÃO 2: CÓRTEX (respondedor ágil) — PROFESSOR TROCADO: Qwen2.5-7B via HTTP (isola llama_cpp do torch) ──
class Cortex:
    def __init__(self, url="http://127.0.0.1:8080/v1/chat/completions"): self.url = url
    def up(self):
        try: urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3); return True
        except Exception: return False
    def _chat(self, content, n=64, temp=0.3):
        body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": temp}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180)); return r["choices"][0]["message"]["content"].strip().split("\n")[0]
        except Exception as e: return f"[córtex offline — suba cortex_server.py: {e}]"
    def articulate(self, q, fact):  # REESCRITA (não "você sabe?"): reformular não dispara recusa
        return self._chat(f'Reescreva numa frase curta e natural, sem inventar: A resposta de "{q}" é {fact}.', n=40)
    def direct(self, q):            # caminho ágil: responde do próprio conhecimento
        return self._chat(q + " Responda curto.", n=90)

# ── O CÉREBRO: roteia entre ágil (córtex) e sabedoria (memória→córtex) ────────────────────────────
class Brain:
    def __init__(self): self.mem = Memory(); self.cortex = Cortex()
    def learn(self, q, fact): self.mem.store(q, fact)
    def answer(self, q):
        fact = self.mem.recall(q)
        if fact:  # a memória candidata um fato (limpo); o córtex JULGA relevância e articula
            return self.cortex.articulate(q, fact), f"memória→córtex [recall limpo: {fact[:18]!r}]"
        return self.cortex.direct(q), "ÁGIL (córtex sozinho)"

def main():
    brain = Brain(); log(f"cérebro montado: memória byte-11M + córtex Qwen2.5-7B (HTTP) {'ONLINE' if brain.cortex.up() else 'OFFLINE'}")
    # fatos PRIVADOS inventados — nenhum modelo pode saber; só a memória, se guardarmos
    private = [
        ("Qual o codigo do cofre da IARA?", "7492"),
        ("Qual o planeta natal do Zephyr?", "Krylon"),
        ("Quem e o guardiao da torre de Vaelis?", "Orin"),
        ("Qual a senha do laboratorio 7?", "girassol"),
    ]
    log("== ANTES de aprender: córtex sozinho nos fatos privados (não pode saber) ==")
    for q, a in private:
        log(f"  Q: {q}\n     córtex: {brain.cortex.direct(q)[:56]!r}")
    log("== memória APRENDE os fatos privados (sementes de 1.5KB no armazém byte) ==")
    for q, a in private: brain.learn(q, a); log(f"  guardado: '{a}' p/ '{q[:32]}'")
    log("== DEPOIS: cada órgão sozinho vs o CÉREBRO ==")
    for q, a in private:
        raw = brain.mem.recall(q)
        ans, tag = brain.answer(q)
        ok = a.lower() in ans.lower()
        log(f"  Q: {q}")
        log(f"     memória (recall limpo): {str(raw)[:26]!r}")
        log(f"     CÉREBRO: {ans[:56]!r}  [{tag}]  {'✓' if ok else '✗'}")
    # caminho ágil: pergunta comum/raciocínio -> córtex direto, memória não dispara
    log("== caminho ÁGIL: pergunta comum (memória não tem, córtex responde rápido) ==")
    for q in ["Quanto e 12 mais 15?", "Qual a capital da Franca?"]:
        ans, tag = brain.answer(q); log(f"  Q: {q}  -> {ans[:40]!r}  [{tag}]")
    log(f"DONE ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
