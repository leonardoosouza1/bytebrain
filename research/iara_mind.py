#!/usr/bin/env python3
"""IARA MIND — o CÉREBRO INTEIRO abstraído em código: órgãos complementares que processam, armazenam,
inferem, conhecem e GERAM DADOS. Materializa a arquitetura (ver diagrama): tudo trafega em BYTES.

Órgãos:
  Bytes    (sentido)     : interface universal — texto <-> bytes.
  Memory   (hipocampo)   : ARMAZENAMENTO. byte-model congelado + sementes; store()/recall() limpo, roteável.
  Router   (tálamo)      : decide o caminho (ágil vs sabedoria) por estado-oculto + margem.
  Cortex   (respondedor) : INFERÊNCIA. reescreve o fato / raciocina. não precisa saber, só articular.
  Source   (aprendizado) : busca o desconhecido (professor/…); vira semente (repasse por bytes).
  Mind     (o todo)      : perceive -> route -> (recall->articulate | reason);  learn();  generate_data().

Demo:  python3 iara_mind.py
"""
import sys, time, math, os, re
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from src.model import ByteGPT
from src.sample import generate as _bytegen
from wisdom_bridge import ByteSeed, enc, dec, quant
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FMT = "P: {q}\nR:"; NL = 10

# ── SENTIDO: bytes = interface universal ─────────────────────────────────────────────────────────
class Bytes:
    @staticmethod
    def to(text): return enc(text)
    @staticmethod
    def frm(ids): return dec(ids)

# ── ARMAZENAMENTO: memória byte + sementes (hipocampo) ───────────────────────────────────────────
class Memory:
    def __init__(self):
        ck = "/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt"
        o = torch.load(ck, map_location=DEV, weights_only=False); c = o["config"]
        self.bb = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
        self.bb.load_state_dict(o["model"])
        for p in self.bb.parameters(): p.requires_grad_(False)
        self.bs = ByteSeed(self.bb); self.lib = []  # (rep, seed, fato, pergunta)
    @torch.no_grad()
    def _rep(self, q):  # estado-oculto (roteamento: separa por margem, byte-space é anisotrópico)
        ids = torch.tensor([enc(FMT.format(q=q))], device=DEV)
        pos = torch.arange(ids.size(1), device=DEV); h = self.bb.tok(ids) + self.bb.pos(pos)[None]
        for b in self.bb.blocks: h = b(h)
        return self.bb.lnf(h)[0].mean(0)
    def store(self, q, fact):
        seed, _ = self.bs.plant(FMT.format(q=q), " " + fact + "\n", K=8, steps=400)
        self.lib.append((self._rep(q), quant(seed, 4), fact, q))    # semente int4 = 1.5KB
    @torch.no_grad()
    def recall(self, q, margin=0.02):
        if not self.lib: return None, 0.0
        qe = self._rep(q)
        sims = sorted(((float(F.cosine_similarity(qe, r, 0)), i) for i, (r, *_ ) in enumerate(self.lib)), reverse=True)
        best, j = sims[0]; second = sims[1][0] if len(sims) > 1 else 0.0
        if best - second < margin: return None, best - second
        return self.bs.recall(self.lib[j][1], FMT.format(q=q), n=20, stop_at=NL).strip(), best - second
    @property
    def size_kb(self): return len(self.lib) * self.lib[0][1].numel() * 0.5 / 1024 if self.lib else 0

# ── TÁLAMO: roteia (embutido na Memory.recall via margem) ────────────────────────────────────────
# ── INFERÊNCIA: córtex ágil (respondedor) ────────────────────────────────────────────────────────
class Cortex:
    def __init__(self, path="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-1.5B-Instruct"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
    @torch.no_grad()
    def _gen(self, msgs, n=48, temp=0.0):
        e = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        e = {k: v.to(DEV) for k, v in e.items()}; nin = e["input_ids"].shape[1]
        o = self.m.generate(**e, max_new_tokens=n, do_sample=temp > 0, temperature=temp or None,
                            top_p=0.9 if temp > 0 else None, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(o[0, nin:], skip_special_tokens=True).strip().split("\n")[0]
    def articulate(self, q, fact):  # REESCRITA (não dispara recusa no modelo pequeno)
        return self._gen([{"role": "user", "content": f'Reescreva numa frase curta e natural: A resposta de "{q}" é {fact}.'}])
    def reason(self, q):            # caminho ágil: raciocina/responde do próprio conhecimento
        return self._gen([{"role": "user", "content": q + " Responda curto."}])
    def ask(self, prompt, n=32, temp=0.0):  # uso geral (ex: gerar dados)
        return self._gen([{"role": "user", "content": prompt}], n=n, temp=temp)

# ── APRENDIZADO: fonte -> semente (repasse por bytes) ────────────────────────────────────────────
class Source:
    """professor residente (poderia ser web verificada). Devolve resposta curta p/ virar semente."""
    def __init__(self, cortex): self.cx = cortex
    def lookup(self, q):
        return self.cx._gen([{"role": "system", "content": "Responda SÓ com a resposta final, no máximo 3 palavras."},
                             {"role": "user", "content": q}], n=16).strip().strip('"').rstrip(".")

# ── GERAÇÃO: byte-model que ESCREVE português (não instruído — CONTINUA texto) ───────────────────
class Generator:
    def __init__(self, ck="/home/leonardo/projects/LLM/bytebrain/research/ckpt_gen/ckpt_best.pt"):
        o = torch.load(ck, map_location=DEV, weights_only=False); c = o["config"]
        self.m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
        self.m.load_state_dict(o["model"]); self.step = o.get("step", 0)
    def write(self, prompt, n=200, temp=0.7):
        return _bytegen(self.m, prompt=prompt, n=n, temperature=temp, top_p=0.9, rep_penalty=1.3, device=DEV)

# ── O CÉREBRO ────────────────────────────────────────────────────────────────────────────────────
class Mind:
    def __init__(self, learn_source=False, with_generator=False):
        log("montando o cérebro: memória byte + córtex" + (" + gerador byte" if with_generator else ""))
        self.mem = Memory(); self.cx = Cortex(); self.src = Source(self.cx) if learn_source else None
        self.gen = Generator() if with_generator else None
        log(f"pronto (memória {len(self.mem.lib)} fatos" + (f", gerador passo {self.gen.step}" if self.gen else "") + ")")
    def write(self, prompt, n=200):       # o cérebro ESCREVE português com o próprio byte-model
        return self.gen.write(prompt, n=n) if self.gen else self.cx.reason(prompt)
    def learn(self, q, fact=None):        # planta conhecimento (dado ou buscado na fonte)
        if fact is None and self.src: fact = self.src.lookup(q)
        self.mem.store(q, fact); return fact
    def perceive(self, query):            # o fluxo completo: route -> recall/articulate | reason
        fact, mar = self.mem.recall(query)
        if fact: return self.cx.articulate(query, fact), f"sabedoria (margem {mar:.2f}, recall '{fact[:16]}')"
        return self.cx.reason(query), "ágil"
    def generate_data(self, n_per_fact=3):
        """o cérebro GERA DADOS: pra cada fato que conhece, o córtex inventa formulações variadas da
        pergunta (aumento de dados) — vira corpus Q&A sintético a partir do que a memória guarda."""
        pairs = []
        for _, _, fact, q in self.mem.lib:
            variants = self.cx.ask(f'Gere {n_per_fact} formas diferentes de perguntar algo cuja resposta e "{fact}". '
                                   f'Uma por linha, so as perguntas.', n=64, temp=0.7)
            for line in variants.split("\n"):
                qq = line.strip(" -0123456789.").strip()
                if len(qq) > 8 and "?" in qq: pairs.append((qq, fact))
        return pairs

def main():
    mind = Mind()
    facts = [("Qual o codigo do cofre da IARA?", "7492"), ("Qual o planeta natal do Zephyr?", "Krylon"),
             ("Quem e o guardiao da torre de Vaelis?", "Orin")]
    log("== APRENDER (plantar conhecimento) ==")
    for q, a in facts: mind.learn(q, a); log(f"  aprendido: {a}")
    log(f"  memória = {mind.mem.size_kb:.1f} KB p/ {len(mind.mem.lib)} fatos")
    log("== PERCEBER (fluxo completo) ==")
    for q in ["Qual o codigo do cofre da IARA?", "De qual planeta o Zephyr e natural?", "Quanto e 8 vezes 7?"]:
        ans, path = mind.perceive(q); log(f"  Q: {q}\n     [{path}] {ans[:60]!r}")
    log("== GERAR DADOS (o cérebro produz corpus a partir do que sabe) ==")
    data = mind.generate_data(n_per_fact=3)
    for q, a in data[:8]: log(f"  gerado: ({q[:44]!r} -> {a})")
    log(f"  total gerado: {len(data)} pares Q&A a partir de {len(mind.mem.lib)} fatos")
    log(f"DONE ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
