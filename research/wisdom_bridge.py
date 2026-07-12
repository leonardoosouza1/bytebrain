#!/usr/bin/env python3
"""WISDOM BRIDGE — repasse de sabedoria de um LLM-professor para o byte-model, via BYTES como comunicação
universal (a tese IARA). O professor (Qwen, residente na VRAM) responde em texto; texto→UTF-8 é a língua
NATIVA do byte-model — sem projeção de embedding, sem "forçar" token. A sabedoria vira uma SEMENTE (soft-
prompt em byte-space) plantada no ByteBrain CONGELADO; "abrir" = o byte-model reconstrói o valor a partir
de poucos bits + seus priors (lei do andaime: modelo treinado precisa só do essencial).

Testes honestos:
  --smoke                 carrega o ByteBrain 8M e gera (confirma o vaso)
  --transfer N            planta N fatos como sementes de byte, mede recall + bits
  --scaffold N            planta os mesmos N no ByteBrain TREINADO vs RANDOM-init (a lei do andaime em byte-space)
"""
import sys, time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BB8M = "/home/leonardo/projects/LLM/bytebrain/export_bytebrain_8m/model.safetensors"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def load_byte_model(trained=True):
    m = ByteGPT(dim=384, n_layers=6, n_heads=6).to(DEV)
    if trained:
        sd = load_file(BB8M)  # ckpt sem dropout no MLP (mlp.2=2ª linear) -> arch atual tem dropout em .2, linear em .3
        sd = {k.replace(".mlp.2.", ".mlp.3."): v for k, v in sd.items()}
        m.load_state_dict(sd)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def enc(s): return list(s.encode("utf-8"))          # texto -> ids de byte 0-255
def dec(ids): return bytes(ids).decode("utf-8", "ignore")

# ---- SEMENTE em byte-space: soft-prompt de K vetores (dim=384) no modelo congelado ------------------
class ByteSeed:
    def __init__(self, model): self.m = model; self.dim = model.tok.weight.shape[1]
    def _emb(self, ids):  # ids de byte -> embeddings (sem pos; forward adiciona pos)
        return self.m.tok(torch.tensor([ids], device=DEV)).detach()[0]
    def plant(self, prompt, target, K=8, steps=400, lr=0.1):
        pe = self._emb(enc(prompt)); te = self._emb(enc(target))
        tgt_ids = enc(target)
        L = K + pe.shape[0] + te.shape[0]
        E = torch.zeros(1, L, self.dim, device=DEV)
        E[0, K:K+pe.shape[0]] = pe; E[0, K+pe.shape[0]:L] = te
        # posições que predizem cada byte do alvo
        pos = [K + pe.shape[0] + i - 1 for i in range(len(tgt_ids))]
        Pp = torch.tensor(pos, device=DEV); T = torch.tensor(tgt_ids, device=DEV)
        seed = nn.Parameter(torch.randn(K, self.dim, device=DEV) * 0.1)
        opt = torch.optim.AdamW([seed], lr=lr)
        for s in range(steps):
            for g in opt.param_groups: g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
            X = E.clone(); X[0, :K] = seed
            logits = self.m(inputs_embeds=X)[0]
            loss = F.cross_entropy(logits[Pp], T)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed], 1.0); opt.step()
        return seed.detach(), float(loss)
    @torch.no_grad()
    def recall(self, seed, prompt, n=24, stop_at=None):
        pe = self._emb(enc(prompt))
        X = torch.cat([seed, pe], 0)[None]
        out = []
        for _ in range(n):
            nx = int(self.m(inputs_embeds=X)[0, -1].argmax())
            if stop_at is not None and nx == stop_at: break  # fronteira de fim-de-resposta -> recall limpo
            out.append(nx)
            X = torch.cat([X, self.m.tok(torch.tensor([[nx]], device=DEV))], 1)
        return dec(out)

def quant(seed, bits, group=64):  # per-grupo (como seedlib)
    K, H = seed.shape; qm = 2 ** (bits - 1) - 1
    xg = seed.reshape(K, H // group, group)
    s = xg.abs().amax(-1, keepdim=True) / qm + 1e-9
    return (torch.round(xg / s).clamp(-qm, qm) * s).reshape(K, H)

FACTS = [
    ("Quem pintou 'A Noite Estrelada'?", "Van Gogh"),
    ("Qual a capital da Mongolia?", "Ulan Bator"),
    ("Quem escreveu 'Dom Casmurro'?", "Machado de Assis"),
    ("Qual o rio mais longo da Asia?", "Yangtze"),
    ("Quem compos 'As Quatro Estacoes'?", "Vivaldi"),
    ("Qual o maior planeta do sistema solar?", "Jupiter"),
    ("Em que ano caiu o Muro de Berlim?", "1989"),
    ("Qual a formula da agua?", "H2O"),
]
FMT = "P: {q}\nR:"

def match(out, ans): return ans.lower()[:6] in out.lower()

def run_transfer(N, model=None, label="treinado"):
    m = model or load_byte_model(trained=True)
    bs = ByteSeed(m); facts = FACTS[:N]
    ok = 0; ok_i4 = 0
    for q, a in facts:
        seed, loss = bs.plant(FMT.format(q=q), " " + a, K=8, steps=400)
        r = bs.recall(seed, FMT.format(q=q)); h = match(r, a)
        ri4 = bs.recall(quant(seed, 4), FMT.format(q=q)); hi4 = match(ri4, a)
        ok += h; ok_i4 += hi4
        log(f"  [{label}] '{a[:16]:16}' loss={loss:.2f} recall={'HIT' if h else 'no '} int4={'HIT' if hi4 else 'no '} | {r[:22]!r}")
    kb = 8 * 384 * 2 / 1024  # fp16 por semente
    log(f"=== [{label}] TRANSFER: fp16 {ok}/{len(facts)} | int4 {ok_i4}/{len(facts)} | {kb:.1f}KB/semente (fp16), {kb/4:.1f}KB int4 ===")
    return ok, ok_i4

# ---- PROFESSOR residente na VRAM: fala texto; bytes são a interlíngua p/ o byte-model ---------------
class Teacher:
    def __init__(self, path="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-1.5B-Instruct", int8=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        if int8:  # professor 7B na VRAM via quanto (fonte mais confiável)
            from transformers import QuantoConfig
            self.m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                     quantization_config=QuantoConfig(weights="int8"), device_map="cuda", low_cpu_mem_usage=True)
        else:
            self.m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
    @torch.no_grad()
    def ask(self, q, n=16):
        # instrução terse: só a resposta (nome/valor), sem repetir a pergunta -> alvo de bytes limpo
        sys = "Você responde SÓ com a resposta final, no máximo 3 palavras, sem repetir a pergunta e sem frase."
        enc = self.tok.apply_chat_template([{"role": "system", "content": sys}, {"role": "user", "content": q}],
                                           add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]
        o = self.m.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id)
        txt = self.tok.decode(o[0, nin:], skip_special_tokens=True).strip()
        return txt.split("\n")[0].strip().strip('"').rstrip(".").strip()

QUESTIONS = [q for q, _ in FACTS] + [
    "Qual a capital do Japao?", "Quem desenvolveu a teoria da relatividade?",
    "Qual o metal liquido a temperatura ambiente?", "Quantos continentes existem?",
]

def run_teach(N):
    """Model→model REAL via bytes: o professor (Qwen) responde, a resposta vira semente de byte no ByteBrain."""
    bb = load_byte_model(trained=True); bs = ByteSeed(bb)
    teach = Teacher(); log("professor Qwen-1.5B residente na VRAM")
    ok = 0; qs = QUESTIONS[:N]
    for q in qs:
        ans = teach.ask(q)[:20]                       # o professor FALA (texto)
        seed, loss = bs.plant(FMT.format(q=q), " " + ans, K=8, steps=400)   # texto->bytes->semente
        r = bs.recall(seed, FMT.format(q=q)); h = ans.lower()[:5] in r.lower()
        ok += h
        log(f"  prof '{q[:30]:30}' -> '{ans[:18]:18}' | byte-model: {'HIT' if h else 'no '} {r[:20]!r}")
    log(f"=== TEACH→BYTE (bytes = comunicação universal): {ok}/{len(qs)} fatos repassados do Qwen ao ByteBrain ===")

def run_agent(N, big=False):
    """AGÊNCIA: base de sabedoria cresce sozinha. Biblioteca de sementes de byte; pergunta nova sem semente
    = DESCONHECIDO -> busca no professor -> vira semente -> a base cresce. (fonte = teacher; poderia ser web)."""
    bb = load_byte_model(trained=True); bs = ByteSeed(bb)
    if big:
        teach = Teacher("/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-7B-Instruct", int8=True)
        log("professor 7B-int8 residente na VRAM (fonte confiável)")
    else:
        teach = Teacher()
    lib = {}  # q -> (seed, ans)
    def known(q):  # roteia: já tem semente pra essa pergunta?
        return q in lib
    log("== AGÊNCIA: o byte-model começa SEM saber nada e cresce buscando no professor ==")
    for q in QUESTIONS[:N]:
        if not known(q):
            log(f"  '{q[:34]:34}' -> DESCONHECIDO, buscando no professor...")
            ans = teach.ask(q)[:20]
            seed, _ = bs.plant(FMT.format(q=q), " " + ans, K=8, steps=350)
            lib[q] = (seed, ans)
            log(f"       aprendido: '{ans[:20]}' (semente 1.5KB int4) | base agora tem {len(lib)} fatos")
        else:
            seed, ans = lib[q]
        r = bs.recall(lib[q][0], FMT.format(q=q))
        log(f"       responde: {r[:26]!r}")
    log(f"=== base de sabedoria cresceu de 0 -> {len(lib)} fatos, sozinha, {len(lib)*1.5:.0f}KB int4 total ===")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--transfer", type=int, default=0)
    ap.add_argument("--scaffold", type=int, default=0)
    ap.add_argument("--teach", type=int, default=0)
    ap.add_argument("--agent", type=int, default=0)
    ap.add_argument("--big", action="store_true", help="usar professor 7B-int8 (fonte confiável)")
    a = ap.parse_args()
    if a.smoke:
        from src.sample import generate
        m = load_byte_model(trained=True)
        log(f"ByteBrain 8M carregado ({m.n_params/1e6:.1f}M params)")
        log(f"geração: {generate(m, prompt='O Brasil ', n=120, device=DEV)!r}")
    if a.transfer:
        run_transfer(a.transfer, label="treinado")
    if a.scaffold:
        log("== LEI DO ANDAIME em byte-space: treinado vs random-init ==")
        mt = load_byte_model(trained=True); run_transfer(a.scaffold, mt, "TREINADO")
        del mt; torch.cuda.empty_cache()
        mr = load_byte_model(trained=False); run_transfer(a.scaffold, mr, "RANDOM  ")
    if a.teach: run_teach(a.teach)
    if a.agent: run_agent(a.agent, big=a.big)
    log(f"DONE ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
