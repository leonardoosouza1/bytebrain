#!/usr/bin/env python3
"""seedlib — motor compartilhado da campanha cross-model (sementes/floresta em tronco congelado).
Trunk = modelo congelado que ARMAZENA sementes (soft-prompt K vetores) e mede neurônios.
Otimizador ESTÁVEL validado (lr0.1 cosine + grad-clip), forward em CHUNKS, alvo 1-token, quant per-grupo."""
import gc, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"; CH = 48

# 50 perguntas de conhecimento REAL (resposta curta) p/ os professores responderem
QUESTIONS = [
    "Qual é a capital da Austrália?", "Qual é o maior planeta do sistema solar?", "Quem pintou a Mona Lisa?",
    "Qual é o metal mais leve?", "Em que ano o homem pisou na Lua?", "Qual é o maior oceano da Terra?",
    "Qual é o rio mais longo do mundo?", "Qual é o osso mais longo do corpo humano?", "Qual gás as plantas absorvem?",
    "Qual é a moeda do Japão?", "Quantos lados tem um hexágono?", "Qual é o símbolo químico do ouro?",
    "Qual é a montanha mais alta do mundo?", "Quem escreveu Dom Casmurro?", "Qual é o menor país do mundo?",
    "Qual planeta é conhecido como planeta vermelho?", "Qual é a capital do Canadá?", "Quantos continentes existem?",
    "Qual é o animal terrestre mais rápido?", "Qual é o maior mamífero do mundo?", "Quem descobriu a gravidade?",
    "Qual é o elemento mais abundante no universo?", "Qual é a capital da Coreia do Sul?", "Qual é o maior deserto quente?",
    "Quantos ossos tem o corpo humano adulto?", "Qual é a língua mais falada do mundo?", "Quem foi o primeiro presidente dos EUA?",
    "Qual é o ponto de ebulição da água em Celsius?", "Qual é a capital da Argentina?", "Qual é o maior órgão do corpo humano?",
    "Quantos planetas há no sistema solar?", "Qual é a fórmula química do sal de cozinha?", "Qual é a capital do Egito?",
    "Quem escreveu Romeu e Julieta?", "Qual é o metal líquido à temperatura ambiente?", "Qual é a maior floresta tropical?",
    "Qual é a capital da Alemanha?", "Quantas cordas tem um violino?", "Qual é o planeta mais próximo do Sol?",
    "Qual é o maior felino do mundo?", "Qual é a capital da França?", "Qual é o quadrado de doze?",
    "Quem desenvolveu a teoria da relatividade?", "Qual é o maior país do mundo em área?", "Qual é o símbolo químico do ferro?",
    "Qual é a capital da Itália?", "Quantas cores tem o arco-íris?", "Qual é o planeta com mais luas?",
    "Qual é a raiz quadrada de 81?", "Qual é a capital de Portugal?",
]

def teacher_answers(path, questions, log=print, maxq=None):
    """carrega professor (nativo ou .gguf), gera resposta curta, LIBERA a VRAM. Retorna lista de strings."""
    import os
    qs = questions[:maxq] if maxq else questions
    try:
        if path.endswith(".gguf"):
            d, fn = os.path.dirname(path), os.path.basename(path)
            tok = AutoTokenizer.from_pretrained(d, gguf_file=fn)
            m = AutoModelForCausalLM.from_pretrained(d, gguf_file=fn, dtype=torch.float16).to(DEV).eval()
        else:
            tok = AutoTokenizer.from_pretrained(path)
            m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
    except Exception as e:
        log(f"    teacher {path} FALHOU load: {str(e)[:80]}"); return None
    ans = []
    for q in qs:
        msg = [{"role": "user", "content": f"Responda com no máximo 3 palavras, sem explicação. {q}"}]
        enc = None
        for kw in ({"enable_thinking": False}, {}):  # Qwen3: desliga o modo-thinking (senão só cospe <think>)
            try:
                enc = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kw)
                enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]; break
            except Exception:
                enc = None
        if enc is None:
            ids = tok(f"Pergunta: {q}\nResposta:", return_tensors="pt").input_ids.to(DEV); enc = {"input_ids": ids}; nin = ids.shape[1]
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, nin:], skip_special_tokens=True)
        if "</think>" in txt: txt = txt.split("</think>")[-1]  # descarta o raciocínio, fica com a resposta
        txt = txt.strip().split("\n")[0].strip().strip(".").strip()[:40]
        ans.append(txt)
    del m; gc.collect(); torch.cuda.empty_cache()
    return ans

def quant(seed, bits, group=128):
    qm = 2 ** (bits - 1) - 1; x = seed.float()
    if group and x.shape[1] % group == 0:
        K, Hd = x.shape; xg = x.reshape(K, Hd // group, group)
        s = (xg.abs().amax(-1, keepdim=True) / qm).clamp_min(1e-8)
        return (torch.round(xg / s).clamp(-qm, qm) * s).reshape(K, Hd).to(torch.float16)
    s = (x.abs().max() / qm).clamp_min(1e-8)
    return (torch.round(x / s).clamp(-qm, qm) * s).to(torch.float16)

def jaccard(a, b): return float((a & b).sum()) / float((a | b).sum() + 1e-9)


class Trunk:
    """modelo congelado = solo onde as sementes crescem."""
    def __init__(self, path, probe_every=5):
        import os
        self.path = path
        if path.endswith(".gguf"):  # tronco a partir de gguf (ex.: Qwen3-4B como solo grande)
            d, fn = os.path.dirname(path), os.path.basename(path)
            self.tok = AutoTokenizer.from_pretrained(d, gguf_file=fn)
            self.model = AutoModelForCausalLM.from_pretrained(d, gguf_file=fn, dtype=torch.float16).to(DEV).eval()
        else:
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.H = self.model.config.hidden_size; self.NL = self.model.config.num_hidden_layers
        self.EL = self.model.get_input_embeddings()
        self.PROBE_L = list(range(2, self.NL, probe_every)); self._a = {}
        for li in self.PROBE_L:
            self._mlp(li).register_forward_hook(self._hk(f"L{li}"))

    def _mlp(self, li): return self.model.model.layers[li].mlp.down_proj
    def _hk(self, nm):
        def h(mod, inp, out): self._a[nm] = inp[0].detach().abs().mean(dim=(0, 1)).float()
        return h

    def free(self):
        del self.model; gc.collect(); torch.cuda.empty_cache()

    def prep(self, pairs):  # alvo COMPLETO (multi-token: fatos reais como "Buenos Aires", "Leonardo da Vinci")
        out = []
        for p, t in pairs:
            ti = self.tok(t, add_special_tokens=False).input_ids
            if ti: out.append((self.tok(p).input_ids, ti))
        return out

    def _batch(self, P, K):
        N = len(P); maxlen = K + max(len(pi) + len(ti) for pi, ti in P)
        E = torch.zeros(N, maxlen, self.H, device=DEV, dtype=torch.float16)
        am = torch.zeros(N, maxlen, device=DEV, dtype=torch.long)
        fpos, ftgt = [], []
        for j, (pi, ti) in enumerate(P):
            pe = self.EL(torch.tensor([pi], device=DEV)).detach()[0]; te = self.EL(torch.tensor([ti], device=DEV)).detach()[0]
            L = K + len(pi) + len(ti); E[j, K:K+len(pi)] = pe; E[j, K+len(pi):L] = te; am[j, :L] = 1
            st = K + len(pi)  # token-alvo k previsto pelos logits na posição st+k-1
            fpos.append([st + k - 1 for k in range(len(ti))]); ftgt.append(list(ti))
        return E, am, fpos, ftgt

    def _fwd(self, seed, Ec, amc):
        X = Ec.clone(); X[:, :seed.shape[0]] = seed.to(torch.float16)
        return self.model(inputs_embeds=X, attention_mask=amc).logits

    def plant(self, pairs, K=1, steps=800, lr=0.1, clip=1.0, init=None, tseed=0):
        P = self.prep(pairs); E, am, fpos, ftgt = self._batch(P, K); N = len(P)
        ntok = max(1, sum(len(t) for t in ftgt))
        g = torch.Generator(device=DEV).manual_seed(tseed)
        if init is not None:
            seed = nn.Parameter(init.float().clone() + torch.randn(K, self.H, generator=g, device=DEV) * 0.05)
        else:
            seed = nn.Parameter(torch.randn(K, self.H, generator=g, device=DEV, dtype=torch.float32) * 0.1)
        opt = torch.optim.AdamW([seed], lr=lr)
        for s in range(steps):
            for gr in opt.param_groups: gr["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
            opt.zero_grad()
            for i in range(0, N, CH):
                lg = self._fwd(seed, E[i:i+CH], am[i:i+CH])
                r, pp, tt = [], [], []
                for jj in range(i, min(i+CH, N)):
                    for a, b in zip(fpos[jj], ftgt[jj]): r.append(jj-i); pp.append(a); tt.append(b)
                if not r: continue
                R = torch.tensor(r, device=DEV); Pp = torch.tensor(pp, device=DEV); T = torch.tensor(tt, device=DEV)
                (F.cross_entropy(lg[R, Pp].float(), T, reduction="sum") / ntok).backward()
            if clip: torch.nn.utils.clip_grad_norm_([seed], clip)
            opt.step()
        return seed.detach().to(torch.float16)

    @torch.no_grad()
    def covered(self, seed, pairs):  # fato coberto = TODOS os tokens do alvo batem
        P = self.prep(pairs); E, am, fpos, ftgt = self._batch(P, seed.shape[0]); ok = set()
        for i in range(0, len(P), CH):
            lg = self._fwd(seed, E[i:i+CH], am[i:i+CH])
            for jj in range(i, min(i+CH, len(P))):
                pp = torch.tensor(fpos[jj], device=DEV); tt = torch.tensor(ftgt[jj], device=DEV)
                if len(pp) and bool((lg[jj-i, pp].argmax(-1) == tt).all()): ok.add(jj)
        return ok

    def recall(self, seed, pairs): return len(self.covered(seed, pairs))

    @torch.no_grad()
    def knows(self, pairs):
        """baseline: o tronco já responde greedy (sem semente)?"""
        cnt = 0
        for p, t in pairs:
            pid = self.tok(p).input_ids; tid = self.tok(t, add_special_tokens=False).input_ids[:1]
            e = self.EL(torch.tensor([pid], device=DEV)).detach()
            nx = int(self.model(inputs_embeds=e).logits[0, -1].argmax())
            cnt += (nx == tid[0])
        return cnt

    @torch.no_grad()
    def mask(self, seed, pairs, topfrac=0.10):
        self._a.clear(); P = self.prep(pairs[:24]); E, am, fpos, ftgt = self._batch(P, seed.shape[0])
        self._fwd(seed, E, am); bits = []
        for li in self.PROBE_L:
            v = self._a[f"L{li}"]; k = max(1, int(len(v) * topfrac))
            m = torch.zeros_like(v, dtype=torch.bool); m[torch.topk(v, k).indices] = True; bits.append(m)
        return torch.cat(bits)
