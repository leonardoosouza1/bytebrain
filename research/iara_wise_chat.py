#!/usr/bin/env python3
"""IARA Wise Chat — o artefato final da bateria M151-M157.

Tese: um modelo LEVE (Qwen2.5-1.5B-Instruct, ~3GB, ~40 tok/s na RX6750XT) responde no chat com a
"sabedoria" de um 7B acoplando as DUAS alavancas validadas empiricamente:

  1. RACIOCÍNIO   → self-consistency (maioria de N amostras).  M153: 1.5B single 9 → maioria@5 10/18,
                   empatando/superando o 7B-int8 (8/18). O modelo leve já RACIOCINA como um 7B.
  2. CONHECIMENTO → BIBLIOTECA de seeds POR-FATO atrás de um ROTEADOR (não um seed global!).
                   M154: 1.5B 11/20 < 7B-int8 15/20 (gap de conhecimento). M155 mostrou que um seed
                   ÚNICO prependido em tudo injeta os fatos MAS corrompe o que o modelo já sabia
                   (M156: keep 0/9). O design CERTO (M129/M137/M157): um seed por fato, e o roteador
                   só o dispara quando a query bate com ele (gate de similaridade) — injeta sem
                   corromper. Cada seed ~4KB (K=4 int4).

Uso:
  # treinar cartuchos de conhecimento (TSV "pergunta<TAB>resposta", 1 seed por linha) e salvar:
  python3 iara_wise_chat.py --train-lib fatos.tsv --save-lib biblioteca.pt
  # conversar (raciocínio auto por self-consistency + roteia cartuchos de conhecimento):
  python3 iara_wise_chat.py --lib biblioteca.pt
  python3 iara_wise_chat.py                       # só o 1.5B + self-consistency
"""
import argparse, math, sys, time, unicodedata
import torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

def _norm(s):  # minúsculas sem acento — verificação do fato robusta a "Yangtzé"/"Yangtze"
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = f"{MODELS}/Qwen2.5-1.5B-Instruct"
# roteador (M160-162): gate por MARGEM (melhor − 2º melhor), NÃO por similaridade absoluta.
# Similaridade absoluta não separa parafraseado (~0.6-0.77) de não-relacionado (até 0.68); a MARGEM
# sim (parafraseado 0.25-0.39 vs não-relacionado <0.13, salvo raro outlier). embedding-médio > hidden
# (hidden é anisotrópico, cos~0.98 pra tudo). Config: margem≥0.20 → 6/6 recall, ~1/8 falso disparo.
MARGIN = 0.20   # dispara o cartucho só se a query bate MUITO mais com ele que com o 2º (query in-library)
FLOOR = 0.45    # e acima de um piso mínimo de similaridade absoluta (rejeita lixo)
FMT = "Pergunta: {q}\nResposta:"  # formato do cartucho: treino E inferência precisam CASAR (M158)

REASON_HINT = ("quant", "cálcul", "calcul", "quanto", "some", "soma", "multipl", "divid", "dias",
               "sequência", "sequencia", "maior", "menor", "quadrado", "idade", "dobro", "primo",
               "custa", "pesa", "quantos", "quantas", "logic", "lógic", "por cento", "%")
def is_reasoning(q):
    ql = q.lower()
    return sum(h in ql for h in REASON_HINT) >= 1 and len(ql) < 240

class WiseChat:
    def __init__(self, lib_path=None):
        self.tok = AutoTokenizer.from_pretrained(BASE)
        self.m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to(DEV).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.H = self.m.config.hidden_size; self.EL = self.m.get_input_embeddings()
        self.lib = []  # lista de dicts {emb, seed, label}
        if lib_path:
            data = torch.load(lib_path, map_location=DEV)
            self.lib = [{"emb": d["emb"].to(DEV), "seed": d["seed"].to(torch.float16), "label": d["label"], "ans": d.get("ans", "")} for d in data]
            kb = sum(d["seed"].numel() for d in self.lib) * 2 / 1024
            print(f"[biblioteca] {len(self.lib)} cartuchos ({kb:.1f}KB fp16)", file=sys.stderr)

    @torch.no_grad()
    def q_embed(self, text):
        ids = self.tok(text).input_ids
        return self.EL(torch.tensor([ids], device=DEV)).detach()[0].mean(0)

    def _route_topk(self, query, k=2):
        """top-k candidatos por similaridade de embedding (barato — só recuperação, sem gerar)."""
        if not self.lib: return []
        qe = self.q_embed(query)
        sims = sorted(((float(F.cosine_similarity(qe, d["emb"], 0)), i) for i, d in enumerate(self.lib)), reverse=True)
        return [i for _, i in sims[:k]]

    def route_margin(self, query):
        """gate por MARGEM (M160-162): dispara se a query bate MUITO mais com o cartucho que com o 2º.
        Mantido como alternativa; o padrão (answer) usa o gate por VERIFICAÇÃO (M166), mais preciso."""
        if not self.lib: return None, None
        qe = self.q_embed(query)
        sims = sorted(((float(F.cosine_similarity(qe, d["emb"], 0)), i) for i, d in enumerate(self.lib)), reverse=True)
        best, j = sims[0]; second = sims[1][0] if len(sims) > 1 else 0.0
        if best >= FLOOR and (best - second) >= MARGIN:
            return self.lib[j]["seed"], self.lib[j]["label"]
        return None, None

    @torch.no_grad()
    def _emb(self, ids, seed):
        e = self.EL(torch.tensor([ids], device=DEV)).detach()[0]
        if seed is not None: e = torch.cat([seed.to(torch.float16), e], 0)
        return e[None]

    @torch.no_grad()
    def _greedy(self, ids, n, seed=None):
        cur = self._emb(ids, seed); out = []
        for _ in range(n):
            nx = int(self.m(inputs_embeds=cur).logits[0, -1].argmax()); out.append(nx)
            if nx == self.tok.eos_token_id: break
            cur = torch.cat([cur, self.EL(torch.tensor([[nx]], device=DEV))], 1)
        return self.tok.decode(out, skip_special_tokens=True)

    @torch.no_grad()
    def _sample(self, ids, n, temp, seed=None):
        cur = self._emb(ids, seed); out = []
        for _ in range(n):
            lg = self.m(inputs_embeds=cur).logits[0, -1] / temp
            p = F.softmax(lg, -1); nx = int(torch.multinomial(p, 1)); out.append(nx)
            if nx == self.tok.eos_token_id: break
            cur = torch.cat([cur, self.EL(torch.tensor([[nx]], device=DEV))], 1)
        return self.tok.decode(out, skip_special_tokens=True)

    def _chat_ids(self, user, cot=False):
        content = user + (" Pense passo a passo e termine com 'Resposta: <valor>'." if cot else "")
        enc = self.tok.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, return_tensors="pt", return_dict=True)
        return enc["input_ids"][0].tolist()

    def answer(self, user, k=5):
        # 1) CONHECIMENTO por gate de MARGEM (M161/M162: 6/6 recall, ~1/8 falso — a opção SIMPLES mais robusta).
        #    Gates por verificação/teacher-forced (M166-168) são FRÁGEIS: seeds têm força inconsistente (alguns
        #    disparam o fato mesmo fora do tópico) → não separáveis pela saída. Fix de raiz = treino contrastivo
        #    do seed (ser específico); enquanto isso, margem+biblioteca escopada por domínio é o prático.
        seed, label = self.route_margin(user)
        if seed is not None:  # gera no MESMO formato cru do treino do cartucho
            raw_ids = self.tok(FMT.format(q=user)).input_ids
            out = self._greedy(raw_ids, 24, seed=seed).split("\n")[0].strip()
            words = out.split()  # colapsa repetição imediata do fato (cosmético)
            out = " ".join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])
            return out, f"[conhecimento · cartucho '{label}' disparado]"
        # 2) RACIOCÍNIO por self-consistency (M153)
        if is_reasoning(user):
            ids = self._chat_ids(user, cot=True)
            cands = [self._greedy(ids, 160)] + [self._sample(ids, 160, 0.7) for _ in range(k - 1)]
            finals = [self._final_token(c) for c in cands]
            cnt = Counter([f for f in finals if f]); vote = cnt.most_common(1)
            best = cands[0]
            if vote:
                for c, f in zip(cands, finals):
                    if f == vote[0][0]: best = c; break
            return best.strip(), f"[raciocínio · maioria@{k}: {dict(cnt)}]"
        # 3) CHAT base
        return self._greedy(self._chat_ids(user, cot=False), 200).strip(), "[chat · 1.5B base]"

    @staticmethod
    def _final_token(txt):
        import re
        m = re.findall(r"[-+]?\d[\d.,]*", txt.split("Resposta")[-1]) or re.findall(r"[-+]?\d[\d.,]*", txt)
        return m[-1].rstrip(".,") if m else None

    def train_library(self, pairs, K=4, steps=300, lr=0.1):
        """1 seed POR fato (M136: generaliza melhor 1-fato/seed). Guarda (emb da pergunta, seed, label)."""
        for i, (q, a) in enumerate(pairs):
            pi = self.tok(FMT.format(q=q)).input_ids; ti = self.tok(a, add_special_tokens=False).input_ids
            pe = self.EL(torch.tensor([pi], device=DEV)).detach()[0]; te = self.EL(torch.tensor([ti], device=DEV)).detach()[0]
            L = K + len(pi) + len(ti); E = torch.zeros(1, L, self.H, device=DEV, dtype=torch.float16)
            E[0, K:K+len(pi)] = pe; E[0, K+len(pi):L] = te
            Pp = torch.tensor([K+len(pi)+kk-1 for kk in range(len(ti))], device=DEV); T = torch.tensor(ti, device=DEV)
            seed = nn.Parameter(torch.randn(K, self.H, device=DEV, dtype=torch.float32) * 0.1)
            opt = torch.optim.AdamW([seed], lr=lr)
            for s in range(steps):
                for g in opt.param_groups: g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
                X = E.clone(); X[:, :K] = seed.to(torch.float16)
                loss = F.cross_entropy(self.m(inputs_embeds=X).logits[0, Pp].float(), T)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed], 1.0); opt.step()
            self.lib.append({"emb": self.q_embed(q).detach(), "seed": seed.detach().to(torch.float16), "label": q[:40], "ans": a.strip()})
            print(f"  cartucho {i+1}/{len(pairs)} '{q[:34]}' loss {loss.item():.3f}", file=sys.stderr)
        return self.lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", help="biblioteca .pt de cartuchos para carregar")
    ap.add_argument("--train-lib", help="TSV pergunta<TAB>resposta (1 cartucho por linha)")
    ap.add_argument("--save-lib", help="onde salvar a biblioteca treinada")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ask", help="pergunta única (não-interativo)")
    a = ap.parse_args()

    chat = WiseChat(lib_path=a.lib)
    if a.train_lib:
        pairs = []
        for line in open(a.train_lib):
            line = line.rstrip("\n")
            if "\t" in line:
                q, ans = line.split("\t", 1); pairs.append((q, " " + ans.strip()))
        print(f"treinando {len(pairs)} cartuchos (1 seed/fato)...", file=sys.stderr)
        chat.train_library(pairs)
        if a.save_lib:
            torch.save([{"emb": d["emb"].cpu(), "seed": d["seed"].cpu(), "label": d["label"], "ans": d.get("ans", "")} for d in chat.lib], a.save_lib)
            kb = sum(d["seed"].numel() for d in chat.lib) * 2 / 1024
            print(f"biblioteca salva em {a.save_lib} ({len(chat.lib)} cartuchos, {kb:.1f}KB)", file=sys.stderr)
        return
    if a.ask:
        ans, tag = chat.answer(a.ask, k=a.k); print(tag); print(ans); return
    print("IARA Wise Chat (1.5B + self-consistency + biblioteca roteada). Ctrl-C p/ sair.\n", file=sys.stderr)
    while True:
        try: user = input("você> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if not user: continue
        t = time.time(); ans, tag = chat.answer(user, k=a.k)
        print(f"iara> {ans}\n      {tag} ({time.time()-t:.1f}s)\n")

if __name__ == "__main__":
    main()
