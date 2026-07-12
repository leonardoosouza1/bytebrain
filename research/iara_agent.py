#!/usr/bin/env python3
"""IARA micro-swarm agent (PoC) — the thesis: intelligence = HARNESS (routing +
memory + composition) + cheap specialized micro-models, with a big generalist only
as fallback. Adaptive Routing = the 'R' in IARA.

  - ROTEADOR: classifies the query → which specialist (v0 heuristic; TODO distilled byte-router).
  - ESPECIALISTAS: byte micro-models (ckpt_broad = facts/code/defs). Fast, tiny, cheap.
  - GENERALISTA: Qwen3-4B (GGUF) — only for open reasoning/conversation the micros can't do.
  - MEMÓRIA: persistent user-facts + conversation context.

Run:  NGL=-1 python research/iara_agent.py --demo   (GPU)
"""
import sys, os, re, json, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant

MEM_FILE = "research/iara_agent_memory.json"

# ---------- ESPECIALISTA byte (micro-model) ----------
class ByteSpecialist:
    def __init__(self, ckpt="ckpt_broad", name="fatos/código"):
        c = torch.load(f"{ckpt}/ckpt_best.pt", map_location="cpu", weights_only=False); self.cf = c["config"]
        set_act_quant(self.cf.get("quant_bits", 0))
        self.m = GraphByteGPT(self.cf["dim"], self.cf["layers"], self.cf["heads"], self.cf["ctx"],
                              topk=self.cf.get("topk", 0), mem=self.cf.get("mem", 0), topic=self.cf.get("topic", 0)).eval()
        self.m.load_state_dict(c["model"]); self.name = name
        self.params = sum(p.numel() for p in self.m.parameters()) / 1e6

    @torch.no_grad()
    def answer(self, prompt, n=48, multiline=False):
        ids = list(prompt.encode())
        for _ in range(n):
            nxt = int(self.m(torch.tensor([ids[-self.cf["ctx"]:]]))[0, -1].argmax())
            ids.append(nxt)
            if not multiline and bytes(ids[-2:]) == b"\n\n": break
        out = bytes(ids[len(prompt.encode()):]).decode("utf-8", "replace")
        return out.rstrip() if multiline else out.split("\n")[0].strip()

# ---------- GENERALISTA (serviço separado: Qwen3 server via HTTP; isola libs ROCm) ----------
class Generalist:
    def __init__(self, url="http://127.0.0.1:8080/v1/chat/completions"): self.url = url
    def answer(self, history, user, sysmsg):
        import urllib.request
        body = json.dumps({"messages": [{"role": "system", "content": sysmsg}] + history +
                           [{"role": "user", "content": user + " /no_think"}],
                           "max_tokens": 400, "temperature": 0.6}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            return r["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[generalista offline — suba o servidor Qwen3: {e}]"

# ---------- ROTEADOR (byte-router APRENDIDO, 0.26M) ----------
from research.byte_router import ByteRouter, encode as _renc, DOMAINS as R_DOMAINS
_router = ByteRouter(); _router.load_state_dict(torch.load("ckpt_router.pt", map_location="cpu", weights_only=False)["model"]); _router.eval()
@torch.no_grad()
def route(q, thresh=0.55):
    x, m = _renc([q]); p = F.softmax(_router(x, m)[0], -1)
    conf, idx = p.max(0)
    dom = R_DOMAINS[int(idx)]
    return (dom if float(conf) > thresh else "geral"), float(conf)

# ---------- MEMÓRIA ----------
def load_mem():
    try: return json.load(open(MEM_FILE))
    except Exception: return {"facts": [], "history": []}
def save_mem(m): json.dump(m, open(MEM_FILE, "w"), ensure_ascii=False, indent=1)

SYS = ("Você é o IARA, um agente que responde em PT-BR, raciocina e usa a memória do usuário. "
       "Seja claro e direto.")

class IARA:
    def __init__(self):
        self.byte = ByteSpecialist(); self.gen = Generalist(); self.mem = load_mem()
        print(f"[IARA] especialista byte carregado ({self.byte.params:.0f}M) | generalista Qwen3-4B lazy", flush=True)

    def ask(self, q):
        # memória: se o usuário afirma um fato pessoal, guarda
        m = re.match(r"(?:meu nome é|me chamo|eu (?:uso|programo em|trabalho com)) (.+)", q.lower())
        if m: self.mem["facts"].append(q); save_mem(self.mem)
        dom, conf = route(q)
        if dom == "fato":
            mm = re.search(r"capital d[eoa]?\s*(?:d[oa])?\s*(.+?)[?.]?$", q.lower())
            if mm:
                pais = mm.group(1).strip()
                art = "do" if pais in ("brasil", "japão", "egito", "méxico", "chile", "peru", "uruguai", "paraguai", "equador") else "da"
                prompt = f"A capital {art} {pais.title()} é"
            else:
                prompt = q.rstrip("?. ") + " é"
            ans = self.byte.answer(prompt, n=40); src = "especialista-byte:fato"
        elif dom == "definicao":
            ans = self.byte.answer(q.rstrip("?. ") + " é", n=42); src = "especialista-byte:definicao"
        elif dom == "codigo":
            ql = q.lower()
            if "fatorial" in ql:   prompt = 'def fatorial(n):\n    """Retorna o fatorial de n."""\n'
            elif "fibonacci" in ql: prompt = 'def fibonacci(n):\n    """Retorna o n-ésimo número de Fibonacci."""\n'
            elif "primo" in ql:    prompt = 'def eh_primo(n):\n    """Retorna True se n for primo."""\n'
            else:
                cls = re.search(r"class(?:e)?\s+(\w+)|(?:classe|objeto)\s+(?:de\s+|para\s+)?(\w+)", ql)
                nm = ((cls.group(1) or cls.group(2)).title() if cls else "MinhaClasse")
                prompt = f"// Classe JavaScript\nclass {nm} {{"
            ans = prompt + self.byte.answer(prompt, n=130, multiline=True); src = "especialista-byte:codigo"
        else:
            ans = self.gen.answer(self.mem["history"][-6:], q, SYS + " Fatos do usuário: " + "; ".join(self.mem["facts"]))
            src = "generalista:Qwen3-4B"
        self.mem["history"] += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        save_mem(self.mem)
        return src, ans

if __name__ == "__main__":
    a = IARA()
    if "--demo" in sys.argv:
        for q in ["Meu nome é Leonardo e programo em Rust.",
                  "Qual é a capital da Itália?",
                  "escreva uma função fatorial",
                  "Em programação, o que é recursão?",
                  "Considerando o que você sabe de mim, que projeto combinaria comigo? Pense."]:
            src, ans = a.ask(q)
            print(f"\n🧑 {q}\n🔀 [{src}]\n🤖 {ans}", flush=True)
    else:
        print("IARA agent. Ctrl-D pra sair.")
        while True:
            try: q = input("\n🧑 ").strip()
            except EOFError: break
            if q:
                src, ans = a.ask(q); print(f"🔀 [{src}]\n🤖 {ans}", flush=True)
