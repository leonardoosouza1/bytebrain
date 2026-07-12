#!/usr/bin/env python3
"""M144 — GAP REAL: código com EXECUÇÃO (pass@1). base-1.5B (não-código) vs +delta_code(Coder) vs Coder.
Gera o corpo de funções e RODA em casos de teste. Se o Coder ganha de verdade, o transplante de pesos
(base + delta_code) deve subir a base pro nível do Coder = o efeito "matador" com gap real.
RAM-safe (per-tensor). GPU compute. Execução sandboxed das minhas próprias funções simples (timeout).
Dump marco144_metrics.json."""
import json, time, gc, signal
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco144_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE = f"{MODELS}/Qwen2.5-1.5B"; CODE = f"{MODELS}/Qwen2.5-Coder-1.5B"
tok = AutoTokenizer.from_pretrained(BASE)

TASKS = [
    ("def soma(a, b):\n    ", "# retorna a soma de a e b", "soma", [((2, 3), 5), ((10, -4), 6)]),
    ("def par(n):\n    ", "# retorna True se n é par, False caso contrário", "par", [((4,), True), ((7,), False)]),
    ("def inverte(s):\n    ", "# retorna a string s invertida", "inverte", [(("abc",), "cba")]),
    ("def maximo(L):\n    ", "# retorna o maior elemento da lista L", "maximo", [(([1, 5, 3],), 5)]),
    ("def fatorial(n):\n    ", "# retorna o fatorial de n", "fatorial", [((5,), 120), ((0,), 1)]),
    ("def conta_vogais(s):\n    ", "# conta quantas vogais aeiou tem em s", "conta_vogais", [(("banana",), 3)]),
    ("def fib(n):\n    ", "# n-esimo Fibonacci, fib(0)=0, fib(1)=1", "fib", [((7,), 13), ((1,), 1)]),
    ("def primo(n):\n    ", "# retorna True se n e primo", "primo", [((7,), True), ((8,), False)]),
    ("def soma_lista(L):\n    ", "# soma dos elementos da lista L", "soma_lista", [(([1, 2, 3],), 6)]),
    ("def repete(s, n):\n    ", "# retorna s repetida n vezes", "repete", [(("ab", 3), "ababab")]),
    ("def maiuscula(s):\n    ", "# retorna s em maiusculas", "maiuscula", [(("abc",), "ABC")]),
    ("def pares(L):\n    ", "# retorna lista com os numeros pares de L", "pares", [(([1, 2, 3, 4],), [2, 4])]),
]

class TO(Exception): pass
def _to(sig, fr): raise TO()

@torch.no_grad()
def genbody(sig, doc):
    prompt = sig + doc + "\n    "
    enc = tok(prompt, return_tensors="pt").to(DEV)
    o = gm.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
    body = tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    lines = []
    for ln in body.split("\n"):
        if ln.strip() and not ln.startswith(" ") and not ln.startswith("\t"): break  # dedent = fim da função
        if ln.strip().startswith("def ") or ln.strip().startswith("#") and lines and not ln.startswith(" "): break
        lines.append(ln)
    return sig + doc + "\n    " + "\n".join(lines)

def run_task(sig, doc, name, tests):
    code = genbody(sig, doc)
    ns = {}
    signal.signal(signal.SIGALRM, _to); signal.alarm(3)
    try:
        exec(code, ns)
        f = ns.get(name)
        ok = all(f(*a) == e for a, e in tests) if callable(f) else False
    except Exception:
        ok = False
    finally:
        signal.alarm(0)
    return ok

def evalc(): return sum(run_task(*t) for t in TASKS)

mtmp = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
base_sd = {k: v.clone() for k, v in mtmp.state_dict().items()}; del mtmp; gc.collect()
gm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to(DEV).eval()

@torch.no_grad()
def set_to(donors):
    for k, p in gm.named_parameters():
        if k in base_sd: p.data = base_sd[k].to(DEV).clone()
    for D in donors:
        mt = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16); dsd = mt.state_dict()
        for k, p in gm.named_parameters():
            if k in base_sd and k in dsd and dsd[k].shape == p.shape:
                d = dsd[k].to(DEV).float() - base_sd[k].to(DEV).float()
                s = (d.abs().max() / 127).clamp_min(1e-8); d = torch.round(d / s).clamp(-127, 127) * s
                p.data = (p.data.float() + d).to(p.dtype); del d
        del mt, dsd; gc.collect(); torch.cuda.empty_cache()

res = {"n": len(TASKS), "conds": {}}
for name, donors in [("base", []), ("base+delta_code(int8)", [CODE]), ("Coder(doador)", [])]:
    if name == "Coder(doador)":
        del gm; gc.collect(); torch.cuda.empty_cache()
        gm = AutoModelForCausalLM.from_pretrained(CODE, dtype=torch.float16).to(DEV).eval()
    else:
        set_to(donors)
    k = evalc(); res["conds"][name] = {"pass": k, "de": len(TASKS)}
    log(f"  {name:22}: pass@1 {k}/{len(TASKS)}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== gap real: se Coder >> base e base+delta ~ Coder → transplante de pesos vale ===")
log(f"DONE M144 ({time.time()-t0:.0f}s)")
