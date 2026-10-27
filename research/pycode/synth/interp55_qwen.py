"""Qwen baselines for the statement-level writer (ledger #55 / I6, 10-27). Same data, same prompts, same K, same
selector (interp55_eval.py) as the byte writers in interp55_writer.py.

  lora     Qwen2.5-Coder-{0.5B,1.5B}-Instruct + LoRA r16 (all attention + MLP projections) on the 1050 train units
           (chat: system + unit prompt -> body). Loss on the body tokens + <|im_end|> only. HF weights from
           /media/llm-data/models/hf/hub, local_files_only. Candidates: 0 greedy, 1..K-1 T=--temp sampling.
  fewshot  Qwen2.5-Coder-3B-Instruct q8_0 GGUF on llama-server (-ngl 99), no training; 2 shots = the 2 nearest TRAIN
           units of the same unit name (byte 4-gram cosine, same retrieval as the `nn` floor), then the query.
           For a unit whose prompt is a train prompt the nearest shot IS its gold body (retrieval copies it; the
           per-function eval units are never train prompts). Candidates: 0 greedy, then seeds 1..K-1 at T=--temp.

Usage:
  interp55_qwen.py lora --data D/data.json --out DIR --model Qwen/Qwen2.5-Coder-0.5B-Instruct --seed 0
  interp55_qwen.py fewshot --data D/data.json --out DIR [--prompts func,test_comp]
  CPU smoke: lora --device cpu --smoke ; fewshot --smoke --gguf <0.5B q8> --ngl 0
Pre-registered predictions (P4, P5, P7): interp55_writer.py header, 2026-10-27 13:06.
Budget note (not equal-budget with the byte writers, by design of the brief): Qwen arms bring pretraining
(0.5B/1.5B/3B params) and the byte writers ~6.5 M from scratch; same data, prompts, K=8 and selector.
"""

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp55_eval as EV  # noqa: E402

CACHE = "/media/llm-data/models/hf/hub"
SYS = ("You write Python. Given a function signature, the signatures of the functions it may call and its contract, "
       "answer with the function BODY only: no def line, no markdown, indentation starting at column 0.")
LLAMA_BIN = "/media/llm-data/scratch/llama-hip-build/bin/llama-server"
GGUF_3B = "/media/llm-data/models/gguf/qwen2.5-coder-3b-instruct-q8_0.gguf"
PORT = 8157


def chat(turns, user):
    s = f"<|im_start|>system\n{SYS}<|im_end|>\n"
    for u, a in turns:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def gen_cap(data, tok=None):
    """generation cap in Qwen tokens, identical for LoRA and few-shot: longest gold body (ALL units, eval too) + 64."""
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", cache_dir=CACHE, local_files_only=True)
    bodies = {u["body"] for row in data["variants"].values() for u in row["units"]}
    return max(len(tok(b, add_special_tokens=False).input_ids) for b in bodies) + 64


def prompt_set(data, which):
    V = data["variants"]
    ps = set()
    for w in which.split(","):
        if w == "func":
            ps |= {fe["prompt"] for fe in data["func_eval"]}
        else:
            for k in data[w]:
                ps |= {u["prompt"] for u in V[k]["units"]}
    return sorted(ps)


# ------------------------------------------------------------------------------------------------------------ lora
def lora(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = a.device
    if dev.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(a.seed)
    data = json.load(open(a.data))
    tr = data["train_units"]
    prompts = prompt_set(data, a.prompts)
    if a.probe:                                         # VRAM probe: the longest train units, 3 steps, no gen
        tr = sorted(tr, key=lambda u: -(len(u["prompt"]) + len(u["body"])))[:a.batch]
        a.steps = 3
    if a.smoke:
        tr = sorted(tr, key=lambda u: len(u["body"]))[:8]
        prompts = [u["prompt"] for u in tr[:2]]
        a.steps, a.K, a.batch, a.micro = 3, 2, 2, 1
    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=CACHE, local_files_only=True)
    tok.padding_side = "left"
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    t_load = time.time()
    kw = {"device_map": {"": 0}} if dev.startswith("cuda") else {}
    model = AutoModelForCausalLM.from_pretrained(a.model, cache_dir=CACHE, local_files_only=True,
                                                 dtype=torch.bfloat16 if dev.startswith("cuda") else torch.float32, **kw)
    if not dev.startswith("cuda"):
        model = model.to(dev)
    n_base = sum(p.numel() for p in model.parameters())
    if a.grad_ckpt:                                     # review 10-27: activation memory of 1.5B at micro 4
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = []
    for u in tr:
        p = tok(chat([], u["prompt"]), add_special_tokens=False).input_ids
        t = tok(u["body"], add_special_tokens=False).input_ids + [im_end]
        enc.append((p, t))
    max_new = gen_cap(data, tok)                        # gate 6: cap > longest gold body (eval too), same for 3B
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    r = random.Random(a.seed)
    order = list(range(len(enc)))
    log = []
    model.train()
    t0 = time.time()
    for s in range(a.steps):
        if s * a.batch % len(enc) < a.batch:
            r.shuffle(order)
        idx = [order[(s * a.batch + i) % len(enc)] for i in range(a.batch)]
        lr = a.lr * min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        tot = 0.0
        for j in range(0, a.batch, a.micro):
            part = [enc[i] for i in idx[j:j + a.micro]]
            T = max(len(p) + len(t) for p, t in part)
            ids = torch.full((len(part), T), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(part), T), -100, dtype=torch.long)
            am = torch.zeros((len(part), T), dtype=torch.long)
            for q, (p, t) in enumerate(part):
                ids[q, :len(p) + len(t)] = torch.tensor(p + t)
                lab[q, len(p):len(p) + len(t)] = torch.tensor(t)
                am[q, :len(p) + len(t)] = 1
            out = model(input_ids=ids.to(dev), attention_mask=am.to(dev), labels=lab.to(dev))
            loss = out.loss * (len(part) / a.batch)
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if s % 25 == 0 or s == a.steps - 1:
            row = {"step": s, "loss": round(tot, 4), "lr": round(lr, 6), "s": round(time.time() - t0, 1)}
            if dev.startswith("cuda"):
                row["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            log.append(row)
            print(json.dumps(row), flush=True)
    if a.probe:
        print(json.dumps({"probe_ok": True, "vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                          if dev.startswith("cuda") else None, "micro": a.micro}), flush=True)
        return
    train_s = round(time.time() - t0, 1)
    model = model.merge_and_unload().eval()
    from transformers import GenerationConfig            # review 10-27 BLOCKER: hub config had repetition_penalty
    model.generation_config = GenerationConfig(eos_token_id=im_end, pad_token_id=tok.pad_token_id)  # 1.05/1.1
    if a.grad_ckpt:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    t0 = time.time()
    cands = {}
    with torch.no_grad():
        for b in range(0, len(prompts), a.gen_prompts):
            ps = prompts[b:b + a.gen_prompts]
            x = tok([chat([], p) for p in ps], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            L = x.input_ids.shape[1]
            g = model.generate(**x, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                               eos_token_id=im_end,
                               pad_token_id=tok.pad_token_id)
            outs = [[tok.decode(g[i, L:], skip_special_tokens=True)] for i in range(len(ps))]
            if a.K > 1:
                torch.manual_seed(a.seed * 100003 + b)
                g = model.generate(**x, max_new_tokens=max_new, do_sample=True, temperature=a.temp, top_k=0, top_p=1.0,
                                   repetition_penalty=1.0,
                                   num_return_sequences=a.K - 1, eos_token_id=im_end, pad_token_id=tok.pad_token_id)
                for i in range(len(ps)):
                    for j in range(a.K - 1):
                        outs[i].append(tok.decode(g[i * (a.K - 1) + j, L:], skip_special_tokens=True))
            for p, o in zip(ps, outs):
                cands[EV.sha(p)] = [EV.clean_body(t) for t in o]
            print(json.dumps({"gen": b + len(ps), "of": len(prompts), "s": round(time.time() - t0, 1)}), flush=True)
    tag = ("q05" if "0.5B" in a.model else "q15" if "1.5B" in a.model else "q30") + "lora"
    meta = {"model": a.model, "params_base": n_base, "params_lora": n_lora, "grad_ckpt": a.grad_ckpt, "micro": a.micro, "steps": a.steps, "batch": a.batch,
            "lr": a.lr, "rank": a.rank, "train_s": train_s, "gen_s": round(time.time() - t0, 1), "load_s":
            round(time.time() - t_load, 1), "max_new": max_new, "K": a.K, "temp": a.temp, "n_train": len(tr),
            "n_prompts": len(prompts), "smoke": a.smoke, "loss_log": log}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"arm": tag, "seed": a.seed, "K": a.K, "meta": meta, "cands": cands},
              open(os.path.join(a.out, f"{tag}_s{a.seed}.cands.json"), "w"))
    print(json.dumps({k: meta[k] for k in ("params_base", "params_lora", "train_s", "gen_s", "max_new")}), flush=True)


# --------------------------------------------------------------------------------------------------------- fewshot
def nearest_shots(data, prompts, k=2):
    import collections
    tr = collections.defaultdict(list)
    for u in data["train_units"]:
        tr[u["name"]].append((EV.ngrams(u["prompt"]), u["prompt"], u["body"]))
    name_of = {u["prompt"]: u["name"] for row in data["variants"].values() for u in row["units"]}
    out = {}
    for p in prompts:
        q = EV.ngrams(p)
        sc = sorted(tr[name_of[p]], key=lambda x: -EV.cos(q, x[0]))
        out[p] = [(sp, sb) for _, sp, sb in sc[:k]][::-1]        # nearest shot last (closest to the query)
    return out


def llama(url, prompt, n_predict, temp, seed):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": temp, "top_k": 0 if temp > 0 else 1,
            "top_p": 1.0, "min_p": 0.0, "seed": seed, "cache_prompt": True, "stop": ["<|im_end|>"]}
    rq = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=900) as f:
        return json.loads(f.read())["content"]


def fewshot(a):
    data = json.load(open(a.data))
    prompts = prompt_set(data, a.prompts)
    if a.smoke:
        prompts = [fe["prompt"] for fe in data["func_eval"]][:2]
        a.K = 2
    shots = nearest_shots(data, prompts)
    url = f"http://127.0.0.1:{PORT}"
    with socket.socket() as sk:
        sk.settimeout(2)
        if sk.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} already in use: refusing to start (not our server)")
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(LLAMA_BIN) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    logf = open(os.path.join(a.out, "llama_server.log"), "w")
    srv = subprocess.Popen([LLAMA_BIN, "-m", a.gguf, "-ngl", str(a.ngl), "-c", str(a.ctx * a.np), "-np", str(a.np),
                            "-t", "2", "--host", "127.0.0.1", "--port", str(PORT)], stdout=logf,
                           stderr=subprocess.STDOUT, env=env, start_new_session=True)
    open(os.path.join(a.out, "llama_server.pid"), "w").write(str(srv.pid))
    try:
        for _ in range(240):
            if srv.poll() is not None:
                raise RuntimeError(f"llama-server exited rc={srv.returncode} before healthy")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as f:
                    if json.loads(f.read()).get("status") == "ok":
                        break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("llama-server did not become healthy")
        n_pred = gen_cap(data) if not a.smoke else 64
        rawf = os.path.join(a.out, f"q30fewshot_s{a.seed}.raw.json")
        done = {}
        if os.path.exists(rawf) and not a.smoke:      # resume (review 10-27: a timeout must not lose everything)
            done = {k: v for k, v in json.load(open(rawf)).items() if not v.startswith("<error")}
        jobs = [(p, j) for p in prompts for j in range(a.K) if f"{EV.sha(p)}|{j}" not in done]
        print(json.dumps({"resumed": len(done), "todo": len(jobs), "n_predict": n_pred}), flush=True)
        t0 = time.time()

        def one(job):
            p, j = job
            pr = chat(shots[p], p)
            try:
                return job, llama(url, pr, n_pred, 0.0 if j == 0 else a.temp, 1000 * a.seed + j)
            except OSError as e:
                return job, f"<error {e}>"
        res = {}
        for k, v in done.items():
            h, j = k.split("|")
            res[(h, int(j))] = v

        def dump():
            json.dump({f"{h}|{j}": v for (h, j), v in res.items()}, open(rawf + ".tmp", "w"))
            os.replace(rawf + ".tmp", rawf)
        with ThreadPoolExecutor(a.np) as ex:
            for n, ((p, j), txt) in enumerate(ex.map(one, jobs)):
                res[(EV.sha(p), j)] = txt
                if n % 200 == 0:
                    dump()
                    print(json.dumps({"done": n, "of": len(jobs), "s": round(time.time() - t0, 1)}), flush=True)
        dump()
        cands = {EV.sha(p): [EV.clean_body(res[(EV.sha(p), j)]) for j in range(a.K)] for p in prompts}
        errs = sum(v.startswith("<error") for v in res.values())
        meta = {"model": a.gguf, "params_base": 3.09e9 if "3b" in a.gguf.lower() else None, "shots": 2, "K": a.K, "temp": a.temp, "n_prompts": len(prompts),
                "gen_s": round(time.time() - t0, 1), "errors": errs, "n_predict": n_pred, "np": a.np,
                "smoke": a.smoke}
        json.dump({"arm": "q30fewshot", "seed": a.seed, "K": a.K, "meta": meta, "cands": cands},
                  open(os.path.join(a.out, f"q30fewshot_s{a.seed}.cands.json"), "w"))
        print(json.dumps({k: meta[k] for k in ("gen_s", "errors", "n_prompts")}), flush=True)
    finally:
        try:
            os.killpg(srv.pid, 15)
        except ProcessLookupError:
            pass
        srv.wait(timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lora", "fewshot"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--gen-prompts", type=int, default=8)
    ap.add_argument("--prompts", default="func,test_comp,test_seen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-frac", type=float, default=0.75)
    ap.add_argument("--gguf", default=GGUF_3B)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--np", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4096, help="context per slot")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true", help="VRAM probe: 3 steps on the longest batch, no generation")
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()
    {"lora": lora, "fewshot": fewshot}[a.cmd](a)


if __name__ == "__main__":
    main()
