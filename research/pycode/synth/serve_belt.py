"""Serving path of the belt (ledger #41): a request in Portuguese -> runnable Python, in seconds.

  understander  Qwen2.5-Coder-0.5B + LoRA student (s5), greedy, the same prompt and decoding as the eval
                (student_understander.evaluate) -> JSON spec
  bridge        e2e_pipeline.check_spec (rule) -> (world, ops) or validator flags
  planner       blockaddr net (e2e ckpt) -> block plan
  writer        multifn.Writer net (e2e ckpt) -> text of every function block
  assembler     e2e_pipeline.assemble -> one module with main()
  run           bwrap, no network, no writes: main() on the probe inputs (CASES)
Judges stay off this path. No gold here: the request has no known answer. What is checked without gold:
  runs          the module loads and main() returns on every probe without an exception, except an exception the
                reference blocks for the same spec raise at the same probe (div/mod give None at b=0, and a formula that
                feeds that None into another op raises by itself: noted as inherent, as in run_belt)
  spec_check    the assembled module gives the same outputs as the reference blocks for the SAME spec
                (planner + writer faithful to the spec; says nothing about whether the spec matches the request)
  status "ok" needs runs, spec_check and no validator flag (an unknown op in a dispatch spec is dropped by the
  bridge and flagged; it is not "ok").
Latency per stage in ms; the understander is timed with cuda.synchronize, like the eval, after one untimed warm-up
generate at load. "run+ref (bwrap)" covers the module and the reference module in one sandbox call.

HTTP front (--http PORT): GET / is a one-page chat (request in, code + checks + latency out); POST /api/belt with
{"request": "<text>"} returns the answer JSON; GET /health. Bound to 127.0.0.1 only: whoever can call it makes the
sandbox run generated code. One request at a time (single-threaded server; the GPU model is not thread-safe). Body
<= 4 KB, request text <= 500 chars, JSON content type required (a cross-site form post cannot send it without CORS,
and the server sends no CORS headers), Host header must be the loopback address (DNS rebinding).

Usage: serve_belt.py [--lora DIR] [--ckpt DIR] "faça uma função de soma" ["...", ...]
       serve_belt.py --repl              (one request per line on stdin; the model stays loaded)
       serve_belt.py --http 8140         (http://127.0.0.1:8140/)
"""

import argparse
import ast
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_pipeline as E  # noqa: E402
import multifn as MF  # noqa: E402
import student_understander as SU  # noqa: E402

RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/e2e"


class Belt:
    def __init__(self, lora, ckpt, mem_frac):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(SU.MODEL, cache_dir=SU.CACHE, local_files_only=True)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        base = AutoModelForCausalLM.from_pretrained(SU.MODEL, cache_dir=SU.CACHE, local_files_only=True,
                                                    dtype=torch.bfloat16).to("cuda")
        self.model = PeftModel.from_pretrained(base, lora).merge_and_unload().eval()
        fmt = json.load(open(os.path.join(os.path.dirname(os.path.normpath(lora)), "result.json"))).get(
            "spec_format", "expr")
        self.max_new = SU.MAX_NEW[fmt]
        self.P, self.W = E.Planner("net", ckpt), E.Writer("net", ckpt)
        self.understand("faça uma função de soma")              # warm-up (kernels, allocator), not timed
        self.load_s = round(time.perf_counter() - t0, 1)

    @torch.no_grad()
    def understand(self, text):
        ids = torch.tensor([self.tok(SU.prompt(text), add_special_tokens=False).input_ids], device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new,
                                  do_sample=False, eos_token_id=self.im_end, pad_token_id=self.tok.pad_token_id,
                                  repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ms

    def answer(self, text):
        lat = {}
        raw, lat["understander"] = self.understand(text)
        r = {"request": text, "raw_spec": raw}
        t0 = time.perf_counter()
        spec, b, flags = E.check_spec(raw)
        r.update(spec=spec, flags=flags)
        if not b:
            lat["bridge"] = 1000 * (time.perf_counter() - t0)
            if isinstance(spec, dict) and isinstance(spec.get("expr"), str) and \
                    any(f["kind"] == "expr_parse_error" for f in flags):
                r["why"] = why_outside(spec["expr"])
            r["lat_ms"] = {k: round(v, 1) for k, v in lat.items()}
            r["total_ms"] = round(sum(lat.values()), 1)
            r["status"] = "rejected by the validator (no code written)"
            return r
        t1 = time.perf_counter()
        lat["bridge"] = 1000 * (t1 - t0)
        plan = self.P.plan([b])[0]
        t2 = time.perf_counter()
        lat["planner"] = 1000 * (t2 - t1)
        nt = self.W.texts(set(plan))
        src = E.assemble(b[0], plan, lambda s: nt.get(s, ""))
        t3 = time.perf_counter()
        lat["writer+assembler"] = 1000 * (t3 - t2)
        keys = [f"d:{o}" for o in b[1]] if b[0] == "dispatch" else ["x:"]
        B = E.Batch()
        h = B.add(src, keys)
        href = B.add(E.assemble(b[0], E.oracle_plan(b), MF.block_text), keys)
        B.run()
        lat["run+ref (bwrap)"] = 1000 * (time.perf_counter() - t3)
        out, ref = B[h], B[href]
        pairs = [(v, rv) for k in keys
                 for v, rv in zip(out[k] or [None] * len(E.calls_for(k)), ref[k] or [None] * len(E.calls_for(k)))]
        vals = [v for v, _ in pairs]
        calls = [c for k in keys for c in E.calls_for(k)]
        inherent = sum(v is not None and str(v).startswith("!") and v == rv for v, rv in pairs)
        r.update(world=b[0], ops=list(b[1]), plan=plan, code=src,
                 runs=all(v is not None and (not str(v).startswith("!") or v == rv) for v, rv in pairs),
                 inherent_exceptions=inherent,
                 spec_check=all(out[k] == ref[k] for k in keys),
                 probes=[f"{fn}({', '.join(map(repr, args))}) = {v}" for (fn, args), v in zip(calls, vals)][::max(1, len(vals) // 6)][:6])
        ok = r["runs"] and r["spec_check"] and not flags
        r["status"] = "ok" if ok else "written, but a check failed" if not (r["runs"] and r["spec_check"]) else \
            "written, but the validator flagged the spec"
        r["lat_ms"] = {k: round(v, 1) for k, v in lat.items()}
        r["total_ms"] = round(sum(lat.values()), 1)
        return r


def why_outside(e):
    """display only (the flag kind stays expr_parse_error for the eval tools): the first part of a rejected expr that
    is outside the compose world, in words"""
    try:
        node = ast.parse(e, mode="eval").body
    except SyntaxError:
        return "not a Python expression"
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and (len(n.args) != 2 or n.keywords):
            name = n.func.id if isinstance(n.func, ast.Name) else ast.unparse(n.func)
            return (f"{name}() is called with {len(n.args)} argument(s); the compose world has 2-argument ops only "
                    "(e.g. sqrt(a*a + b*b) is hyp(a, b))")
        if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            return "infix operator; ops are calls such as add(a, b)"
        if isinstance(n, ast.Constant):
            return f"number {n.value!r}; leaves are a and b only"
    return "outside the compose world (shape)"


def show(r):
    print(f"\n>>> {r['request']}")
    print(f"spec: {r['raw_spec']}")
    if r.get("code"):
        print(r["code"].rstrip())
        print(f"runs: {r['runs']} (inherent exceptions {r['inherent_exceptions']})   "
              f"spec_check (planner+writer faithful to the spec): {r['spec_check']}")
        for p in r["probes"]:
            print("   ", p)
    if r["flags"]:
        print("flags:", json.dumps(r["flags"], ensure_ascii=False))
    if r.get("why"):
        print("why:", r["why"])
    print(f"status: {r['status']}   latency ms: {json.dumps(r['lat_ms'])}   total {r.get('total_ms')}", flush=True)


PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Belt</title>
<style>
:root { --bg:#111418; --panel:#1a1f26; --ink:#e6e9ee; --muted:#9aa4b1; --ok:#4cc38a; --bad:#e5604d; --line:#2a313b; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:16px; }
form { display:flex; gap:8px; }
input { flex:1; min-width:0; padding:10px 12px; background:var(--panel); color:var(--ink); border:1px solid var(--line);
        border-radius:6px; font:inherit; }
button { padding:10px 16px; background:#2d6cdf; color:#fff; border:0; border-radius:6px; font:inherit; cursor:pointer; }
button:disabled { opacity:.5; cursor:wait; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin-top:14px; }
.req { color:var(--muted); }
pre { white-space:pre-wrap; word-break:break-word; background:#0c0f13; padding:10px; border-radius:6px;
      font:13px/1.4 ui-monospace, monospace; margin:8px 0; }
.ok { color:var(--ok); } .bad { color:var(--bad); }
.lat { color:var(--muted); font-size:13px; }
</style></head><body><main>
<form id="f"><input id="q" maxlength="500" placeholder="faça uma função de soma" autofocus><button id="b">gerar</button></form>
<div id="out"></div>
</main><script>
const f = document.getElementById("f"), q = document.getElementById("q"), b = document.getElementById("b"),
      out = document.getElementById("out");
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text; return e; }
f.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = q.value.trim(); if (!text) return;
  b.disabled = true;
  const card = el("div", "card"); card.append(el("div", "req", "> " + text)); out.prepend(card);
  try {
    const t0 = performance.now();
    const res = await fetch("/api/belt", { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify({ request: text }) });
    const r = await res.json();
    if (!res.ok) { card.append(el("div", "bad", r.error || ("HTTP " + res.status))); return; }
    card.append(el("div", "", "spec: " + r.raw_spec));
    if (r.code) card.append(el("pre", "", r.code));
    if (r.probes) card.append(el("pre", "", r.probes.join("\\n")));
    if (r.why) card.append(el("div", "bad", "why: " + r.why));
    if (r.flags && r.flags.length) card.append(el("div", "bad", "flags: " + JSON.stringify(r.flags)));
    card.append(el("div", r.status === "ok" ? "ok" : "bad", "status: " + r.status));
    card.append(el("div", "lat", "server ms: " + JSON.stringify(r.lat_ms) + "  total " + r.total_ms +
                   "  |  round trip " + Math.round(performance.now() - t0) + " ms"));
  } catch (e) { card.append(el("div", "bad", String(e))); }
  finally { b.disabled = false; q.select(); }
});
</script></body></html>
"""

MAX_BODY, MAX_TEXT = 4096, 500


def serve_http(belt, port, fh):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        timeout = 10        # review #48 g4 S2: a client that stalls mid-request no longer blocks the server

        def send(self, code, body, ctype="application/json; charset=utf-8"):
            data = (body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def host_ok(self):
            if self.headers.get("Host") in hosts:
                return True
            self.send(421, {"error": "unexpected Host header"})
            return False

        def do_GET(self):
            if not self.host_ok():
                return
            if self.path == "/":
                self.send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/health":
                self.send(200, {"ok": True, "load_s": belt.load_s})
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            if not self.host_ok():
                return
            if self.path != "/api/belt":
                return self.send(404, {"error": "not found"})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "Content-Type must be application/json"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 < n <= MAX_BODY:
                return self.send(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
            try:
                text = json.loads(self.rfile.read(n))["request"]
                if not isinstance(text, str):
                    raise TypeError
            except (ValueError, KeyError, TypeError):
                return self.send(400, {"error": 'body must be {"request": "<text>"}'})
            text = text.strip()
            if not 0 < len(text) <= MAX_TEXT:
                return self.send(400, {"error": f"request must be 1..{MAX_TEXT} characters"})
            try:
                r = belt.answer(text)
            except Exception as e:                              # keep the loaded model alive
                print(f"[http] answer failed: {type(e).__name__}: {e}", flush=True)
                return self.send(500, {"error": f"internal error ({type(e).__name__})"})
            if fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
            self.send(200, r)

        def log_message(self, fmt, *args):
            print(f"[http] {fmt % args}", flush=True)

    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"serving http://127.0.0.1:{port}/", flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests", nargs="*")
    ap.add_argument("--lora", default=f"{RUN}/student_s5_s1/lora")
    ap.add_argument("--ckpt", default=f"{RUN}/ckpt")
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--http", type=int, default=0, help="serve the chat page and /api/belt on 127.0.0.1:PORT")
    ap.add_argument("--out", default=None, help="JSONL of every answer")
    ap.add_argument("--mem-frac", type=float, default=0.25)
    ap.add_argument("--parity", type=int, default=0,
                    help="positive control: the first N requests of RUN/v2pool must give the raw specs the eval saved")
    a = ap.parse_args()
    belt = Belt(a.lora, a.ckpt, a.mem_frac)
    tag = os.path.basename(os.path.dirname(os.path.normpath(a.lora)))
    print(f"belt loaded in {belt.load_s} s (understander {tag})", flush=True)
    if a.parity:
        saved = json.load(open(f"{RUN}/v2pool/{tag}/raws.json"))
        reqs = json.load(open(f"{RUN}/v2pool/requests.json"))[:a.parity]
        diff = [q["id"] for q in reqs if belt.understand(q["text"])[0] != saved[q["id"]]["raw"]]
        print(f"parity: {len(reqs) - len(diff)}/{len(reqs)} raw specs identical to the eval's"
              + (f"; differ: {diff}" if diff else ""), flush=True)
        if diff:
            sys.exit(1)
    fh = open(a.out, "a") if a.out else None
    if a.http:
        serve_http(belt, a.http, fh)
        return
    src = sys.stdin if a.repl else a.requests
    for line in src:
        text = line.strip()
        if not text:
            continue
        try:
            r = belt.answer(text)
        except Exception as e:                                  # keep the loaded model alive in --repl
            print(f"\n>>> {text}\nerror: {type(e).__name__}: {e}", flush=True)
            continue
        show(r)
        if fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
