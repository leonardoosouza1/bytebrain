"""Persistent-context (prefix KV cache) benchmark for Qwen2.5-Coder-0.5B-Instruct on the RX 6750 XT.

Phases (all timings use torch.cuda.synchronize):
  i    prefill of the whole calculator file from scratch (batch 1), plus the cost of keeping/moving
       the resulting KV: VRAM clone, expand to batch 16, VRAM -> pinned RAM -> VRAM round trip.
  ii   16 samples x 256 new tokens for one function:
         A  reuse the stored prefix KV (expanded to batch 16), only the ask suffix is prefilled
         B  re-prefill the full prompt once per sample (what num_return_sequences=16 does), same decode
       plus an exactness check: next-token logits via cache+suffix vs one full forward.
  iii  plain decode speed on the small gen.py-style prompt: batch 1 / 16, with and without stop_strings,
       a manual forward loop (no generate overhead), and the exact gen.py config -- to explain the 5 tok/s.
  iv   edit-validity probe for "per-function KV blocks": one function body is edited; compare the
       exact cache against stitched approximations (stale, block recomputed in isolation, block
       recomputed with true prefix; later blocks RoPE-shifted) by KL / top-1 agreement on the suffix.

VRAM estimate (peak, bf16): weights 0.99 GB; phase ii-B holds 16 per-sample caches of ~4.1k tokens
(16 x 4.1k x 12 KiB = 0.81 GB) plus their stacked copy (0.81 GB) -> ~2.7 GB; phase i 4k prefill with
SDPA math fallback materialises 14 x 4k x 4k scores per layer (~0.45-0.9 GB) -> ~2.0 GB. Expected peak
~2.7 GB. A hard cap is set with torch.cuda.set_per_process_memory_fraction (default 4 GB): exceeding
it raises OOM instead of spilling. Expected GPU time with an idle GPU: ~40-90 s.

Run (GPU idle, nothing else on the card):
  cd /home/leonardo/projects/LLM/bytebrain/research/hybrid_ideas/persistent_kv && \
  HF_HOME=/media/llm-data/models/hf HF_HUB_OFFLINE=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  /home/leonardo/projects/LLM/.venv-rocm/bin/python kv_bench.py
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "codegen_calc"))
import spec  # noqa: E402

CALC = "/media/llm-data/universe-runs/codegen-calc-20261021/calc_generated.py"
OUT = "/media/llm-data/universe-runs/hybrid-ideas-20261021/persistent_kv"
MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
NEVER_STOP = "§§§"  # 3-char stop string that never occurs: pays StopStringCriteria cost, never fires
ASK = "Implement this function of a scientific calculator in Python. Use only the standard library.\n\n```python\n{}```"
SENT = "@@SENTINEL@@"
NEWTON = ("    if x < 0:\n        raise ValueError('negative input')\n    if x == 0:\n        return 0.0\n"
          "    g = float(x)\n    for _ in range(60):\n        g = 0.5 * (g + x / g)\n    return g\n")
# flops/token: 2 * non-embedding linear params; attention adds 4 * d_attn * ctx per layer
CFG = dict(L=24, d=896, h=14, kvh=2, hd=64, ff=4864, V=151936)


def lin_params(c=CFG):
    return c["L"] * (c["d"] * (2 * c["d"] + 2 * c["kvh"] * c["hd"]) + 3 * c["d"] * c["ff"])


def prefill_flops(n, ctx0=0, c=CFG):
    """FLOPs to prefill n tokens that start at position ctx0 (logits only for the last token)."""
    attn = 4 * c["h"] * c["hd"] * c["L"] * sum(range(ctx0 + 1, ctx0 + n + 1))
    return 2 * lin_params(c) * n + attn + 2 * c["d"] * c["V"]


def sync():
    torch.cuda.synchronize()


def now():
    sync()
    return time.perf_counter()


def kvs(cache):
    return [(layer.keys, layer.values) for layer in cache.layers]


def make(pairs):
    return DynamicCache(ddp_cache_data=[(k, v) for k, v in pairs])


def clone(cache, start=0, stop=None):
    return make([(k[:, :, start:stop].clone(), v[:, :, start:stop].clone()) for k, v in kvs(cache)])


def expand(cache, n):
    return make([(k.repeat_interleave(n, 0), v.repeat_interleave(n, 0)) for k, v in kvs(cache)])


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 3)


def gpu_busy():
    try:
        with open("/sys/class/drm/card1/device/gpu_busy_percent") as fh:
            return int(fh.read())
    except OSError:
        return None


@torch.no_grad()
def forward(model, ids, cache, pos=None, keep=1):
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=keep, position_ids=pos)
    return out.logits


def build_prompt(tok, dev):
    src = open(CALC).read()
    starts = [m.start() for m in re.finditer(r"^import math\n\ndef ", src, re.M)] + [len(src)]
    chunks = [src[a:b] for a, b in zip(starts, starts[1:])]
    assert "".join(chunks) == src[starts[0]:]
    names = [re.search(r"def (\w+)", c).group(1) for c in chunks]
    tmpl = tok.apply_chat_template([{"role": "user", "content": SENT}], tokenize=False, add_generation_prompt=True)
    pre, post = tmpl.split(SENT)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    head = enc(pre + "Here is the module calc.py:\n```python\n")
    chunk_ids = [enc(c) for c in chunks]
    tail = enc("```\n\n")
    return dict(head=head, chunks=chunks, chunk_ids=chunk_ids, names=names, tail=tail, post=post, enc=enc)


def suffix_ids(P, name):
    e = next(x for x in spec.SPEC if x["name"] == name)
    txt = (f"Rewrite the function `{name}` so that it passes its doctests. Use only the standard library."
           + P["post"] + "```python\nimport math\n\n" + spec.stub(e))
    return P["enc"](txt)


def prefix_of(P, chunk_ids=None):
    ids = list(P["head"])
    for c in (chunk_ids or P["chunk_ids"]):
        ids += c
    return ids + P["tail"]


def T(ids, dev):
    return torch.tensor([ids], device=dev)


def phase_prefill(model, P, dev, reps):
    ids = T(prefix_of(P), dev)
    n = ids.shape[1]
    times = []
    for _ in range(reps):
        t = now()
        cache = DynamicCache()
        forward(model, ids, cache)
        times.append(now() - t)
    med = statistics.median(times)
    nbytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kvs(cache))
    t = now(); c2 = clone(cache); t_clone = now() - t; del c2
    t = now(); c16 = expand(cache, 16); t_exp = now() - t; del c16
    host = [(k.to("cpu").pin_memory(), v.to("cpu").pin_memory()) for k, v in kvs(cache)]
    t = now(); back = make([(k.to(dev, non_blocking=True), v.to(dev, non_blocking=True)) for k, v in host]); t_h2d = now() - t
    del back, host
    fl = prefill_flops(n)
    return cache, dict(tokens=n, s_all=[round(x, 4) for x in times], s_median=round(med, 4),
                       tok_per_s=round(n / med, 1), flops=fl, eff_tflops=round(fl / med / 1e12, 2),
                       kv_bytes=nbytes, kv_bytes_per_token=nbytes // n, s_vram_clone=round(t_clone, 5),
                       s_expand16=round(t_exp, 5), s_ram_to_vram=round(t_h2d, 5),
                       ram_to_vram_gb_s=round(nbytes / t_h2d / 1e9, 2), peak_gb=peak_gb())


def gen_kw(tok, n_new, sample):
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new, pad_token_id=tok.eos_token_id)
    kw.update(dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False))
    return kw


@torch.no_grad()
def phase_samples(model, tok, P, dev, prefix_cache, func, n_samp, n_new):
    pre = prefix_of(P)
    full = pre + suffix_ids(P, func)
    ids16 = T(full, dev).repeat(n_samp, 1)
    res = dict(func=func, prefix_tokens=len(pre), suffix_tokens=len(full) - len(pre), samples=n_samp, new_tokens=n_new)
    # exactness: cached prefix + suffix vs one full forward
    a = forward(model, T(full[len(pre):], dev), clone(prefix_cache))[0, -1].float()
    b = forward(model, T(full, dev), DynamicCache())[0, -1].float()
    res["exact_check"] = dict(max_abs_logit_diff=round(float((a - b).abs().max()), 4), argmax_equal=bool(a.argmax() == b.argmax()))
    torch.cuda.reset_peak_memory_stats()
    t0 = now()
    c16 = expand(prefix_cache, n_samp)
    t1 = now()
    out = model.generate(input_ids=ids16, attention_mask=torch.ones_like(ids16), past_key_values=c16, **gen_kw(tok, n_new, True))
    t2 = now()
    res["A_reuse"] = dict(s_expand=round(t1 - t0, 4), s_generate=round(t2 - t1, 3), s_total=round(t2 - t0, 3),
                          gen_len=int(out.shape[1] - ids16.shape[1]), peak_gb=peak_gb())
    del c16, out
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = now()
    caches = []
    for _ in range(n_samp):
        c = DynamicCache()
        forward(model, T(full[:-1], dev), c)
        caches.append(kvs(c))
    stacked = make([(torch.cat([c[l][0] for c in caches]), torch.cat([c[l][1] for c in caches])) for l in range(len(caches[0]))])
    del caches
    t1 = now()
    out = model.generate(input_ids=ids16, attention_mask=torch.ones_like(ids16), past_key_values=stacked, **gen_kw(tok, n_new, True))
    t2 = now()
    res["B_reprefill"] = dict(s_prefill_x16=round(t1 - t0, 3), s_generate=round(t2 - t1, 3), s_total=round(t2 - t0, 3),
                              gen_len=int(out.shape[1] - ids16.shape[1]), peak_gb=peak_gb())
    res["speedup_total_B_over_A"] = round(res["B_reprefill"]["s_total"] / res["A_reuse"]["s_total"], 2)
    del stacked, out
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def phase_decode(model, tok, dev, func, n_new):
    e = next(x for x in spec.SPEC if x["name"] == func)
    st = spec.stub(e)
    text = tok.apply_chat_template([{"role": "user", "content": ASK.format(st)}], tokenize=False,
                                   add_generation_prompt=True) + "```python\nimport math\n\n\n" + st
    enc = tok([text], return_tensors="pt").to(dev)
    rows = []

    def run(name, bs, sample, stop, forced=True):
        kw = gen_kw(tok, n_new, sample)
        if not forced:
            kw.pop("min_new_tokens")
        if bs > 1:
            kw["num_return_sequences"] = bs
        if stop:
            kw.update(stop_strings=[stop], tokenizer=tok)
        t = now()
        out = model.generate(**enc, **kw)
        dt = now() - t
        steps = int(out.shape[1] - enc["input_ids"].shape[1])
        rows.append(dict(variant=name, batch=bs, stop=stop, s=round(dt, 3), steps=steps,
                         ms_per_step=round(1000 * dt / max(steps, 1), 2), seq_tok_per_s=round(steps / dt, 1),
                         agg_tok_per_s=round(bs * steps / dt, 1), gpu_busy_after=gpu_busy()))

    run("b1_greedy", 1, False, None)
    run("b1_greedy_stop_first_call", 1, False, NEVER_STOP)
    run("b1_greedy_stop", 1, False, NEVER_STOP)
    run("b16_sample", 16, True, None)
    run("b16_sample_stop", 16, True, NEVER_STOP)
    run("genpy_exact_b1_greedy_stop_backticks", 1, False, "```", forced=False)
    # manual loop: raw forward latency without generate / logits processors / stopping criteria
    cache = DynamicCache()
    logits = forward(model, enc["input_ids"], cache)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    t = now()
    for _ in range(n_new):
        nxt = forward(model, nxt, cache)[:, -1].argmax(-1, keepdim=True)
    dt = now() - t
    rows.append(dict(variant="b1_manual_forward_loop", batch=1, stop=None, s=round(dt, 3), steps=n_new,
                     ms_per_step=round(1000 * dt / n_new, 2), seq_tok_per_s=round(n_new / dt, 1),
                     agg_tok_per_s=round(n_new / dt, 1), gpu_busy_after=gpu_busy()))
    return dict(prompt_tokens=int(enc["input_ids"].shape[1]), rows=rows)


def shift(model, k, delta):
    """Re-rotate already-RoPE'd keys by delta positions: R(p+delta) k = R(delta) R(p) k."""
    if delta == 0:
        return k
    pos = torch.full((1, 1), delta, device=k.device)
    cos, sin = model.model.rotary_emb(k.float(), pos)
    cos, sin = cos[:, None], sin[:, None]
    kf = k.float()
    half = kf.shape[-1] // 2
    rot = torch.cat((-kf[..., half:], kf[..., :half]), dim=-1)
    return (kf * cos + rot * sin).to(k.dtype)


@torch.no_grad()
def phase_edit_probe(model, P, dev, func, n_ref):
    j = P["names"].index(func)
    old_chunk = P["chunks"][j]
    m = re.search(r'    """\n', old_chunk[old_chunk.index('"""') + 3:])
    body_at = old_chunk.index('"""') + 3 + m.end()
    new_chunk = old_chunk[:body_at] + NEWTON + "\n"
    new_ids = P["enc"](new_chunk)
    chunk_ids = list(P["chunk_ids"]); chunk_ids[j] = new_ids
    s = len(P["head"]) + sum(len(c) for c in P["chunk_ids"][:j])
    e_old = s + len(P["chunk_ids"][j])
    delta = len(new_ids) - len(P["chunk_ids"][j])
    old_pre, new_pre = prefix_of(P), prefix_of(P, chunk_ids)
    suf = suffix_ids(P, func)
    res = dict(func=func, block_start=s, old_block_tokens=e_old - s, new_block_tokens=len(new_ids), delta=delta,
               tokens_after_block=len(old_pre) - e_old)

    old_cache = DynamicCache(); forward(model, T(old_pre, dev), old_cache)
    t = now(); exact = DynamicCache(); forward(model, T(new_pre, dev), exact); res["s_full_reprefill"] = round(now() - t, 4)
    # reference continuation: greedy n_ref tokens from the exact context
    c = clone(exact)
    nxt = forward(model, T(suf, dev), c)[:, -1].argmax(-1, keepdim=True)
    ref = [int(nxt)]
    for _ in range(n_ref - 1):
        nxt = forward(model, nxt, c)[:, -1].argmax(-1, keepdim=True)
        ref.append(int(nxt))
    probe = T(suf + ref[:-1], dev)
    ref_logits = forward(model, probe, clone(exact), keep=0)[0, len(suf) - 1:].float()

    def score(cache):
        lg = forward(model, probe, clone(cache), keep=0)[0, len(suf) - 1:].float()
        lp, lq = torch.log_softmax(ref_logits, -1), torch.log_softmax(lg, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        return dict(kl_mean=round(float(kl.mean()), 5), kl_max=round(float(kl.max()), 5),
                    top1_agree=round(float((lg.argmax(-1) == ref_logits.argmax(-1)).float().mean()), 4))

    def stitched(block_pairs, t_build):
        pairs = []
        for (ko, vo), (kb, vb) in zip(kvs(old_cache), block_pairs):
            pairs.append((torch.cat([ko[:, :, :s], kb, shift(model, ko[:, :, e_old:], delta)], 2),
                          torch.cat([vo[:, :, :s], vb, vo[:, :, e_old:]], 2)))
        cache = make(pairs)
        return cache, round(now() - t_build, 4)

    variants = {}
    variants["exact_full_reprefill"] = dict(score=score(exact), s_build=res["s_full_reprefill"])
    variants["stale_no_edit"] = dict(score=score(old_cache), s_build=0.0)
    t = now(); ca = clone(old_cache, 0, s); forward(model, T(new_pre[s:], dev), ca)
    variants["a_recompute_suffix"] = dict(score=score(ca), s_build=round(now() - t, 4)); del ca
    t = now(); cb = DynamicCache(); forward(model, T(new_ids, dev), cb, pos=torch.arange(s, s + len(new_ids), device=dev)[None])
    cache, sb = stitched(kvs(cb), t)
    variants["b_block_isolated_shifted"] = dict(score=score(cache), s_build=sb); del cb, cache
    t = now(); cc = clone(old_cache, 0, s); forward(model, T(new_ids, dev), cc)
    cache, sb = stitched([(k[:, :, s:], v[:, :, s:]) for k, v in kvs(cc)], t)
    variants["b_block_in_context_shifted"] = dict(score=score(cache), s_build=sb)
    # RoPE-shift sanity: layer-0 keys depend only on token + position, so shifted stale == exact
    k_sh = kvs(cache)[0][0][:, :, s + len(new_ids):].float()
    k_ex = kvs(exact)[0][0][:, :, s + len(new_ids):].float()
    res["rope_shift_layer0_max_abs_diff"] = round(float((k_sh - k_ex).abs().max()), 5)
    res["rope_shift_layer0_key_abs_max"] = round(float(k_ex.abs().max()), 3)
    del cc, cache
    res["variants"] = variants
    res["n_ref"] = n_ref
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="sqrt")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ref-tokens", type=int, default=48)
    ap.add_argument("--vram-cap-gb", type=float, default=4.0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--skip", default="", help="comma list of phases to skip: i,ii,iii,iv")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gb * 1e9 / total), 0)
    rep = dict(args=vars(args), torch=torch.__version__, hip=torch.version.hip,
               device=torch.cuda.get_device_name(0), gpu_busy_at_start=gpu_busy(),
               env={k: os.environ.get(k) for k in ("HSA_CU_MASK", "HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES",
                                                   "HIP_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
               load_avg=os.getloadavg(), flops_model=dict(lin_params=lin_params(), lm_head_params=CFG["d"] * CFG["V"]))
    t_all = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=args.attn)
    model.eval()
    rep["attn_impl"] = model.config._attn_implementation
    torch.manual_seed(0)
    P = build_prompt(tok, dev)
    rep["prompt"] = dict(file_tokens=sum(len(c) for c in P["chunk_ids"]), functions=len(P["names"]),
                         head_tokens=len(P["head"]), tail_tokens=len(P["tail"]))
    # warmup (kernels, allocator)
    w = T(prefix_of(P)[:256], dev)
    forward(model, w, DynamicCache())
    model.generate(input_ids=w, attention_mask=torch.ones_like(w), max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    sync()

    def save():
        rep["wall_s"] = round(time.perf_counter() - t_all, 1)
        rep["peak_gb_overall"] = peak_gb()
        path = os.path.join(OUT, f"kv_bench_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as fh:
            json.dump(rep, fh, indent=1)
        return path

    torch.cuda.reset_peak_memory_stats()
    cache, rep["i_prefill"] = phase_prefill(model, P, dev, args.reps)
    print("i", json.dumps(rep["i_prefill"]), flush=True)
    if "ii" not in skip:
        rep["ii_samples"] = phase_samples(model, tok, P, dev, cache, args.func, args.samples, args.new)
        print("ii", json.dumps(rep["ii_samples"]), flush=True)
    del cache
    torch.cuda.empty_cache()
    if "iii" not in skip:
        rep["iii_decode"] = phase_decode(model, tok, dev, args.func, args.decode_tokens)
        for r in rep["iii_decode"]["rows"]:
            print("iii", json.dumps(r), flush=True)
    if "iv" not in skip:
        rep["iv_edit_probe"] = phase_edit_probe(model, P, dev, args.func, args.ref_tokens)
        print("iv", json.dumps(rep["iv_edit_probe"]), flush=True)
    print("saved", save(), flush=True)


if __name__ == "__main__":
    main()
