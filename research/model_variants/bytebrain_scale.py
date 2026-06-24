"""
ByteBrain — BATERIA DE ESCALA (autonoma): ate onde o byte chega? (varias linguas + codigo)

Leonardo deu liberdade (~1h). Objetivo: treinar um byte-TRANSFORMER (GPT-style, vocab 256) num
corpus DIVERSO (29 linguas/scripts da Wikipedia + codigo Py/TS/bash/JS + prosa PT) e VARRER:
  A) tamanho do MODELO (2/4/6/8 camadas) -> curva bits-per-byte vs params
  B) tamanho dos DADOS (10/30/100%)       -> curva bpb vs dados
  C) melhor config -> treino longo + AMOSTRAS geradas (multilingue/codigo)
Mede bpb geral + por-script. Extrapola pro que a AMD (192GB) destravaria. GPU, sem queimar o PC.
Tudo logado em /tmp/bytebrain_scale.log ; metricas em scale_metrics.json ; ckpt do melhor.
"""
import os, re, json, math, time, random, urllib.request, urllib.parse, traceback
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "/home/leonardo/projects/LLM/bytebrain/data/multiscript"
os.makedirs(CACHE, exist_ok=True)
LB = 256
TIME_BUDGET = 2700      # ~45 min de teto pra varredura; depois vai pra geracao/relatorio
t_start = time.time()

WIKI = {  # lang -> titulo (cobre Latin, Cirilico, Grego, Hebraico, Arabe, Persa, Indic, CJK, Georgiano, Armenio, Thai)
    "en": "United States", "pt": "Brasil", "es": "España", "fr": "France", "de": "Deutschland",
    "it": "Italia", "nl": "Nederland", "sv": "Sverige", "pl": "Polska", "tr": "Türkiye",
    "vi": "Việt Nam", "id": "Indonesia", "ru": "Россия", "uk": "Україна", "bg": "България",
    "el": "Ελλάδα", "he": "ישראל", "ar": "مصر", "fa": "ایران", "ur": "پاکستان",
    "hi": "भारत", "bn": "বাংলাদেশ", "ta": "தமிழ் நாடு", "th": "ประเทศไทย",
    "ja": "日本", "ko": "대한민국", "ka": "საქართველო", "hy": "Հայաստան",
}
CODE = {
    "py": "/home/leonardo/projects/LLM/byte-language-lab/data/.oc_corpus.txt",
    "ts": "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_big4.txt",
    "bash": "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_distill.txt",
    "js": "/home/leonardo/projects/LLM/byte-language-lab/data/code_train.txt",
}


def log(*a):
    print(*a, flush=True)


def fetch(lang, title):
    fp = f"{CACHE}/{lang}.txt"
    if os.path.exists(fp) and os.path.getsize(fp) > 2000:
        return open(fp, encoding="utf-8").read()
    url = f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles={urllib.parse.quote(title)}"
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "bytebrain/0.1"}), timeout=20))
        txt = next(iter(d["query"]["pages"].values())).get("extract", "")
        if txt:
            open(fp, "w", encoding="utf-8").write(txt)
        return txt
    except Exception as e:
        log(f"  falha {lang}: {e}"); return ""


def build_corpus():
    log("buscando corpus multilingue...")
    natural = {}
    for lg, ti in WIKI.items():
        t = fetch(lg, ti)
        if len(t) > 1500:
            natural[lg] = t
    log(f"  {len(natural)} linguas ok, {sum(len(v) for v in natural.values()):,} chars")
    code = {}
    for k, p in CODE.items():
        try:
            code[k] = open(p, errors="ignore").read(3_000_000)
        except Exception:
            pass
    log(f"  codigo: {[ (k,len(v)) for k,v in code.items() ]}")
    # bytes por fonte (para val por-categoria)
    src_bytes = {}
    for lg, t in natural.items():
        src_bytes[f"nat:{lg}"] = t.encode("utf-8")
    for k, t in code.items():
        src_bytes[f"code:{k}"] = t.encode("utf-8")
    return src_bytes


def make_split(src_bytes, frac=1.0):
    """concatena (subamostrado por frac) -> train/val byte arrays."""
    tr, va = [], []
    for name, b in src_bytes.items():
        n = int(len(b) * frac)
        b = b[:n]
        cut = int(len(b) * 0.9)
        tr.append(np.frombuffer(b[:cut], np.uint8)); va.append((name, np.frombuffer(b[cut:], np.uint8)))
    train = np.concatenate(tr)
    return train, va


def batch(arr, bs):
    ix = np.random.randint(0, len(arr) - LB - 1, bs)
    x = np.stack([arr[i:i+LB] for i in ix]).astype(np.int64)
    return torch.from_numpy(x)


class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, L, D = x.shape; h = s.ln1(x)
        qkv = s.qkv(h).view(B, L, 3, s.nh, D//s.nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + s.mlp(s.ln2(x))


class ByteGPT(nn.Module):
    def __init__(s, nl, d, nh):
        super().__init__(); s.tok = nn.Embedding(256, d); s.pos = nn.Embedding(LB, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)]); s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, 256)
    def forward(s, x):
        p = torch.arange(x.size(1), device=x.device)
        h = s.tok(x) + s.pos(p)[None]
        for b in s.blocks: h = b(h)
        return s.head(s.lnf(h))


def val_bpb(model, va):
    model.eval(); res = {}
    with torch.no_grad():
        for name, arr in va:
            if len(arr) < LB + 1: continue
            xs = []
            for i in range(0, min(len(arr) - LB - 1, 30 * LB), LB):
                xs.append(arr[i:i+LB])
            if not xs: continue
            x = torch.from_numpy(np.stack(xs).astype(np.int64)).to(DEV)
            lo = model(x[:, :-1]); l = F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1).to(DEV))
            res[name] = l.item() / math.log(2)
    model.train(); return res


def train_model(cfg, train, va, steps, bs=64, lr=3e-4):
    nl, d, nh = cfg
    model = ByteGPT(nl, d, nh).to(DEV)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr); model.train()
    t0 = time.time()
    for st in range(steps):
        try:
            x = batch(train, bs).to(DEV)
            lo = model(x[:, :-1]); loss = F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs = max(8, bs//2); log(f"    OOM -> bs={bs}"); continue
        if st % max(1, steps//4) == 0:
            log(f"    step {st}/{steps} loss {loss.item()/math.log(2):.3f} bpb")
    vb = val_bpb(model, va); overall = float(np.mean(list(vb.values())))
    dt = time.time() - t0
    return model, params, overall, vb, dt


def generate(model, seed, n=200, temp=0.8):
    model.eval(); ids = list(seed.encode("utf-8"))[:LB-n-1] or [32]
    x = torch.tensor([ids], device=DEV)
    with torch.no_grad():
        for _ in range(n):
            lo = model(x[:, -LB:]); p = F.softmax(lo[0, -1]/temp, -1); nx = torch.multinomial(p, 1).item()
            x = torch.cat([x, torch.tensor([[nx]], device=DEV)], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def main():
    log(f"device {DEV} | budget {TIME_BUDGET}s")
    src = build_corpus()
    train_full, va = make_split(src, 1.0)
    log(f"train bytes: {len(train_full):,} | val sources: {len(va)}")
    metrics = {"corpus_bytes": int(len(train_full)), "model_sweep": [], "data_sweep": []}

    # A) SWEEP de MODELO
    log("\n=== A) SWEEP DE MODELO (bpb vs tamanho) ===")
    configs = [(2, 128, 4), (4, 256, 4), (6, 384, 6), (8, 512, 8)]
    best = None
    for cfg in configs:
        if time.time() - t_start > TIME_BUDGET:
            log("  (budget atingido, parando sweep)"); break
        log(f"  config L{cfg[0]} d{cfg[1]} h{cfg[2]} ...")
        try:
            m, p, ov, vb, dt = train_model(cfg, train_full, va, steps=5000, bs=64)
        except Exception as e:
            log(f"  ERRO {cfg}: {e}"); traceback.print_exc(); continue
        nat = float(np.mean([v for k, v in vb.items() if k.startswith("nat:")]))
        cod = float(np.mean([v for k, v in vb.items() if k.startswith("code:")]))
        log(f"    -> {p:,} params | bpb geral {ov:.3f} (natural {nat:.3f} | codigo {cod:.3f}) | {dt:.0f}s")
        metrics["model_sweep"].append({"cfg": cfg, "params": int(p), "bpb": round(ov, 3), "bpb_nat": round(nat, 3), "bpb_code": round(cod, 3), "sec": round(dt)})
        json.dump(metrics, open("/home/leonardo/projects/LLM/bytebrain/scale_metrics.json", "w"), indent=2)
        if best is None or ov < best[1]:
            best = (cfg, ov, m)

    # B) SWEEP de DADOS (config media fixa)
    log("\n=== B) SWEEP DE DADOS (bpb vs quantidade) ===")
    for frac in [0.1, 0.3, 1.0]:
        if time.time() - t_start > TIME_BUDGET: break
        tr_f, va_f = make_split(src, frac)
        log(f"  dados {int(frac*100)}% ({len(tr_f):,} bytes) ...")
        try:
            m, p, ov, vb, dt = train_model((4, 256, 4), tr_f, va, steps=4000, bs=64)
            metrics["data_sweep"].append({"frac": frac, "bytes": int(len(tr_f)), "bpb": round(ov, 3)})
            log(f"    -> bpb {ov:.3f}")
            json.dump(metrics, open("/home/leonardo/projects/LLM/bytebrain/scale_metrics.json", "w"), indent=2)
        except Exception as e:
            log(f"  ERRO data {frac}: {e}")

    # C) MELHOR config -> treino mais longo + amostras + ckpt
    if best is not None:
        cfg = best[0]
        log(f"\n=== C) MELHOR config {cfg}: treino longo + amostras ===")
        try:
            m, p, ov, vb, dt = train_model(cfg, train_full, va, steps=12000, bs=64)
            metrics["best"] = {"cfg": cfg, "params": int(p), "bpb": round(ov, 3),
                               "bpb_por_fonte": {k: round(v, 3) for k, v in sorted(vb.items())}}
            torch.save(m.state_dict(), "/home/leonardo/projects/LLM/bytebrain/best_bytegpt.pt")
            log(f"  bpb final {ov:.3f} | ckpt salvo")
            samples = {}
            for seed in ["The ", "O Brasil ", "Россия ", "def main(", "function ", "日本は", "import "]:
                try:
                    samples[seed] = generate(m, seed, 160)
                    log(f"  [{seed!r}] -> {samples[seed][:120]!r}")
                except Exception as e:
                    log(f"  gen falhou {seed}: {e}")
            metrics["samples"] = samples
        except Exception as e:
            log(f"  ERRO best: {e}"); traceback.print_exc()

    json.dump(metrics, open("/home/leonardo/projects/LLM/bytebrain/scale_metrics.json", "w"), indent=2, ensure_ascii=False)
    log(f"\n=== FIM ({time.time()-t_start:.0f}s) -> scale_metrics.json ===")


if __name__ == "__main__":
    main()
