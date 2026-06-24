"""
ByteBrain — CO-CIENTISTA autonomo. Baterias na zona eficiente (30-40M), metricas profissionais.

Leonardo saiu pra jantar; trabalho: ficar em loop, sem PC ocioso, testes variados + validacao real.
fp32 SEM checkpointing (rapido na RDNA2; bf16 trava). Modelo ~40M (zona dele). Baterias:
  A) SWEEP de zona: ~25M / ~40M / ~57M, 5 min cada -> tempo/step, tok/s, VRAM, bpb (achar o ponto ideal)
  B) GENERALISTA ~40M, 25 min, todas as linguas + codigo -> amostras multilingue/codigo
  C) ESPECIALISTA PT ~40M, 12 min, SO portugues -> quanto demora a virar expert, frases PT,
     e COMPARA: bpb do especialista-PT vs generalista, ambos no PT held-out (transfer vs especializacao)
  D) ESPECIALISTA codigo ~40M, 12 min -> amostras de codigo
Loga tempo/step + VRAM + bpb a cada 50 steps. Salva metricas+amostras em coscientist_metrics.json.
"""
import os, glob, math, time, random, json, urllib.request, urllib.parse, traceback
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda"
CACHE = "/home/leonardo/projects/LLM/bytebrain/data/multiscript"
LANGC = "/home/leonardo/projects/LLM/bytebrain/data/lang_corpus"
os.makedirs(LANGC, exist_ok=True)
BRAIN = "/home/leonardo/projects/LLM/iara-brain"
CODE_FILES = ["/home/leonardo/projects/LLM/byte-language-lab/data/.oc_corpus.txt",
              "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_big4.txt",
              "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_distill.txt",
              "/home/leonardo/projects/LLM/byte-language-lab/data/code_train.txt"]
LB = 256; BATCH = 48; LOG = "/tmp/bytebrain_coscientist.log"
RESULTS = {"baterias": {}, "amostras": {}, "comparacao_PT": {}}


def log(*a):
    print(*a, flush=True)


def fetch(lang, title):
    fp = f"{LANGC}/{lang}__{title.replace('/','_')}.txt"
    if os.path.exists(fp) and os.path.getsize(fp) > 1500:
        return open(fp, encoding="utf-8").read()
    try:
        url = f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles={urllib.parse.quote(title)}"
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "bytebrain-research/0.3"}), timeout=20))
        t = next(iter(d["query"]["pages"].values())).get("extract", "")
        if t: open(fp, "w", encoding="utf-8").write(t)
        return t
    except Exception as e:
        log(f"  fetch falhou {lang}:{title} {e}"); return ""


def enrich():
    log("enriquecendo corpus PT/EN (Wikipedia)...")
    pt_titles = ["Portugal", "Física", "História", "Matemática", "Música", "Futebol",
                 "Inteligência artificial", "Computador", "Ciência", "Filosofia", "Economia", "Biologia"]
    en_titles = ["Physics", "History", "Music", "Computer", "Science", "Philosophy", "Economics", "Biology"]
    pt = open(f"{CACHE}/pt.txt", encoding="utf-8").read() if os.path.exists(f"{CACHE}/pt.txt") else ""
    for f in glob.glob(f"{BRAIN}/*.txt"):
        try: pt += "\n" + open(f, encoding="utf-8").read()
        except Exception: pass
    en = open(f"{CACHE}/en.txt", encoding="utf-8").read() if os.path.exists(f"{CACHE}/en.txt") else ""
    for t in pt_titles:
        pt += "\n" + fetch("pt", t); time.sleep(5)
    for t in en_titles:
        en += "\n" + fetch("en", t); time.sleep(5)
    log(f"  PT {len(pt):,} chars | EN {len(en):,} chars")
    return pt, en


def to_arr(text):
    return np.frombuffer(text.encode("utf-8"), np.uint8)


def split(arr):
    cut = int(len(arr) * 0.95); return arr[:cut], arr[cut:]


class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, L, D = x.shape; h = s.ln1(x); qkv = s.qkv(h).view(B, L, 3, s.nh, D//s.nh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, L, D)); return x + s.mlp(s.ln2(x))


class GPT(nn.Module):
    def __init__(s, d, nl, nh):
        super().__init__(); s.tok = nn.Embedding(256, d); s.pos = nn.Embedding(LB, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)]); s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, 256)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))[None]
        for b in s.blocks: h = b(h)
        return s.head(s.lnf(h))


def batch(arr, bs):
    ix = np.random.randint(0, len(arr)-LB-1, bs)
    return torch.from_numpy(np.stack([arr[i:i+LB] for i in ix]).astype(np.int64)).to(DEV)


def val_bpb(model, varr):
    model.eval(); tot = 0; n = 0
    with torch.no_grad():
        for _ in range(15):
            x = batch(varr, 32); lo = model(x[:, :-1]); tot += F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1)).item(); n += 1
    model.train(); return tot/n/math.log(2)


def train(name, cfg, train_arr, val_arr, minutes):
    d, nl, nh = cfg; model = GPT(d, nl, nh).to(DEV)
    P = sum(p.numel() for p in model.parameters()); opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.1)
    log(f"\n[{name}] {P/1e6:.0f}M params (d{d} L{nl} h{nh}) | fp32 | batch {BATCH} | alvo {minutes}min")
    model.train(); t0 = time.time(); step = 0; tstep = 0
    while time.time() - t0 < minutes*60:
        s0 = time.time(); x = batch(train_arr, BATCH); lo = model(x[:, :-1]); loss = F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1; tstep += time.time()-s0
        if step % 50 == 0:
            sps = tstep/step; vr = torch.cuda.memory_allocated()/1e9; pk = torch.cuda.max_memory_allocated()/1e9
            log(f"  step {step} | bpb {loss.item()/math.log(2):.3f} | {sps*1000:.0f}ms/step | {BATCH*255/sps:.0f} tok/s | VRAM {vr:.1f}/{pk:.1f}GB | {time.time()-t0:.0f}s")
    vb = val_bpb(model, val_arr); sps = tstep/max(1, step)
    log(f"  -> FIM {name}: {P/1e6:.0f}M | val bpb {vb:.3f} | {step} steps | {sps*1000:.0f}ms/step | {BATCH*255/sps:.0f} tok/s")
    return model, {"params_M": round(P/1e6, 1), "val_bpb": round(vb, 3), "steps": step, "ms_step": round(sps*1000), "tok_s": round(BATCH*255/sps)}


def gen(model, seed, n=160, temp=0.8):
    model.eval(); x = torch.tensor([list(seed.encode("utf-8"))], device=DEV)
    with torch.no_grad():
        for _ in range(n):
            lo = model(x[:, -LB:]); p = F.softmax(lo[0, -1]/temp, -1); x = torch.cat([x, torch.multinomial(p, 1)[None]], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def save():
    json.dump(RESULTS, open("/home/leonardo/projects/LLM/bytebrain/coscientist_metrics.json", "w"), indent=2, ensure_ascii=False)


def main():
    log(f"=== CO-CIENTISTA bytebrain | device {DEV} ===")
    pt_txt, en_txt = enrich()
    # corpora
    all_parts = [open(f, encoding="utf-8").read() for f in sorted(glob.glob(f"{CACHE}/*.txt"))]
    for f in CODE_FILES:
        try: all_parts.append(open(f, errors="ignore").read(2_000_000))
        except Exception: pass
    ALL = to_arr("\n".join(all_parts)); PT = to_arr(pt_txt); EN = to_arr(en_txt)
    CODEtxt = ""
    for f in CODE_FILES:
        try: CODEtxt += open(f, errors="ignore").read(1_500_000)
        except Exception: pass
    CODE = to_arr(CODEtxt)
    log(f"corpora: ALL {len(ALL):,} | PT {len(PT):,} | EN {len(EN):,} | CODE {len(CODE):,} bytes")
    a_tr, a_va = split(ALL); pt_tr, pt_va = split(PT); en_tr, en_va = split(EN); c_tr, c_va = split(CODE)

    # ===== BATERIA A: sweep de zona =====
    log("\n######## BATERIA A — sweep de zona (achar o ponto ideal) ########")
    for tag, cfg in [("A_25M", (512, 8, 8)), ("A_40M", (640, 8, 10)), ("A_57M", (704, 10, 11))]:
        try:
            _, m = train(tag, cfg, a_tr, a_va, 5); RESULTS["baterias"][tag] = m; save()
        except Exception as e:
            log(f"  ERRO {tag}: {e}"); traceback.print_exc()

    # ===== BATERIA B: generalista 40M longo =====
    log("\n######## BATERIA B — GENERALISTA ~40M (25 min, tudo) ########")
    gen_model = None
    try:
        gen_model, m = train("B_generalista", (640, 8, 10), a_tr, a_va, 25); RESULTS["baterias"]["B_generalista"] = m
        RESULTS["amostras"]["generalista"] = {s: gen(gen_model, s, 150) for s in ["The country ", "O Brasil ", "Россия ", "def main():\n    ", "function "]}
        m["bpb_no_PT_holdout"] = round(val_bpb(gen_model, pt_va), 3); save()
        for s, t in RESULTS["amostras"]["generalista"].items(): log(f"  [{s!r}] {t[:90]!r}")
    except Exception as e:
        log(f"  ERRO B: {e}"); traceback.print_exc()

    # ===== BATERIA C: especialista PT + comparacao =====
    log("\n######## BATERIA C — ESPECIALISTA PORTUGUES (12 min, so PT) ########")
    try:
        pt_model, m = train("C_PT_specialist", (640, 8, 10), pt_tr, pt_va, 12); RESULTS["baterias"]["C_PT_specialist"] = m
        RESULTS["amostras"]["PT_specialist"] = {s: gen(pt_model, s, 160) for s in ["O Brasil ", "A história ", "Em ", "O projeto "]}
        for s, t in RESULTS["amostras"]["PT_specialist"].items(): log(f"  [{s!r}] {t[:110]!r}")
        # comparacao: quem escreve PT melhor (bpb no PT held-out)?
        spec_pt = m["val_bpb"]; gen_pt = RESULTS["baterias"].get("B_generalista", {}).get("bpb_no_PT_holdout")
        RESULTS["comparacao_PT"] = {"especialista_PT_bpb": spec_pt, "generalista_no_PT_bpb": gen_pt,
                                    "veredito": "especialista ganha" if (gen_pt and spec_pt < gen_pt) else "generalista competitivo/ganha"}
        log(f"  COMPARACAO PT: especialista {spec_pt} vs generalista-no-PT {gen_pt} -> {RESULTS['comparacao_PT']['veredito']}")
        save()
    except Exception as e:
        log(f"  ERRO C: {e}"); traceback.print_exc()

    # ===== BATERIA D: especialista codigo =====
    log("\n######## BATERIA D — ESPECIALISTA CODIGO (10 min) ########")
    try:
        cm, m = train("D_code_specialist", (640, 8, 10), c_tr, c_va, 10); RESULTS["baterias"]["D_code_specialist"] = m
        RESULTS["amostras"]["code_specialist"] = {s: gen(cm, s, 160) for s in ["def main():\n    ", "function add(", "import "]}
        for s, t in RESULTS["amostras"]["code_specialist"].items(): log(f"  [{s!r}] {t[:110]!r}")
        save()
    except Exception as e:
        log(f"  ERRO D: {e}"); traceback.print_exc()

    log("\n=== CO-CIENTISTA FIM — coscientist_metrics.json salvo ===")
    save()


if __name__ == "__main__":
    main()
