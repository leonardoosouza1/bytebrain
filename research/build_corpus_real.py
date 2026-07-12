#!/usr/bin/env python3
"""Constrói um corpus REAL rico em texto (Wikipedia PT via streaming) p/ dar ao MemByte priors AMPLOS —
vocabulário enorme (nomes, lugares, termos técnicos, números) que cura o gap dos priors estreitos do
corpus sintético ('carmesim' rabeava). Grava UTF-8 até ~TARGET_MB. Não carrega tudo em RAM (streaming)."""
import sys, time, os
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
from datasets import load_dataset

OUT = "/home/leonardo/projects/LLM/bytebrain/data"; os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(OUT, "pt_corpus.txt")
TARGET_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 900

# tenta configs de Wikipedia PT conhecidas (nomes mudam por snapshot)
CONFIGS = [("wikimedia/wikipedia", "20231101.pt"), ("graelo/wikipedia", "20230901.pt"),
           ("graelo/wikipedia", "20230601.pt")]
ds = None
for repo, cfg in CONFIGS:
    try:
        log(f"tentando {repo} :: {cfg} (streaming)...")
        ds = load_dataset(repo, cfg, split="train", streaming=True)
        _ = next(iter(ds)); log(f"OK: {repo} {cfg}"); break
    except Exception as e:
        log(f"  falhou: {str(e)[:80]}")
if ds is None:
    log("nenhuma config de Wikipedia funcionou; abortando"); sys.exit(1)

written = 0; n = 0; t_last = t0
with open(PATH, "w", encoding="utf-8") as f:
    for ex in ds:
        txt = (ex.get("text") or "").strip()
        if len(txt) < 200: continue   # descarta stubs
        f.write(txt); f.write("\n\n")
        written += len(txt.encode("utf-8")) + 2; n += 1
        if n % 2000 == 0:
            log(f"  {n} artigos, {written/1e6:.0f} MB")
        if written >= TARGET_MB * 1e6: break
log(f"corpus salvo: {PATH} | {written/1e6:.0f} MB, {n} artigos")
log(f"DONE ({time.time()-t0:.0f}s)")
