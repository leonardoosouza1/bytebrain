#!/usr/bin/env python3
"""Monta um corpus de BINÁRIOS + CÓDIGO do próprio sistema (ELF executáveis, .so, fontes) pra treinar o
ANALISADOR — um byte-model que aprende estrutura de máquina/código (não texto). Raw bytes, memmap-able."""
import os, sys, glob, time
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
OUT = "/home/leonardo/projects/LLM/bytebrain/data/binary_corpus.bin"
TARGET_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PER_FILE = 256 * 1024   # cap por arquivo p/ diversidade

def gather(patterns, exts=None, is_bin=None):
    files = []
    for pat in patterns:
        files += glob.glob(pat, recursive=True)
    out = []
    for f in files:
        try:
            if not os.path.isfile(f) or os.path.islink(f): continue
            if exts and not any(f.endswith(e) for e in exts): continue
            with open(f, "rb") as fh:
                b = fh.read(PER_FILE)
            if is_bin is not None:
                head = b[:4]
                if is_bin and head[:4] != b"\x7fELF": continue      # só ELF
                if not is_bin and b"\x00" in b[:200]: continue        # código não tem null cedo
            if len(b) > 500: out.append(b)
        except Exception:
            continue
    return out

log("coletando ELF (executáveis + bibliotecas)...")
elf = gather(["/usr/bin/*", "/bin/*", "/usr/lib/x86_64-linux-gnu/*.so*", "/usr/lib/x86_64-linux-gnu/**/*.so*"], is_bin=True)
log(f"  {len(elf)} arquivos ELF")
log("coletando código-fonte (C/py/rs/h)...")
code = gather(["/usr/include/**/*.h", "/usr/include/*.h",
               "/home/leonardo/projects/LLM/**/*.py", "/home/leonardo/projects/LLM/**/*.rs",
               "/home/leonardo/projects/LLM/**/*.c", "/home/leonardo/projects/LLM/**/*.h"],
              exts=(".h", ".py", ".rs", ".c"), is_bin=False)
log(f"  {len(code)} arquivos de código")

# intercala binário e código (~70% binário, 30% código) até TARGET
import itertools
written = 0; bi = 0; ci = 0
with open(OUT, "wb") as out:
    order = []
    for k in range(max(len(elf), len(code)) * 2):
        if k % 3 == 2 and ci < len(code): order.append(("c", ci)); ci += 1
        elif bi < len(elf): order.append(("b", bi)); bi += 1
        elif ci < len(code): order.append(("c", ci)); ci += 1
    for typ, idx in order:
        b = elf[idx] if typ == "b" else code[idx]
        out.write(b); out.write(b"\n"); written += len(b) + 1
        if written >= TARGET_MB * 1e6: break
log(f"corpus salvo: {OUT} | {written/1e6:.0f} MB (ELF+código)")
log(f"DONE ({time.time()-t0:.0f}s)")
