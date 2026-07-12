#!/usr/bin/env python3
"""Cria CORPUS DE INSTRUÇÃO (P:/R:) barato pra o byte-model virar CÓRTEX (responder, não só continuar):
- Wikipedia: 1ª frase "X é ..." vira 'P: O que é X?\\nR: X é ...' (definição factual).
- mistura formulações. É o começo do córtex byte; dados de raciocínio do Qwen entram depois."""
import re, os, time
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
WIKI = "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt"
OUT = "/home/leonardo/projects/LLM/bytebrain/data/instruct_corpus.txt"
TARGET_MB = 80

SUBJ = re.compile(r"^(.{2,40}?)\s+(é|foi|são|era|eram|consiste|refere-se)\s", re.I)
data = open(WIKI, "r", encoding="utf-8", errors="ignore")
written = 0; n = 0
with open(OUT, "w", encoding="utf-8") as out:
    buf = ""
    for line in data:
        if line.strip() == "":
            art = buf.strip(); buf = ""
            if len(art) < 60: continue
            frase = art.split(". ")[0].strip()
            if not (20 < len(frase) < 320): continue
            m = SUBJ.match(frase)
            if not m: continue
            subj = m.group(1).strip(" ,\"'()")
            if not subj or len(subj.split()) > 6 or subj[0].islower(): continue
            resp = frase if frase.endswith(".") else frase + "."
            for q in (f"O que é {subj}?", f"O que significa {subj}?", f"Defina {subj}."):
                out.write(f"P: {q}\nR: {resp}\n\n"); written += len(q) + len(resp) + 8
            n += 1
            if written >= TARGET_MB * 1e6: break
        else:
            buf += line
log(f"instruct salvo: {OUT} | {written/1e6:.0f} MB, {n} definições → Q&A")
log(f"DONE ({time.time()-t0:.0f}s)")
