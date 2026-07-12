#!/usr/bin/env python3
"""Extrai PORTUGUÊS LIMPO das sessões do Claude Code (~/.claude/projects/**/*.jsonl) — mensagens do
Leonardo + prosa PT das respostas — filtrando inglês, código, XML e tool-output. Dá PT técnico do nosso
domínio pra enriquecer o corpus do MemByte. Grava sessions_pt.txt."""
import json, glob, re, os, time
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
OUT = "/home/leonardo/projects/LLM/bytebrain/data/sessions_pt.txt"; os.makedirs(os.path.dirname(OUT), exist_ok=True)

ACC = set("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ")
PT_STOP = ["você","não","para","que","com","uma","por","mais","como","está","também","então","porque",
           "isso","aqui","fazer","tem","ser","já","só","agora","gente","cara","vamos","pra","né","coisa"]
CODE = set("{}[]<>=/\\|`#$~^*_")
def is_pt_prose(t):
    t = t.strip()
    if len(t) < 40 or len(t) > 6000: return False
    low = t.lower()
    # descarta XML/tool/observações/paths/erros
    if low.startswith(("<", "{", "[", "http", "/home", "```", "you are", "your previous", "hello memory")): return False
    if "<observ" in low or "tool_use" in low or "function_calls" in low or "system-reminder" in low: return False
    letters = sum(c.isalpha() for c in t)
    if letters < 30: return False
    code_ratio = sum(c in CODE for c in t) / len(t)
    if code_ratio > 0.04: return False                       # muita pontuação de código
    acc_ratio = sum(c in ACC for c in t) / max(letters, 1)
    stop_hits = sum(1 for w in PT_STOP if w in low)
    # PT: tem acento OU várias stopwords PT; e não é predominantemente inglês
    en_hits = sum(1 for w in [" the "," and "," to "," of "," is "," for "," with "," you "] if w in " "+low+" ")
    return (acc_ratio > 0.008 or stop_hits >= 3) and stop_hits >= 2 and en_hits <= stop_hits

def texts_from(obj):
    msg = obj.get("message", obj)
    if not isinstance(msg, dict): return
    c = msg.get("content")
    if isinstance(c, str): yield c
    elif isinstance(c, list):
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                yield b["text"]

files = sorted(glob.glob("/home/leonardo/.claude/projects/**/*.jsonl", recursive=True))
log(f"{len(files)} transcripts")
seen = set(); written = 0; kept = 0
with open(OUT, "w", encoding="utf-8") as out:
    for fp in files:
        try:
            for line in open(fp, encoding="utf-8", errors="ignore"):
                try: o = json.loads(line)
                except: continue
                for t in texts_from(o):
                    # limpa blocos de código inline e normaliza
                    for para in re.split(r"\n\s*\n", t):
                        para = para.strip()
                        if is_pt_prose(para):
                            h = hash(para)
                            if h in seen: continue
                            seen.add(h); out.write(para + "\n\n"); written += len(para) + 2; kept += 1
        except Exception:
            continue
    log(f"salvo {OUT} | {written/1e6:.1f} MB PT, {kept} parágrafos")
log(f"DONE ({time.time()-t0:.0f}s)")
