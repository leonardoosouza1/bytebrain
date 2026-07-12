#!/usr/bin/env python3
"""ByteBrain capaz v1 — build the training corpus from the broad Qwen corpus.
Augment capital-facts, keep code/science/history/defs/math, repeat (atomic),
mix CLEAN pt_big Wikipedia (NOT pt_overnight=project-notes). Raw bytes out."""
import re, os

cap = re.compile(r"A capital (d[ao]) (.+?) é a cidade de ([^,.\n]+)")
snips = [s.strip() for s in open("data/qwen_broad.txt", encoding="utf-8").read().split("\n\n")
         if s.strip() and "=== DONE ===" not in s]

facts, code = [], []
for s in snips:
    m = cap.search(s)
    if m and "_" not in m.group(3) and len(m.group(3).strip()) >= 3:
        prep, c, cp = m.group(1), m.group(2).strip(), m.group(3).strip()
        facts += [f"A capital {prep} {c} é {cp}.", f"{cp} é a capital {prep} {c}.",
                  f"Qual é a capital {prep} {c}? A capital é {cp}.", s]
    elif "class " in s or "def " in s:
        code.append(s)
    else:
        facts += [s, s]

fact_block = ("\n".join(facts) + "\n").encode("utf-8")
code_block = ("\n\n".join(code) + "\n\n").encode("utf-8")
wiki = open("data/pt_big.txt", "rb").read(800_000)          # CLEAN Wikipedia

fr = fact_block * max(1, 800_000 // len(fact_block))
cr = code_block * max(1, 400_000 // max(1, len(code_block)))
with open("data/distill_broad.txt", "wb") as f:
    f.write(fr + b"\n" + cr + b"\n" + wiki)

print(f"facts: {len(facts)} lines, block {len(fact_block)}B ×{800_000//len(fact_block)} = {len(fr)}B")
print(f"code : {len(code)} snippets, block {len(code_block)}B ×{400_000//max(1,len(code_block))} = {len(cr)}B")
print(f"wiki : {len(wiki)}B (clean) | total {os.path.getsize('data/distill_broad.txt')}B → data/distill_broad.txt")
