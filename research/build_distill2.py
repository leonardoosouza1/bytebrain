#!/usr/bin/env python3
"""Build a COMBINED distillation corpus: Qwen facts + Qwen code/defs + wiki.
Goal: a ByteBrain that knows facts AND can produce code patterns. Short atomic
examples, repeated (Leonardo's idea). Output raw bytes for train_graph."""
import re, os

cap = re.compile(r"A capital (d[ao]) (.+?) é a cidade de ([^,.\n]+)")

# --- facts (augmented) ---
aug = []
for l in (x.strip() for x in open("data/qwen_facts.txt", encoding="utf-8") if x.strip()):
    m = cap.search(l)
    if m and "_" not in m.group(3) and len(m.group(3).strip()) >= 3:
        prep, c, cap_ = m.group(1), m.group(2).strip(), m.group(3).strip()
        aug += [f"A capital {prep} {c} é {cap_}.", f"{cap_} é a capital {prep} {c}.",
                f"Qual é a capital {prep} {c}? A capital é {cap_}.", l]
    else:
        aug.append(l)
fact_block = ("\n".join(aug) + "\n").encode("utf-8")

# --- code (snippets as Qwen wrote them) ---
code_txt = open("data/qwen_code.txt", encoding="utf-8").read()
code_block = (code_txt.strip() + "\n\n").encode("utf-8")

wiki = open("data/pt_overnight.txt", "rb").read(800_000)

facts_rep = fact_block * max(1, 600_000 // len(fact_block))
code_rep = code_block * max(1, 600_000 // len(code_block))

with open("data/distill2.txt", "wb") as f:
    f.write(facts_rep + b"\n" + code_rep + b"\n" + wiki)

print(f"facts: {len(fact_block)}B ×{600_000//len(fact_block)} = {len(facts_rep)}B")
print(f"code : {len(code_block)}B ×{600_000//len(code_block)} = {len(code_rep)}B")
print(f"wiki : {len(wiki)}B | total {os.path.getsize('data/distill2.txt')}B → data/distill2.txt")
