#!/usr/bin/env python3
"""Transfer step 2 — build the ByteBrain training corpus from Qwen's distilled facts.
Augments each fact into several phrasings (statement / reversed / Q&A) so the small
byte model can form the association, repeats them heavily (memorization signal), and
mixes in PT-Wikipedia so it doesn't forget fluency. Output = raw byte corpus."""
import re, os

FACTS = "data/qwen_facts.txt"
WIKI = "data/pt_overnight.txt"
OUT = "data/distill.txt"

cap = re.compile(r"A capital (d[ao]) (.+?) é a cidade de ([^,.\n]+)")
lines = [l.strip() for l in open(FACTS, encoding="utf-8") if l.strip()]
aug = []
n_cap = 0
for l in lines:
    m = cap.search(l)
    if m and "_" not in m.group(3) and len(m.group(3).strip()) >= 3:
        prep, country, capital = m.group(1), m.group(2).strip(), m.group(3).strip()
        aug += [
            f"A capital {prep} {country} é {capital}.",
            f"{capital} é a capital {prep} {country}.",
            f"Qual é a capital {prep} {country}? A capital é {capital}.",
            f"Pergunta: capital {prep} {country}. Resposta: {capital}.",
            l,                                          # Qwen's full original sentence
        ]
        n_cap += 1
    else:
        aug += [l, l]                                   # science/other facts: keep (x2)

fact_block = ("\n".join(aug) + "\n").encode("utf-8")
TARGET_FACTS = 1_200_000                                # ~1.2MB of (repeated) facts
reps = max(1, TARGET_FACTS // len(fact_block))
facts_bytes = fact_block * reps

wiki = open(WIKI, "rb").read(1_000_000)                 # 1MB PT to retain fluency

with open(OUT, "wb") as f:
    f.write(facts_bytes + b"\n" + wiki)

print(f"facts parsed: {n_cap} capitals + {len(lines)-n_cap} other")
print(f"augmented lines: {len(aug)}  | fact block {len(fact_block)}B × {reps} reps = {len(facts_bytes)}B")
print(f"+ {len(wiki)}B wiki = {os.path.getsize(OUT)}B total → {OUT}")
print("sample augmented facts:")
for a in aug[:6]:
    print("  ", a)
