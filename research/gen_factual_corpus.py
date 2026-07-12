#!/usr/bin/env python3
"""Transfer step 1 — DISTILL Qwen's factual knowledge into a byte corpus.
Qwen completes high-confidence factual prompts (capitals, science) greedily; we
keep its completion + the confidence (mean top-prob). Output = a PT corpus of true
statements (in Qwen's words) that ByteBrain will then train on, as BYTES. We log
confidence so we can filter out low-confidence (likely-wrong) facts. CPU."""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
OUT = "data/qwen_facts.txt"
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()

PAISES = ["Brasil", "França", "Portugal", "Argentina", "Japão", "Itália", "Espanha",
          "Alemanha", "Rússia", "China", "Egito", "México", "Canadá", "Chile",
          "Peru", "Colômbia", "Grécia", "Inglaterra", "Índia", "Austrália"]
SEEDS = [f"A capital do {p} é a cidade de" if p in ("Brasil","Japão","Canadá","México","Egito","Chile","Peru")
         else f"A capital da {p} é a cidade de" for p in PAISES] + [
    "A água é composta pelos elementos químicos",
    "A fotossíntese é o processo pelo qual as plantas convertem luz solar em",
    "O Sol é uma estrela que fornece luz e",
    "A força da gravidade faz com que os objetos",
    "O coração humano tem a função de bombear",
    "Os planetas do sistema solar giram ao redor do",
    "A célula é a menor unidade de um ser",
    "O oxigênio é essencial para a respiração dos seres",
    "A Terra leva aproximadamente 365 dias para dar uma volta ao redor do",
    "A velocidade da luz no vácuo é de aproximadamente",
    "O Brasil foi colonizado pelos",
    "A língua oficial do Brasil é o",
    "O maior oceano do planeta é o oceano",
    "O ferro é um metal que enferruja quando exposto ao",
    "O ciclo da água inclui evaporação, condensação e",
]

@torch.no_grad()
def complete(prompt, n=22):
    ids = tok(prompt, return_tensors="pt").input_ids
    confs = []
    for _ in range(n):
        p = torch.softmax(m(ids).logits[0, -1], -1)
        top = torch.topk(p, 1); c = int(top.indices[0])
        if c == tok.eos_token_id: break
        confs.append(float(top.values[0]))
        ids = torch.cat([ids, torch.tensor([[c]])], 1)
    txt = tok.decode(ids[0, :], skip_special_tokens=True)
    # cut at first sentence end after the prompt
    body = txt[len(prompt):]
    for end in [". ", ".\n", "\n"]:
        if end in body: body = body[:body.index(end) + 1]; break
    return prompt + body, sum(confs) / max(1, len(confs))

rows = []
for s in SEEDS:
    txt, conf = complete(s)
    txt = " ".join(txt.split())
    rows.append((conf, txt))
    print(f"[{conf:.2f}] {txt}", flush=True)

rows.sort(reverse=True)
with open(OUT, "w") as f:
    for conf, txt in rows:
        if conf >= 0.45:                       # keep confident (likely-true) facts
            f.write(txt.strip() + "\n")
kept = sum(1 for c, _ in rows if c >= 0.45)
print(f"\nwrote {kept}/{len(rows)} facts (conf>=0.45) to {OUT}")
print("DONE gen_factual_corpus")
