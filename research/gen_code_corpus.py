#!/usr/bin/env python3
"""Distill CODE + short definitions from Qwen (Leonardo's idea: small atomic
examples; code is byte-friendly/structural). Qwen base completes varied class/
function headers; we save prompt+completion as a corpus to fine-tune ByteBrain. GPU."""
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "data/qwen_code.txt"
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()

@torch.no_grad()
def gen(prompt, n=150):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return prompt + tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def trim_js(t):                          # cut after the class's closing brace at col 0
    i = t.find("\n}")
    return t[:i + 2] if i > 0 else t

CLASSES = [
    ("Conta bancária com depósito e saque", "ContaBancaria"),
    ("Pilha (stack) com push, pop e peek", "Pilha"),
    ("Fila (queue) com enqueue e dequeue", "Fila"),
    ("Retângulo com área e perímetro", "Retangulo"),
    ("Carro com acelerar e frear", "Carro"),
    ("Pessoa com nome e idade", "Pessoa"),
    ("Calculadora com somar e subtrair", "Calculadora"),
    ("Círculo com cálculo de área", "Circulo"),
    ("Contador com incrementar e resetar", "Contador"),
    ("Temperatura com conversão Celsius/Fahrenheit", "Temperatura"),
]
PYF = [
    ("fibonacci(n)", "Retorna o n-ésimo número de Fibonacci"),
    ("fatorial(n)", "Retorna o fatorial de n"),
    ("eh_primo(n)", "Retorna True se n for primo"),
    ("inverter_string(s)", "Retorna a string invertida"),
    ("soma_lista(lista)", "Retorna a soma dos elementos da lista"),
    ("maximo(lista)", "Retorna o maior elemento da lista"),
]
DEFS = ["recursão", "variável", "função", "laço de repetição", "array",
        "objeto", "herança", "condicional", "loop", "string"]

snippets = []
for desc, name in CLASSES:
    snippets.append(trim_js(gen(f"// Classe JavaScript: {desc}\nclass {name} {{", 150)))
for sig, doc in PYF:
    snippets.append(gen(f'def {sig}:\n    """{doc}."""\n', 90))
for term in DEFS:
    snippets.append(gen(f"Em programação, {term} é", 36).split("\n")[0])

with open(OUT, "w") as f:
    f.write("\n\n".join(s.strip() for s in snippets) + "\n")
print(f"wrote {len(snippets)} snippets ({len(CLASSES)} JS classes + {len(PYF)} py funcs + {len(DEFS)} defs) to {OUT}")
print(f"corpus size: {sum(len(s) for s in snippets)} chars")
print("\n----- amostra (1ª classe) -----\n" + snippets[0][:400])
print("\nDONE gen_code_corpus")
