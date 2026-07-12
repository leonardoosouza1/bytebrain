#!/usr/bin/env python3
"""ByteBrain capaz v1 — generate a BROAD multi-teacher knowledge corpus from Qwen.
Facts + code + definitions (Qwen2.5-1.5B) and step-by-step arithmetic (Qwen2.5-Math).
Short atomic examples (distill well). Writes INCREMENTALLY (survives reboot). GPU."""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "data/qwen_broad.txt"
out = open(OUT, "a", buffering=1)                       # line-buffered, append (incremental)

def load(name):
    tok = AutoTokenizer.from_pretrained(f"{MODELS}/{name}")
    m = AutoModelForCausalLM.from_pretrained(f"{MODELS}/{name}", dtype=torch.bfloat16).to(DEV).eval()
    return tok, m

@torch.no_grad()
def gen(tok, m, prompt, n=90):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    o = m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return prompt + tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)

def emit(text):
    out.write(text.strip() + "\n\n"); print("·", text.strip()[:70], flush=True)

PAISES_DO = ["Brasil", "Japão", "Canadá", "México", "Egito", "Chile", "Peru", "Uruguai", "Paraguai", "Equador"]
PAISES_DA = ["França", "Portugal", "Argentina", "Itália", "Espanha", "Alemanha", "Rússia", "China", "Colômbia",
             "Grécia", "Inglaterra", "Índia", "Austrália", "Bolívia", "Venezuela", "Holanda", "Suécia", "Noruega"]
SCIENCE = ["A água é composta pelos elementos", "A fotossíntese é o processo pelo qual as plantas",
    "O Sol é uma estrela que", "A força da gravidade faz com que os objetos", "A célula é a menor unidade",
    "O oxigênio é essencial para a", "Os planetas giram ao redor do", "A velocidade da luz é de aproximadamente",
    "O DNA é a molécula que armazena", "A Terra leva 365 dias para", "O cérebro humano é responsável por",
    "Os átomos são compostos por prótons, nêutrons e", "A energia não pode ser criada nem destruída, apenas",
    "O som se propaga através de", "A evolução das espécies foi descrita por"]
HIST = ["O Brasil foi descoberto em", "A Segunda Guerra Mundial terminou em", "A Revolução Francesa começou em",
    "A independência do Brasil foi proclamada em", "O homem chegou à Lua em", "A queda do Muro de Berlim ocorreu em",
    "A Roma Antiga foi fundada", "A escravidão no Brasil foi abolida em", "A imprensa foi inventada por",
    "A Primeira Guerra Mundial começou em"]
DEFS = ["recursão", "variável", "função", "array", "objeto", "herança", "condicional", "loop", "API",
    "algoritmo", "compilador", "banco de dados", "rede neural", "polimorfismo"]
CODE = [("Conta bancária com depósito e saque", "ContaBancaria"), ("Pilha com push, pop e peek", "Pilha"),
    ("Fila com enqueue e dequeue", "Fila"), ("Retângulo com área e perímetro", "Retangulo"),
    ("Pessoa com nome e idade", "Pessoa"), ("Calculadora com somar e subtrair", "Calculadora"),
    ("Círculo com área", "Circulo"), ("Contador com incrementar e resetar", "Contador")]
PYF = [("fibonacci(n)", "n-ésimo número de Fibonacci"), ("fatorial(n)", "fatorial de n"),
    ("eh_primo(n)", "True se n for primo"), ("inverter_string(s)", "string invertida"),
    ("soma_lista(lista)", "soma dos elementos"), ("maximo(lista)", "maior elemento")]

if "math" not in sys.argv:
    tok, m = load("qwen2.5-1.5b")
    for p in PAISES_DO: emit(gen(tok, m, f"A capital do {p} é a cidade de", 22))
    for p in PAISES_DA: emit(gen(tok, m, f"A capital da {p} é a cidade de", 22))
    for s in SCIENCE:   emit(gen(tok, m, s, 30))
    for h in HIST:      emit(gen(tok, m, h, 26))
    for t in DEFS:      emit(gen(tok, m, f"Em programação, {t} é", 34).split("\n")[0])
    for d, nm in CODE:  emit(gen(tok, m, f"// Classe JavaScript: {d}\nclass {nm} {{", 130))
    for sig, doc in PYF: emit(gen(tok, m, f'def {sig}:\n    """Retorna o {doc}."""\n', 80))
    del m; torch.cuda.empty_cache()

# arithmetic / step-by-step from Qwen-Math
tokM, mM = load("Qwen2.5-Math-1.5B")
ARI = ["7 x 8 =", "12 + 15 =", "100 - 37 =", "9 x 9 =", "144 / 12 =", "25 + 38 =", "6 x 7 =", "200 - 89 =",
       "15 x 4 =", "81 / 9 =", "47 + 56 =", "13 x 3 ="]
for a in ARI: emit(gen(tokM, mM, f"Quanto é {a}", 40))
for a in ["23 * 47", "patos: tenho 3 caixas com 12 ovos cada, quantos ovos no total"]:
    emit(gen(tokM, mM, f"Resolva passo a passo: {a}.", 110))
out.write("=== DONE ===\n"); out.close()
print("DONE gen_broad_corpus", flush=True)
