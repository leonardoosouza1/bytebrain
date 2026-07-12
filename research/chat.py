#!/usr/bin/env python3
"""Chat PT-BR com Qwen3-4B (GGUF) via llama-cpp-python. Mantém o histórico (absorve
contexto) e raciocina. Uso:
  NGL=-1 python research/chat.py --demo     # demo scripted (prova contexto+raciocínio), GPU
  NGL=0  python research/chat.py            # interativo na CPU
"""
import sys, os
from llama_cpp import Llama

MODEL = "/home/leonardo/projects/LLM/llm-lab/models/qwen3-4b-q4km.gguf"
NGL = int(os.environ.get("NGL", "-1"))                 # -1 = todas as camadas na GPU; 0 = CPU
SYS = ("Você é um assistente prestativo que raciocina com cuidado e responde sempre "
       "em português do Brasil, de forma clara e objetiva.")

print(f"carregando Qwen3-4B (n_gpu_layers={NGL})...", flush=True)
llm = Llama(model_path=MODEL, n_gpu_layers=NGL, n_ctx=4096, chat_format="chatml", verbose=False)

def chat(history, user, max_tokens=600):
    history.append({"role": "user", "content": user})
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYS}] + history,
        max_tokens=max_tokens, temperature=0.6, top_p=0.95)
    ans = out["choices"][0]["message"]["content"].strip()
    history.append({"role": "assistant", "content": ans})
    return ans

if "--demo" in sys.argv:
    h = []
    turns = [
        "Oi! Meu nome é Leonardo e eu programo em Rust com foco em IA na GPU.",
        "Tenho 3 caixas com 12 ovos cada e quebrei 5 ovos. Quantos sobraram? Pense passo a passo.",
        "Qual linguagem de programação eu disse que uso?",          # testa MEMÓRIA de contexto
    ]
    for u in turns:
        print(f"\n🧑 {u}\n🤖 {chat(h, u)}", flush=True)
    print("\n[demo ok]")
else:
    h = []
    print("Chat PT-BR (Qwen3-4B). Digite e Enter; Ctrl-D pra sair.")
    while True:
        try:
            u = input("\n🧑 ").strip()
        except EOFError:
            break
        if u:
            print("🤖 " + chat(h, u), flush=True)
