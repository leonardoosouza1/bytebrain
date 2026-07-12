#!/usr/bin/env python3
"""BIT-LENS — olhar a frase por BIT, não por byte (a visão do Leonardo). Mostra a matriz de bits (caractere
× 8 bits, de cima=bit7 pra baixo=bit0), qual bit codifica o quê (estrutura vs conteúdo), e a relação entre
caracteres (A vs I, maiúscula↔minúscula, vogal↔consoante). Puro CPU."""
import numpy as np, collections
FRASE = "A Inteligencia Artificial pensa"
def bits(b): return [(b >> i) & 1 for i in range(7, -1, -1)]   # bit7 .. bit0

print(f"FRASE: {FRASE!r}\n")
print("MATRIZ DE BITS (colunas = caracteres; linhas = bit7 no topo → bit0 embaixo):")
print("       " + "".join(c if c != ' ' else '_' for c in FRASE))
for i, bit in enumerate(range(7, -1, -1)):
    row = "".join("█" if (ord(c) >> bit) & 1 else "·" for c in FRASE)
    rolelabel = {7:"UTF8/ASCII", 6:"é-letra?", 5:"CASO (maiúsc↔minúsc)", 4:"grupo", 3:"", 2:"", 1:"", 0:"posição/conteúdo"}[bit]
    print(f"  bit{bit} {row}   {rolelabel}")

print("\n--- POR QUE cada bit (ASCII) ---")
for ch in "A I a i E O . _".split(" ") if False else ["A","I","a","i","E","O"," ",".","1"]:
    b = ord(ch); bs = "".join(map(str, bits(b)))
    print(f"  {ch!r:4} = {b:3}  {bs}   bit7={bs[0]} bit6(letra)={bs[1]} bit5(caso)={bs[2]} bits4-0(qual)={bs[3:]}")

print("\n--- RELAÇÃO A vs I (mesma 'estrutura', muda o 'conteúdo') ---")
A, I = ord("A"), ord("I")
print(f"  A={A:08b}  I={I:08b}  XOR={A^I:08b}  → diferem só nos bits baixos (posição no alfabeto)")
print(f"  A↔a XOR = {ord('A')^ord('a'):08b} = 0x20 = SÓ o bit5 (o caso). Trocar caixa = flipar 1 bit.")

print("\n--- ESTRUTURA vs CONTEÚDO num corpus real (variância de cada bit) ---")
data = np.frombuffer(open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt","rb").read(2_000_000), dtype=np.uint8)
for bit in range(7, -1, -1):
    col = (data >> bit) & 1
    p = col.mean(); ent = 0.0 if p in (0,1) else -(p*np.log2(p)+(1-p)*np.log2(1-p))
    tipo = "ESTRUTURAL (quase fixo)" if ent < 0.4 else ("misto" if ent < 0.95 else "CONTEÚDO (varia muito)")
    print(f"  bit{bit}: p(1)={p:.2f} entropia={ent:.2f}  {tipo}")
print("\n→ bits ALTOS = estrutura (é-letra, caso, grupo); bits BAIXOS = conteúdo (qual caractere).")
print("  'Tabelar por bit' separa ESTRUTURA de CONTEÚDO — é onde a coesão mora.")
