#!/usr/bin/env python3
"""BIT-PREDICT — como o byte-model constrói COESÃO? Mede, por BIT, quão bem ele prevê o próximo byte.
Hipótese: ele CRAVA os bits ESTRUTURAIS (é-letra/caso/fronteira = coesão) e CHUTA os de CONTEÚDO. Roda o
MemByte na CPU sobre um texto e reporta, pra cada bit 7..0, a massa de probabilidade no valor correto. CPU."""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cpu"
ck = torch.load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt", map_location=DEV, weights_only=False)
c = ck["config"]; M = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
M.load_state_dict(ck["model"])

# máscaras: quais dos 256 bytes têm bit k = 1
vals = np.arange(256)
bitmask = {k: torch.tensor((vals >> k) & 1, dtype=torch.float32) for k in range(8)}

texto = "A inteligencia artificial pensa em bytes, e cada caractere e uma sequencia de oito bits que carrega estrutura e conteudo juntos."
ids = list(texto.encode())
x = torch.tensor([ids], device=DEV)
CTX = c["ctx"]
acc_bit = {k: [] for k in range(8)}   # P(bit correto)
argmax_bit = {k: [] for k in range(8)}
with torch.no_grad():
    logits = M(x[:, :CTX])[0]          # [L,256]
    probs = F.softmax(logits.float(), -1)
for pos in range(len(ids) - 1):
    p = probs[pos]                      # distribuição do próximo byte
    true = ids[pos + 1]; pred = int(p.argmax())
    for k in range(8):
        tb = (true >> k) & 1
        p1 = float((p * bitmask[k]).sum())      # P(bit k = 1)
        acc_bit[k].append(p1 if tb == 1 else 1 - p1)   # massa no valor correto
        argmax_bit[k].append(int(((pred >> k) & 1) == tb))

print(f"texto ({len(ids)} bytes): {texto[:60]}...\n")
print("BIT  papel                 P(bit correto)   acerto-argmax")
role = {7:"UTF8/ASCII (estrut)",6:"é-letra? (estrut)",5:"CASO (estrut)",4:"grupo",3:"conteúdo",2:"conteúdo",1:"conteúdo",0:"conteúdo"}
for k in range(7, -1, -1):
    pc = np.mean(acc_bit[k]); am = np.mean(argmax_bit[k])
    bar = "█" * round(pc * 20)
    print(f"bit{k}  {role[k]:20} {pc:.2f} {bar:<20}  {am:.2f}")
print("\n→ se os bits ALTOS têm P alto (o modelo CRAVA) e os BAIXOS caem (ele CHUTA), a COESÃO mora na estrutura:")
print("  o modelo sabe 'vem uma letra minúscula' (estrutura) muito melhor do que 'qual letra' (conteúdo).")
