#!/usr/bin/env python3
"""VER O CHAT NASCER: mini-SFT no ByteBrain 40M (ckpt_broad). ANTES: o modelo continua texto
wiki-style. Treino: ~400 passos em diálogos USER/BOT (fatos que ele JÁ sabe da destilação +
small talk) → DEPOIS: ele vira o PERSONAGEM assistente (responde e para). É a fase 2 (SFT) do
pipeline de treinamento, demonstrada em casa. CPU (GPU ocupada). Salva ckpt_chat/."""
import os, sys, time
import torch, torch.nn.functional as Fn
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from train_graph import GraphByteGPT
try: from train_graph import set_act_quant
except ImportError: set_act_quant = lambda *_: None
t0 = time.time(); torch.manual_seed(0)
DEV = "cpu"; STEPS = 400; BATCH = 8; CTX = 256; LR = 2e-4

QA = [
 ("Qual é a capital do Brasil?", "A capital do Brasil é Brasília."),
 ("Qual é a capital da França?", "A capital da França é Paris."),
 ("Qual é a capital da Itália?", "A capital da Itália é Roma."),
 ("Qual é a capital do Japão?", "A capital do Japão é Tóquio."),
 ("Qual é a capital de Portugal?", "A capital de Portugal é Lisboa."),
 ("Qual é a velocidade da luz?", "A velocidade da luz é de cerca de 300.000 km por segundo."),
 ("O que é uma célula?", "A célula é a unidade básica da vida."),
 ("O que é o DNA?", "O DNA é a molécula que guarda as informações genéticas."),
 ("O que é fotossíntese?", "Fotossíntese é o processo em que as plantas convertem luz em energia."),
 ("Para que serve o coração?", "O coração bombeia o sangue para todo o corpo."),
 ("Oi, tudo bem?", "Oi! Tudo bem sim, e com você?"),
 ("Bom dia!", "Bom dia! Como posso ajudar?"),
 ("Obrigado!", "De nada! Precisando, é só chamar."),
 ("Quem é você?", "Eu sou o IARA, um assistente pequeno que roda no seu computador."),
 ("Você fala português?", "Sim! Eu falo português."),
 ("O que você sabe fazer?", "Eu respondo perguntas sobre fatos, ciência e programação."),
 ("Qual é o maior planeta?", "O maior planeta do sistema solar é Júpiter."),
 ("Quantos planetas existem?", "O sistema solar tem oito planetas."),
 ("O que é a gravidade?", "A gravidade é a força que atrai os corpos entre si."),
 ("O que é um vírus?", "Um vírus é um agente microscópico que infecta células."),
]
VARI = ["{}", "Me diga: {}", "Uma pergunta: {}", "{} Responda rápido."]
corpus = ""
for _ in range(3):
    for v in VARI:
        for q, a in QA:
            corpus += f"USER: {v.format(q)}\nBOT: {a}\n\n"
data = torch.tensor(list(corpus.encode()), dtype=torch.long)
print(f"corpus SFT: {len(corpus)} chars, {len(QA)} QAs × {len(VARI)} formas × 3", flush=True)

c = torch.load("/home/leonardo/projects/LLM/bytebrain/ckpt_broad/ckpt_best.pt", map_location="cpu", weights_only=False)
cf = c["config"]; set_act_quant(cf.get("quant_bits", 0))
m = GraphByteGPT(cf["dim"], cf["layers"], cf["heads"], cf["ctx"], topk=cf.get("topk", 0),
                 mem=cf.get("mem", 0), topic=cf.get("topic", 0))
m.load_state_dict(c["model"])
print(f"ByteBrain carregado ({sum(p.numel() for p in m.parameters())/1e6:.0f}M)", flush=True)

@torch.no_grad()
def sample(prompt, n=60):
    m.eval(); ids = list(prompt.encode())
    for _ in range(n):
        x = torch.tensor([ids[-cf["ctx"]:]], dtype=torch.long)
        out = m(x); lg = out[0] if isinstance(out, tuple) else out
        nxt = int(lg[0, -1].argmax()); ids.append(nxt)
        if bytes(ids[-2:]) == b"\n\n": break
    return bytes(ids[len(prompt.encode()):]).decode("utf-8", errors="ignore")

TESTS = ["USER: Qual é a capital da Itália?\nBOT:", "USER: Oi, tudo bem?\nBOT:",
         "USER: O que é o DNA?\nBOT:", "USER: Quem é você?\nBOT:",
         "USER: Qual é o maior planeta?\nBOT:"]
print("\n===== ANTES do SFT (modelo continua texto solto) =====", flush=True)
before = {t: sample(t) for t in TESTS}
for t in TESTS: print(f"  {t.splitlines()[0][:44]} → {before[t][:70]!r}", flush=True)

opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
m.train()
for step in range(STEPS):
    ix = torch.randint(0, len(data) - CTX - 1, (BATCH,))
    xb = torch.stack([data[i:i+CTX] for i in ix])
    yb = torch.stack([data[i+1:i+CTX+1] for i in ix])
    out = m(xb); lg = out[0] if isinstance(out, tuple) else out
    loss = Fn.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"  passo {step:>3}: loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

print("\n===== DEPOIS do SFT (o personagem assistente nasceu?) =====", flush=True)
after = {t: sample(t) for t in TESTS}
for t in TESTS: print(f"  {t.splitlines()[0][:44]} → {after[t][:70]!r}", flush=True)

os.makedirs("/home/leonardo/projects/LLM/bytebrain/ckpt_chat", exist_ok=True)
torch.save({"model": m.state_dict(), "config": cf},
           "/home/leonardo/projects/LLM/bytebrain/ckpt_chat/ckpt_best.pt")
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## SFT chat no ByteBrain ({int(time.time()-t0)}s, {STEPS} passos CPU)\n")
    for t in TESTS:
        f.write(f"- {t.splitlines()[0]} | ANTES {before[t][:50]!r} | DEPOIS {after[t][:50]!r}\n")
print(f"\nDONE sft_chat → ckpt_chat/ ({time.time()-t0:.0f}s)", flush=True)
