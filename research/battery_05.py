#!/usr/bin/env python3
"""BATERIA 5: T12+T15 — BYTES UNIVERSAIS: jogar texto/código/IMAGEM/ÁUDIO (bytes crus) no
NOSSO ByteBrain 40M (treinado só em texto PT) e ver se territórios de MODALIDADE se formam
nos neurônios. T10 — CROSS-MODEL: os territórios replicam no Qwen2.5-1.5B (vs 4B)?
Dump battery_05.json + journal."""
import json, sys, time, wave
import numpy as np, torch
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
t0 = time.time()

# ================= T12+T15: ByteBrain multimodal por bytes =================
from train_graph import GraphByteGPT
try: from train_graph import set_act_quant
except ImportError: set_act_quant = lambda *_: None
c = torch.load("/home/leonardo/projects/LLM/bytebrain/ckpt_broad/ckpt_best.pt", map_location="cpu", weights_only=False)
cf = c["config"]; set_act_quant(cf.get("quant_bits", 0))
bm = GraphByteGPT(cf["dim"], cf["layers"], cf["heads"], cf["ctx"], topk=cf.get("topk", 0),
                  mem=cf.get("mem", 0), topic=cf.get("topic", 0)).eval()
bm.load_state_dict(c["model"])
NBL = len(bm.blocks); BDIM = cf["dim"] * 4; BNEUR = NBL * BDIM
CTX = min(384, cf["ctx"] - 2)
print(f"ByteBrain {sum(p.numel() for p in bm.parameters())/1e6:.0f}M | {NBL}L×{BDIM} = {BNEUR} neurônios | ctx {cf['ctx']} ({time.time()-t0:.0f}s)", flush=True)

bcap = [None] * NBL
def mkb(i):
    def h(m, inp): bcap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
for i in range(NBL): bm.blocks[i].mlp[3].register_forward_pre_hook(mkb(i))
@torch.no_grad()
def bfp(data: bytes):
    ids = torch.tensor([list(data[:CTX])], dtype=torch.long)
    bm(ids)
    return torch.stack(bcap).reshape(-1).numpy()

def chunks(path, n=4, sz=CTX):
    raw = open(path, "rb").read()
    if len(raw) < sz * 2: return [raw[:sz]]
    step = max(1, (len(raw) - sz) // n)
    return [raw[i * step:i * step + sz] for i in range(n)]

TXT = ["O Brasil é o maior país da América do Sul e sua capital é Brasília. " * 4,
       "A fotossíntese é o processo pelo qual as plantas convertem luz solar em energia química. " * 3,
       "A história do Brasil começa com a chegada dos portugueses em 1500. " * 4,
       "O coração humano bombeia sangue para todo o corpo através das artérias. " * 3,
       "A música brasileira é conhecida mundialmente pelo samba e pela bossa nova. " * 3,
       "Machado de Assis é considerado o maior escritor da literatura brasileira. " * 3]
CODE = ["def fatorial(n):\n    if n == 0:\n        return 1\n    return n * fatorial(n-1)\n\n" * 3,
        "for i in range(100):\n    if i % 2 == 0:\n        print(i)\n" * 4,
        "class Animal:\n    def __init__(self, nome):\n        self.nome = nome\n" * 3,
        "SELECT nome, idade FROM usuarios WHERE idade > 18 ORDER BY nome;\n" * 5,
        "const soma = (a, b) => a + b;\nconst dobro = x => x * 2;\n" * 5]
IMGS = ["/home/leonardo/projects/LLM/Universe/image.png", "/home/leonardo/projects/LLM/jarvis/image.png",
        "/home/leonardo/projects/LLM/make-shorts-video/validation_check.jpg"]
AUD = "/home/leonardo/projects/LLM/ONDA/mix.wav"

probes = [(t.encode()[:CTX], "texto") for t in TXT] + [(t.encode()[:CTX], "código") for t in CODE]
for p in IMGS:
    try:
        for ch in chunks(p, 4): probes.append((ch, "imagem"))
    except Exception as e: print("skip img", p, e)
try:
    for ch in chunks(AUD, 8): probes.append((ch, "áudio"))
except Exception as e: print("skip aud", e)
rng0 = np.random.default_rng(7)
for _ in range(5): probes.append((bytes(rng0.integers(0, 256, CTX).tolist()), "aleatório"))
srcs = [t for _, t in probes]; P = len(probes)
print(f"probes byte: {dict((t, srcs.count(t)) for t in sorted(set(srcs)))}", flush=True)

F = np.zeros((BNEUR, P), np.float32)
for j, (d, _) in enumerate(probes): F[:, j] = bfp(d)
TYPES = sorted(set(srcs))
tmean = np.stack([F[:, [j for j in range(P) if srcs[j] == t]].mean(1) for t in TYPES])
home = tmean.argmax(0); mx, mn = tmean.max(0), tmean.mean(0)
sel = (mx - mn) / (mx + mn + 1e-9); GM = tmean.mean(0)
SPEC = (GM > np.median(GM) * 0.05) & (sel > 0.30)
terr = {TYPES[ti]: int((SPEC & (home == ti)).sum()) for ti in range(len(TYPES))}
print(f"T12/T15 — territórios de MODALIDADE no ByteBrain (40M, treinado SÓ texto PT): {terr}", flush=True)
# roteador de modalidade leave-one-out simplificado: metade treino/metade teste
ok, tot = 0, 0
for t in TYPES:
    idx = [j for j in range(P) if srcs[j] == t]
    if len(idx) < 2: continue
    half = len(idx) // 2
    tm2 = np.stack([F[:, [j for j in range(P) if srcs[j] == u and j not in idx[half:]]].mean(1) for u in TYPES])
    h2 = tm2.argmax(0); mx2, mn2 = tm2.max(0), tm2.mean(0)
    s2 = (mx2 - mn2) / (mx2 + mn2 + 1e-9)
    S2 = (tm2.mean(0) > np.median(tm2.mean(0)) * 0.05) & (s2 > 0.30)
    spec2 = {u: (S2 & (h2 == ui)) for ui, u in enumerate(TYPES)}
    base2 = {u: float(tm2[ui, spec2[u]].mean()) if spec2[u].any() else 1.0 for ui, u in enumerate(TYPES)}
    for j in idx[half:]:
        sc = [F[spec2[u], j].mean() / (base2[u] + 1e-9) if spec2[u].any() else 0 for u in TYPES]
        ok += (TYPES[int(np.argmax(sc))] == t); tot += 1
bacc = ok / max(1, tot)
print(f"T12/T15 — roteador de modalidade por neurônios: {ok}/{tot} = {bacc*100:.0f}%", flush=True)
del bm, F

# ================= T10: cross-model (Qwen2.5-1.5B) =================
from transformers import AutoModelForCausalLM, AutoTokenizer
M15 = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
tok = AutoTokenizer.from_pretrained(M15)
qm = AutoModelForCausalLM.from_pretrained(M15, dtype=torch.float16).to("cuda").eval()
ql = qm.model.layers; QNL = len(ql); QIN = qm.config.intermediate_size; QNEUR = QNL * QIN
print(f"\nQwen2.5-1.5B carregado ({time.time()-t0:.0f}s) | {QNL}L×{QIN}", flush=True)
qcap = [None] * QNL
def mkq(i):
    def h(m, inp): qcap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
for i in range(QNL): ql[i].mlp.down_proj.register_forward_pre_hook(mkq(i))
def qfp(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=80).input_ids.to("cuda")
    with torch.no_grad(): qm(ids)
    return torch.stack(qcap).reshape(-1).float().cpu().numpy()

QTRAIN = {
 "português": ["O Brasil é o maior país da América do Sul.", "A fotossíntese ocorre nas plantas.",
   "A capital da França é Paris.", "O coração bombeia sangue.", "A gravidade mantém os planetas.",
   "As células são a unidade da vida.", "A economia cresceu este ano.", "O rio corre para o mar."],
 "código": ["def soma(a, b):\n    return a + b", "for i in range(10):\n    print(i)", "class A:\n    pass",
   "import numpy as np", "SELECT * FROM users;", "const f = a => a * 2;", "while n > 0:\n    n -= 1",
   "x = [i for i in range(5)]"],
 "matemática": ["2 + 2 = 4", "a² + b² = c²", "3 * 7 = 21", "√169 = 13", "f(x) = 3x + 2", "5! = 120",
   "12 / 4 = 3", "2^8 = 256"],
 "inglês": ["The cat sat on the mat.", "The economy grew last year.", "Water freezes cold.",
   "Birds migrate in winter.", "She teaches at school.", "The computer runs fast.",
   "Dogs love long walks.", "Rain falls in spring."],
}
QTEST = {
 "português": ["A Lua influencia as marés.", "Napoleão foi imperador.", "O Sol é uma estrela.",
   "As bactérias são microscópicas.", "A independência foi em 1822.", "O petróleo é combustível."],
 "código": ["def fatorial(n):\n    return 1 if n==0 else n*fatorial(n-1)", "arr = [x*x for x in range(5)]",
   "git push origin main", "df.groupby('col').sum()", "public int add(int a){return a;}", "let x = vec![1,2];"],
 "matemática": ["7 * 8 = 56", "100 - 37 = 63", "derivada de x³ é 3x²", "2^10 = 1024", "π r²", "média de 4 e 6 é 5"],
 "inglês": ["The ocean covers the planet.", "Electric cars reduce pollution.", "Ancient Rome was vast.",
   "Vaccines protect health.", "The river flows east.", "Music brings people joy."],
}
QT = list(QTRAIN)
qtm = np.stack([np.mean([qfp(s) for s in QTRAIN[t]], 0) for t in QT])
qh = qtm.argmax(0); qmx, qmn = qtm.max(0), qtm.mean(0)
qsel = (qmx - qmn) / (qmx + qmn + 1e-9); qGM = qtm.mean(0)
qSPEC = (qGM > np.median(qGM) * 0.05) & (qsel > 0.30)
qterr = {QT[ti]: int((qSPEC & (qh == ti)).sum()) for ti in range(len(QT))}
qspec = {ti: (qSPEC & (qh == ti)) for ti in range(len(QT))}
qbase = {ti: float(qtm[ti, qspec[ti]].mean()) if qspec[ti].any() else 1.0 for ti in range(len(QT))}
qok, qtot = 0, 0
det = {}
for t in QTEST:
    ti_true = QT.index(t); hits = 0
    for s in QTEST[t]:
        v = qfp(s)
        sc = [v[qspec[ti]].mean() / (qbase[ti] + 1e-9) if qspec[ti].any() else 0 for ti in range(len(QT))]
        p = int(np.argmax(sc)); qok += (p == ti_true); hits += (p == ti_true); qtot += 1
    det[t] = f"{hits}/{len(QTEST[t])}"
qacc = qok / qtot
print(f"T10 — territórios no 1.5B: {qterr} ({round(float(qSPEC.mean())*100,1)}% spec)", flush=True)
print(f"T10 — roteador neural no 1.5B: {qok}/{qtot} = {qacc*100:.0f}% (4B foi 90%) | {det}", flush=True)

R = {"T12_T15_bytebrain": {"territorios": terr, "router_modalidade_acc": round(bacc, 3), "n_neurons": BNEUR},
     "T10_qwen15b": {"territorios": qterr, "router_acc": round(qacc, 3), "det": det, "n_neurons": QNEUR,
                     "ref_4b": {"router_acc": 0.90}}}
json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/battery_05.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Lote 5 — T12/T15 bytes universais + T10 cross-model ({int(time.time()-t0)}s)\n")
    f.write(f"- ByteBrain 40M territórios de modalidade: {terr} | roteador {bacc*100:.0f}%\n")
    f.write(f"- Qwen1.5B territórios: {qterr} | roteador {qacc*100:.0f}% (4B: 90%)\n")
print(f"\nDONE battery_05 ({time.time()-t0:.0f}s)", flush=True)
