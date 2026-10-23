# usage: addr_rank.py CKPT SPLITS N OUT.json
"""score every unit's block-0 lines under the addrseg ckpt (teacher forced, CPU); rank of the true unit."""
import json, random, sys, collections
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/pycode/synth")
import torch, torch.nn.functional as F
torch.set_num_threads(3)
import units as U, synth2
from synth2 import UNITS
ck = torch.load(sys.argv[1], map_location="cpu")
split = json.load(open(f"{synth2.D}/splits.json"))
unit = U.make_unit("addrseg", split)
net = U.AddrGPT(unit.vocab, 256, 4, 4, U.SHAPE["addrseg"]["ctx"], "addrseg"); net.load_state_dict(ck["model"]); net.eval()
cands = sorted(u for u in UNITS)          # held units included: their lines exist (test_num names them)
res = {}
for sp in sys.argv[2].split(","):
    tasks = split[sp][:]; random.Random(1).shuffle(tasks); tasks = tasks[:int(sys.argv[3])]
    ranks, norm_ranks = [], []
    for u, f in tasks:
        pre = U.PREFIX + synth2.head(u, f)
        texts = [pre + "".join(f"    {l}\n" for l in UNITS[c]) for c in cands]
        ids = [unit.encode(t) for t in texts]; L = max(map(len, ids)); n0 = len(unit.encode(pre))
        x = torch.zeros(len(ids), L, dtype=torch.long); a = torch.zeros(len(ids), L, 3, dtype=torch.long)
        for i, (t, q) in enumerate(zip(texts, ids)):
            x[i, :len(q)] = torch.tensor(q); a[i, :len(q)] = torch.tensor(U.addresses(t.encode()))
        with torch.no_grad():
            lp = F.log_softmax(net(x, a).float(), -1)
        tot, per = [], []
        for i, q in enumerate(ids):
            s = lp[i, n0 - 1:len(q) - 1].gather(1, torch.tensor(q[n0:])[:, None]).sum().item()
            tot.append(s); per.append(s / (len(q) - n0))
        for sc, out in ((tot, ranks), (per, norm_ranks)):
            order = sorted(range(len(cands)), key=lambda i: -sc[i]); out.append(order.index(cands.index(u[0])))
    res[sp] = {"n": len(tasks), "top1_sum_logp": sum(r == 0 for r in ranks) / len(ranks),
               "top1_mean_logp": sum(r == 0 for r in norm_ranks) / len(ranks),
               "top4_sum": sum(r < 4 for r in ranks) / len(ranks)}
    print(sp, json.dumps(res[sp]), flush=True)
json.dump(res, open(sys.argv[4], "w"), indent=1)
