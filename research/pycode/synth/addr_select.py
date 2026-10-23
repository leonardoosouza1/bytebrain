# usage: NVIS=8 addr_select.py CKPT SPLITS N OUT.json [K]
"""selector: top-4 block-0 candidates by model score -> greedy completion each -> pick the first that passes 3 visible
examples (fresh inputs, not the 40 graded ones) -> strict grade on the 40 hidden inputs."""
import json, random, sys
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/pycode/synth")
import torch, torch.nn.functional as F
torch.set_num_threads(3)
import units as U, synth2
from synth2 import UNITS, grade
from eval_samples import cut_function
ck = torch.load(sys.argv[1], map_location="cpu")
split = json.load(open(f"{synth2.D}/splits.json"))
unit = U.make_unit("addrseg", split)
sh = U.SHAPE["addrseg"]
net = U.AddrGPT(unit.vocab, 256, 4, 4, sh["ctx"], "addrseg"); net.load_state_dict(ck["model"]); net.eval()
raw, stop = torch.tensor(unit.to_q), U.stop_tables(unit)
cands = sorted(UNITS)
HIDDEN = synth2.INPUTS
_r = random.Random(99)
VISIBLE = [[[_r.randint(-9, 19) for _ in range(_r.randint(3, 9))], _r.randint(0, 6)] for _ in range(int(__import__("os").environ.get("NVIS", 3)))]
assert not any(v in HIDDEN for v in VISIBLE)
K = int(sys.argv[5]) if len(sys.argv) > 5 else 4

def top(u, f, k):
    pre = U.PREFIX + synth2.head(u, f)
    texts = [pre + "".join(f"    {l}\n" for l in UNITS[c]) for c in cands]
    ids = [unit.encode(t) for t in texts]; L = max(map(len, ids)); n0 = len(unit.encode(pre))
    x = torch.zeros(len(ids), L, dtype=torch.long); a = torch.zeros(len(ids), L, 3, dtype=torch.long)
    for i, (t, q) in enumerate(zip(texts, ids)):
        x[i, :len(q)] = torch.tensor(q); a[i, :len(q)] = torch.tensor(U.addresses(t.encode()))
    with torch.no_grad():
        lp = F.log_softmax(net(x, a).float(), -1)
    sc = [lp[i, n0 - 1:len(q) - 1].gather(1, torch.tensor(q[n0:])[:, None]).sum().item() for i, q in enumerate(ids)]
    return [texts[i] for i in sorted(range(len(cands)), key=lambda i: -sc[i])[:k]]

res = {}
for sp in sys.argv[2].split(","):
    tasks = split[sp][:]; random.Random(1).shuffle(tasks); tasks = tasks[:int(sys.argv[3])]
    picked, first, tried, info = [], [], 0, []
    for u, f in tasks:
        outs = []
        for t in top(u, f, K):
            p = unit.encode(t)
            with torch.no_grad():
                g = U.sample(net, torch.tensor([p]), sh["gen"], 0, None, stop, raw)[0].tolist()
            outs.append(cut_function(unit.decode(p + g)[len(U.PREFIX):])[0])
        synth2.INPUTS = VISIBLE
        vis = grade([(u, f)], [outs], strict=False)[0]
        synth2.INPUTS = HIDDEN
        j = next((i for i, v in enumerate(vis) if v == 1.0), 0)
        tried += j + 1
        picked.append([outs[j]]); first.append([outs[0]]); info.append((u, f, any(v == 1.0 for v in vis), j))
    gs, gl = grade(tasks, picked), grade(tasks, picked, strict=False)
    sel = sum(s[0] == 1 for s in gs) / len(tasks)
    import collections
    fail = collections.Counter()
    for (u, f, anyv, j), a, b in zip(info, gs, gl):
        if a[0] == 1: continue
        fail[("no_cand_passed_visible" if not anyv else "passed_visible") + ("|output_ok_state_wrong" if b[0] == 1 else "|output_wrong") + "|final=" + (f.split("_")[0] or "none")] += 1
    print(" failures:", dict(fail.most_common()))
    print(" nonstrict:", sum(s[0] == 1 for s in gl) / len(tasks))
    base = sum(s[0] == 1 for s in grade(tasks, first)) / len(tasks)
    res[sp] = {"n": len(tasks), "k": K, "n_visible": len(VISIBLE), "top1_strict": base, "selected_strict": sel, "mean_candidates_run": tried / len(tasks)}
    print(sp, json.dumps(res[sp]), flush=True)
json.dump(res, open(sys.argv[4], "w"), indent=1)
