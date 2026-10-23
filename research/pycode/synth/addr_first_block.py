# usage: addr_first_block.py CKPT SPLITS N OUT.json [force]   (force = block 0 given correct)
"""E0: where does the addrseg checkpoint fail? greedy decode on CPU, segment the body into units, first wrong block."""
import json, random, sys, collections
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/pycode/synth")
import torch
torch.set_num_threads(4)
import units as U, synth2
from synth2 import UNITS, FINAL, name_of, source, grade
from eval_samples import cut_function

FORCE = len(sys.argv) > 5
ck = torch.load(sys.argv[1], map_location="cpu")
split = json.load(open(f"{synth2.D}/splits.json")); split["test_len4"] = U.len4_split(split)
unit = U.make_unit("addrseg", split)
sh = U.SHAPE["addrseg"]
net = U.AddrGPT(unit.vocab, 256, 4, 4, sh["ctx"], "addrseg"); net.load_state_dict(ck["model"]); net.eval()
raw = torch.tensor(unit.to_q)
lut = {tuple(v): u for u, v in UNITS.items()}
fin = {f"{v}": k for k, v in FINAL.items()}

def op(u): a, _, b = u.rpartition("_"); return (a, b) if b.isdigit() else (u, "")

def parse(body):
    ls, i, out = [l[4:] for l in body], 0, []
    while i < len(ls):
        m = [u for t, u in lut.items() if tuple(ls[i:i + len(t)]) == t]
        if not m:
            return out, ls[i:]
        u = max(m, key=lambda u: len(UNITS[u])); out.append(u); i += len(UNITS[u])
    return out, []

res = {}
for sp in sys.argv[2].split(","):
    tasks = split[sp][:]; random.Random(1).shuffle(tasks); tasks = tasks[:int(sys.argv[3])]
    srcs, cls = [], collections.Counter()
    for u, f in tasks:
        p = U.prompt_ids(unit, u, f)
        if FORCE: p = unit.encode(U.PREFIX + synth2.head(u, f) + "".join(f"    {l}\n" for l in UNITS[u[0]]))
        with torch.no_grad():
            g = U.sample(net, torch.tensor([p]), sh["gen"], 0, None, U.stop_tables(unit), raw)[0].tolist()
        t = unit.decode(p + g); s = cut_function(t[len(U.PREFIX):])[0]; srcs.append([s])
        lines = s.split("\n")[1:]; lines = [l for l in lines if l.strip()]
        body, last = lines[:-1], lines[-1] if lines else ""
        gu, rest = parse(body)
        j = next((i for i in range(max(len(u), len(gu))) if i >= len(u) or i >= len(gu) or gu[i] != u[i]), None)
        if j is None and not rest:
            c = "ok_units" if last.strip() == FINAL[f].strip() or (not FINAL[f] and last.strip().startswith("return")) else "final_wrong"
        elif j is None:
            c = "garbled_tail"
        elif j >= len(gu):
            c = f"blk{j}:garbled" if rest else f"blk{j}:missing"
        elif j >= len(u):
            c = f"blk{j}:extra"
        else:
            a, b = op(gu[j]), op(u[j])
            c = f"blk{j}:" + ("wrong_N" if a[0] == b[0] else "wrong_op" + ("(=seg%d)" % u.index(gu[j]) if gu[j] in u else ""))
        cls[c] += 1
    ok = sum(x[0] == 1 for x in grade(tasks, srcs))
    res[sp] = {"n": len(tasks), "greedy_pass": ok / len(tasks), "classes": dict(cls.most_common())}
    print(sp, json.dumps(res[sp]), flush=True)
json.dump(res, open(sys.argv[4], "w"), indent=1)
