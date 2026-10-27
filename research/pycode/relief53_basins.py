"""#53 adaptation (rule 7) of the FLOOD arm: score lines by POOL VOLUME of connected flood components.

Written 2026-10-27 10:40 (from `date`), BEFORE running.

Why: the Opus review of relief53.py found that with max-aggregation flood depth = S + W with W ~ 0 almost
everywhere (every line touches column 0, next to the outlet), so flood-max ranks lines exactly like relief-max
(139/140 smoke instances identical). That arm cannot test water. iara-agua's inpaint found holes by FLOOD-FILL PER
COMPONENT: here each connected set of wet cells (depth > 0, 4-neighbour) is one pool; its volume = sum of depth.
A line's score = the largest volume among pools that touch the line (a pool spanning rows 3-4 scores on both).
Same relief, same defects (defects.jsonl of the full run), same targets, same tie-break rule.
Arms: relief-max (repeated as reference), flood-max (repeated), basin-volume (flood), basin-volume on the
plain raised to the function's median height ("sea level = typical code", so only below-typical regions are wet).

PREDICTIONS (pooled top-1, written before running):
  B1 basin-volume vs relief-max: within -10..+3 pp on >= 5 of 6 real defect types (volume favours long surprising
     lines, which mostly are not the defect); positive control junk stays >= 0.85.
  B2 median-plain basin-volume: within +-5 pp of basin-volume.
  B3 delete_line is the only type where basin-volume may beat relief (gap straddles 2 rows), +0..+5 pp.

10:52: outlet rows below the function (same fix as relief53.py run 2); run 1 kept as run1_basins.*.
Run (CPU, no LM needed):
  cd /home/leonardo/projects/LLM/bytebrain/research/pycode && HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=2 nice -n 5 timeout 3600 /home/leonardo/projects/LLM/.venv-rocm/bin/python -u relief53_basins.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relief53 as R  # noqa: E402

OUT = R.OUT


def main():
    torch.set_num_threads(2)
    t0 = time.time()
    bng = R.ByteNgram(None, os.path.join(OUT, "byte_ngram.npz"))
    jn = R.Junctions(None, os.path.join(OUT, "junctions.json"))
    jn.setup()
    stats = json.load(open(os.path.join(OUT, "results.json")))["config"]["calib_stats"]
    inst = [json.loads(l) for l in open(os.path.join(OUT, "defects.jsonl"))]
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
    log(f"{len(inst)} instances")

    def relief(t):
        c = R.channels(t, bng, jn)
        z = [np.nan_to_num((c[ch] - stats[ch][0]) / stats[ch][1], nan=0.0) for ch in ("byte", "tok", "line")]
        return np.where(c["mask"], z[0] + z[1] + z[2], 0.0), c["mask"]

    Hc, Wc = R.RMAX + 2, R.CMAX + 3
    rng = np.random.default_rng(53)
    rows_out = []
    for b in range(0, len(inst), 512):
        part = inst[b:b + 512]
        SM = [relief(i["text"]) for i in part]
        h0 = R.canvas([-S for S, _ in SM], Hc, Wc)
        om = R.outlet_mask([S.shape[0] for S, _ in SM], Hc, Wc, "cpu")
        d0 = R.flood(h0, om).numpy()
        # median-plain variant: plain (off-code cells) at the function's median code height
        hm = []
        for S, m in SM:
            hm.append(np.where(m, -S, float(np.median(-S[m]))))
        hmc = R.canvas(hm, Hc, Wc)
        for k, (S, m) in enumerate(SM):     # also the canvas area outside the function grid
            med = float(np.median(-S[m]))
            r_, c_ = S.shape
            pad = torch.full((Hc, Wc), med)
            pad[0, :] = pad[-1, :] = pad[:, 0] = pad[:, -1] = 0.0
            pad[1 + r_:, :] = 0.0                                   # outlet rows below the function (run-2 fix)
            pad[1:1 + r_, 1:1 + c_] = hmc[k, 1:1 + r_, 1:1 + c_]
            hmc[k] = pad
        dm = R.flood(hmc, om).numpy()
        for k, i in enumerate(part):
            S, m = SM[k]
            nR = S.shape[0]
            out = {"seed": i["seed"], "kind": i["kind"], "fid": i["fid"], "targets": i["targets"]}
            sc = {"relief": R.line_scores(S, m, "max"),
                  "flood": R.line_scores(d0[k, 1:1 + nR, 1:R.CMAX + 2], m, "max")}
            for name, d in (("basin", d0[k]), ("basin_median", dm[k])):
                wet = d > 1e-6
                lab, n = ndimage.label(wet)
                vol = ndimage.sum(d, lab, index=np.arange(1, n + 1)) if n else np.zeros(0)
                vmap = np.zeros_like(d)
                if n:
                    vmap[wet] = vol[lab[wet] - 1]
                sc[name] = R.line_scores(vmap[1:1 + nR, 1:R.CMAX + 2], m, "max")
            for kk, v in sc.items():
                v = v + rng.random(nR) * 1e-7
                o = np.argsort(-v)
                out[kk + "@1"] = int(o[0] in i["targets"])
                out[kk + "@3"] = int(any(x in i["targets"] for x in o[:3]))
            rows_out.append(out)
        log(f"{min(b + 512, len(inst))}/{len(inst)}")

    def boot(vals, fids, n=1000):
        r = np.random.default_rng(0)
        u, inv = np.unique(fids, return_inverse=True)
        sm = np.bincount(inv, weights=vals, minlength=len(u))
        ct = np.bincount(inv, minlength=len(u)).astype(float)
        pick = r.integers(0, len(u), size=(n, len(u)))
        bs = sm[pick].sum(1) / ct[pick].sum(1)
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    res = {}
    arms = ("relief", "flood", "basin", "basin_median")
    log("kind".ljust(14) + "".join(a.rjust(14) for a in arms) + "   basin-relief [CI]   basin_median-relief [CI]")
    for kind in R.DEFECTS:
        rr = [r for r in rows_out if r["kind"] == kind]
        fids = np.array([r["fid"] for r in rr])
        e = {"n": len(rr)}
        for a in arms:
            for k in ("@1", "@3"):
                v = np.array([r[a + k] for r in rr], float)
                e[a + k] = float(v.mean())
                e[a + k + "_ci"] = boot(v, fids)
        for a in ("basin", "basin_median"):
            d = np.array([r[a + "@1"] - r["relief@1"] for r in rr], float)
            e[a + "-relief@1"] = float(d.mean())
            e[a + "-relief@1_ci"] = boot(d, fids)
        res[kind] = e
        log(kind.ljust(14) + "".join(f"{e[a + '@1']:.3f}".rjust(14) for a in arms)
            + f"   {e['basin-relief@1']:+.3f} [{e['basin-relief@1_ci'][0]:+.3f},{e['basin-relief@1_ci'][1]:+.3f}]"
            + f"   {e['basin_median-relief@1']:+.3f} [{e['basin_median-relief@1_ci'][0]:+.3f},"
              f"{e['basin_median-relief@1_ci'][1]:+.3f}]")
    json.dump(res, open(os.path.join(OUT, "basins.json"), "w"), indent=1)
    log(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
