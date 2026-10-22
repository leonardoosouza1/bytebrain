"""One table for the training arms: mirror axes (mirror_*.json) + line-type law + GPU checker on the same samples.

Usage: ab_table.py RUN_DIR   (expects mirror_real.json and mirror_{clean,raw,small}.json there)
"""

import json
import sys

import numpy as np
import torch

import token_pairs as TP
from line_laws import D, LineLaw, split_funcs


def main():
    run = sys.argv[1]
    torch.cuda.set_per_process_memory_fraction(0.1)
    law = LineLaw([f for f in split_funcs(f"{D}/clean_train.txt") if f.strip()])
    table = np.load(TP.TABLE)
    rows = []
    for arm in sys.argv[2:] or ("real", "clean", "raw", "small"):
        m = json.load(open(f"{run}/mirror_{arm}.json"))
        srcs = m.get("samples") or m["examples"]
        tp = TP.run(srcs, lambda B: TP.TokenPairs(B, table=table))
        ll = law.score(srcs)
        names, syn = m["names"][0], m["syntax"][0]
        mi = m["mi_ratio_per_k"]
        rows.append({
            "arm": arm, "n": len(srcs), "parses": syn["parses"],
            "gpu_flagged": round(float(tp.flagged.float().mean()), 3),
            "params_used": names["params_used"], "loads_free": names["loads_free"],
            "line_bits": ll["bits_per_line"], "line_unseen": ll["unseen_transitions"],
            "match_tv": m["match_tv"], "match_mean": m["match_mean"][0], "bigram_kl": m["bigram_kl"],
            "mi_ratio_k8": mi.get("8"), "mi_ratio_k32": mi.get("32"), "len_mean": m["len_mean"][0],
        })
    keys = list(rows[0])
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))
    json.dump(rows, open(f"{run}/ab_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
