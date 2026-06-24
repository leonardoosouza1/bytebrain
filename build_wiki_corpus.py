"""Turn the Wikipedia PT dump into a clean training corpus.

  1. wikiextractor strips wiki markup from the .bz2 dump -> plain-text JSON articles.
  2. every paragraph is passed through src.data.is_portuguese_prose (our quality gate) and
     deduplicated, then streamed to data/pt_big.txt.

Run after the dump finishes downloading:  python build_wiki_corpus.py
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import is_portuguese_prose  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
DUMP = f"{BASE}/data/dumps/ptwiki-latest-pages-articles.xml.bz2"
EXTRACT = f"{BASE}/data/dumps/extracted"
OUT = f"{BASE}/data/pt_big.txt"


def extract():
    if glob.glob(f"{EXTRACT}/**/wiki_*", recursive=True):
        print("ja extraido, pulando wikiextractor", flush=True)
        return
    print("extraindo markup com wikiextractor (alguns minutos)...", flush=True)
    subprocess.run([sys.executable, "-m", "wikiextractor.WikiExtractor", DUMP,
                    "-o", EXTRACT, "--json", "-q", "--processes", "4"], check=True)


def build():
    files = sorted(glob.glob(f"{EXTRACT}/**/wiki_*", recursive=True))
    print(f"{len(files)} arquivos extraidos -> filtrando", flush=True)
    seen = set()
    kept = tot = raw = 0
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as out:
        for k, fp in enumerate(files):
            for line in open(fp, encoding="utf-8", errors="ignore"):
                try:
                    txt = json.loads(line).get("text", "")
                except Exception:
                    continue
                raw += len(txt)
                for para in re.split(r"\n+", txt):
                    para = re.sub(r"[ \t]+", " ", para).strip()
                    if is_portuguese_prose(para):
                        h = hashlib.md5(para[:120].encode()).hexdigest()
                        if h not in seen:
                            seen.add(h)
                            out.write(para + "\n\n")
                            kept += 1
                            tot += len(para.encode())
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(files)} arquivos | limpo {tot/1e6:.0f}MB ({kept} paragrafos) | {(time.time()-t0)/60:.0f}min", flush=True)
    print(f"\nPRONTO: bruto {raw/1e6:.0f}MB -> LIMPO {tot/1e6:.0f}MB ({kept} paragrafos) em {(time.time()-t0)/60:.0f}min", flush=True)
    print(f"corpus salvo em {OUT}", flush=True)


if __name__ == "__main__":
    extract()
    build()
