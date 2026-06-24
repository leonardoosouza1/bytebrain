"""Turn the Wikipedia PT dump into a clean training corpus — dependency-free.

wikiextractor is broken on Python 3.12, so we stream the bz2 XML ourselves: pull the <text> of
each page, strip the common wiki markup with regexes, split into paragraphs, and pass every
paragraph through src.data.is_portuguese_prose (our quality gate, which drops anything still
markup-y). Streams page-by-page, so memory stays flat even on the multi-GB dump.

    python build_wiki_corpus.py
"""
import bz2
import hashlib
import html
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import is_portuguese_prose  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
DUMP = f"{BASE}/data/dumps/ptwiki-latest-pages-articles.xml.bz2"
OUT = f"{BASE}/data/pt_big.txt"

_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_REF1 = re.compile(r"<ref[^>]*?/>")
_REF2 = re.compile(r"<ref[^>]*?>.*?</ref>", re.S)
_FILELINK = re.compile(r"\[\[(?:File|Image|Ficheiro|Imagem|Categoria|Category|Anexo|Predefinição)[^\]]*\]\]", re.I)
_PIPELINK = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")
_LINK = re.compile(r"\[\[([^\]]*)\]\]")
_EXTLINK1 = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
_EXTLINK2 = re.compile(r"\[https?://\S+\]")
_HEADING = re.compile(r"={2,}\s*([^=\n]*?)\s*={2,}")
_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_HTML = re.compile(r"<[^>]+>")


def clean(t):
    t = html.unescape(t)
    t = _TABLE.sub("", t)
    t = _REF2.sub("", t)
    t = _REF1.sub("", t)
    for _ in range(6):                       # templates can nest
        t2 = _TEMPLATE.sub("", t)
        if t2 == t:
            break
        t = t2
    t = _FILELINK.sub("", t)
    t = _PIPELINK.sub(r"\2", t)
    t = _LINK.sub(r"\1", t)
    t = _EXTLINK1.sub(r"\1", t)
    t = _EXTLINK2.sub("", t)
    t = _HEADING.sub(r"\1.", t)
    t = t.replace("'''", "").replace("''", "")
    t = _HTML.sub("", t)
    return t


def pages(dump):
    """Stream the <text>...</text> body of each page from the bz2 XML."""
    in_text = False
    buf = []
    with bz2.open(dump, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not in_text:
                i = line.find("<text")
                if i != -1:
                    j = line.find(">", i)
                    rest = line[j + 1:] if j != -1 else ""
                    if "</text>" in rest:
                        yield rest.split("</text>")[0]
                    else:
                        in_text = True
                        buf = [rest]
            else:
                if "</text>" in line:
                    buf.append(line.split("</text>")[0])
                    in_text = False
                    yield "".join(buf)
                else:
                    buf.append(line)


def main():
    seen = set()
    kept = tot = npages = 0
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as out:
        for raw in pages(DUMP):
            npages += 1
            for para in re.split(r"\n+", clean(raw)):
                para = re.sub(r"[ \t]+", " ", para).strip()
                if is_portuguese_prose(para):
                    h = hashlib.md5(para[:120].encode()).hexdigest()
                    if h not in seen:
                        seen.add(h)
                        out.write(para + "\n\n")
                        kept += 1
                        tot += len(para.encode())
            if npages % 50000 == 0:
                print(f"  {npages} paginas | limpo {tot/1e6:.0f}MB ({kept} paragrafos) | {(time.time()-t0)/60:.0f}min", flush=True)
    print(f"\nPRONTO: {npages} paginas -> LIMPO {tot/1e6:.0f}MB ({kept} paragrafos) em {(time.time()-t0)/60:.0f}min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
