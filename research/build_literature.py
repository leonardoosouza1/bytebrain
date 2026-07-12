#!/usr/bin/env python3
"""Baixa LITERATURA clássica em português (domínio público) via gutendex/Gutenberg — o "português bem
escrito" que o Leonardo pediu (Machado de Assis, Eça de Queirós, Camões...). Limpa boilerplate do
Gutenberg. Grava literature_pt.txt até ~TARGET_MB."""
import sys, time, os, re, json, urllib.request
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
OUT = "/home/leonardo/projects/LLM/bytebrain/data/literature_pt.txt"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
TARGET_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 40

def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (corpus builder)"})
    return urllib.request.urlopen(req, timeout=timeout).read()

def strip_gutenberg(txt):
    # mantém só o corpo entre os marcadores START/END do Project Gutenberg
    s = re.search(r"\*\*\* ?START OF.*?\*\*\*", txt, re.I)
    e = re.search(r"\*\*\* ?END OF.*?\*\*\*", txt, re.I)
    if s: txt = txt[s.end():]
    if e: txt = txt[:e.start()] if not s else txt[:e.start()-s.end() if e.start()>s.end() else len(txt)]
    return txt.strip()

written = 0; nb = 0; page = 1; seen = set()
with open(OUT, "w", encoding="utf-8") as out:
    while written < TARGET_MB * 1e6 and page <= 12:
        try:
            data = json.loads(get(f"https://gutendex.com/books?languages=pt&page={page}"))
        except Exception as ex:
            log(f"gutendex page {page} falhou: {str(ex)[:60]}"); break
        results = data.get("results", [])
        if not results: break
        log(f"página {page}: {len(results)} livros")
        for b in results:
            if written >= TARGET_MB * 1e6: break
            fmts = b.get("formats", {})
            url = None
            for k, v in fmts.items():
                if k.startswith("text/plain") and not v.endswith(".zip"): url = v; break
            if not url or url in seen: continue
            seen.add(url)
            try:
                raw = get(url).decode("utf-8", "ignore")
            except Exception:
                continue
            body = strip_gutenberg(raw)
            if len(body) < 5000: continue
            title = (b.get("title") or "?")[:40]
            out.write(body); out.write("\n\n"); written += len(body.encode("utf-8")) + 2; nb += 1
            log(f"  + {title:40} ({len(body)/1e3:.0f}KB) | total {written/1e6:.1f}MB, {nb} livros")
        page += 1
log(f"salvo {OUT} | {written/1e6:.1f} MB, {nb} livros de literatura PT")
log(f"DONE ({time.time()-t0:.0f}s)")
