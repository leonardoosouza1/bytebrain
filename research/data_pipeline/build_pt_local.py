"""
Monta corpus PT-BR LIMPO de fontes locais (obsidian, iara-brain, docs) + cache.
Filtra por densidade de PT, limpa markdown/código pesado. Qualidade > volume (coesão).
"""
import os, re, glob
OUT = "/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt"
PT_CHARS = set("ãõáéíóúâêôàçÃÕÁÉÍÓÚÂÊÔÀÇ")
PT_WORDS = re.compile(r"\b(que|não|para|uma|com|por|mais|como|está|são|foi|também|sobre|entre|quando|porque|então|isso|cada|sua|seu|nós|você)\b", re.I)

ROOTS = ["/home/leonardo/obsidian-claude-vault", "/home/leonardo/projects/LLM"]
EXTRA = ["/home/leonardo/projects/LLM/bytebrain/data/multiscript/pt.txt"] + glob.glob("/home/leonardo/projects/LLM/iara-brain/*.txt")


def pt_score(t):
    if len(t) < 500: return 0
    acc = sum(c in PT_CHARS for c in t[:20000]) / min(len(t), 20000)
    words = len(PT_WORDS.findall(t[:20000])) / max(1, len(t[:20000])//100)
    return acc*1000 + words   # densidade de acentos + stopwords PT


def clean(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)          # remove blocos de codigo
    t = re.sub(r"`[^`]*`", " ", t)                         # inline code
    t = re.sub(r"https?://\S+", " ", t)                    # urls
    t = re.sub(r"^[\s|>#*\-=+`~_\[\]()]+$", "", t, flags=re.M)  # linhas so de simbolos/markdown
    t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main():
    files = set()
    for r in ROOTS:
        for ext in ("*.md", "*.txt"):
            files.update(glob.glob(f"{r}/**/{ext}", recursive=True))
    files = [f for f in files if not re.search(r"node_modules|\.venv|target/|site-packages|\.git/", f)]
    kept = []; tot = 0
    for f in sorted(files):
        try: t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        if pt_score(t) > 7:                  # PT-heavy
            c = clean(t)
            if len(c) > 400: kept.append(c); tot += len(c.encode())
    # extras (cache PT)
    for f in EXTRA:
        if os.path.exists(f):
            c = clean(open(f, encoding="utf-8").read())
            if len(c) > 400: kept.append(c); tot += len(c.encode())
    # nlm ja gerado (se houver no pt_overnight antigo)
    blob = "\n\n".join(kept)
    open(OUT, "w", encoding="utf-8").write(blob)
    print(f"PT local: {len(kept)} arquivos | {tot/1e6:.2f}MB limpo -> {OUT}")
    print(f"amostra:\n{blob[:400]}")


if __name__ == "__main__":
    main()
