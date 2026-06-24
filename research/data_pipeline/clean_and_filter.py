"""
Limpa + FILTRA o corpus PT bruto (wiki + wikisource + nlm) -> so prosa PT coerente.
Rejeita codigo, markdown, ingles, artefatos, fragmentos. Verifica com wtrans no fim.
"""
import re, glob, hashlib, math
from collections import Counter
RAW = "/home/leonardo/projects/LLM/bytebrain/data/raw"
EXTRA = ["/home/leonardo/projects/LLM/bytebrain/data/multiscript/pt.txt"]
OUT = "/home/leonardo/projects/LLM/bytebrain/data/pt_clean.txt"
PT_SW = re.compile(r"\b(que|não|para|uma|com|de|do|da|em|os|as|ele|ela|mais|como|foi|são|seu|sua|por|isso|este|esta|quando|porque|também|entre|sobre|ser|ter|sem)\b", re.I)
EN_SW = re.compile(r"\b(the|and|of|with|is|are|this|that|for|from|was|were|which|their|have|will)\b", re.I)
ACC = set("ãõáéíóúâêôàçÃÕÁÉÍÓÚÂÊÔÀÇ")
CODE = re.compile(r"[{}<>]|;\s|=>|::|def |function |import |return |```|##| \| |https?://|www\.|R\$|MODEL:|USER:|\bself\b|\bconst\b|\bvoid\b")


def good(p):
    p = p.strip()
    if not (120 <= len(p) <= 3000): return False
    w = re.findall(r"\S+", p)
    if len(w) < 20: return False
    if CODE.search(p): return False
    pt = len(PT_SW.findall(p)); en = len(EN_SW.findall(p))
    if pt < len(w)*0.07 or en > pt: return False                      # PT denso, nao ingles
    if sum(c in ACC for c in p)/len(p) < 0.004: return False          # tem acento
    if sum(c.isdigit() or c in "|/\\[]{}*#=+`~_" for c in p)/len(p) > 0.12: return False  # pouco simbolo/digito
    if p.count(". ") + p.count("? ") + p.count("! ") < 2: return False  # frases de verdade
    return True


def main():
    files = glob.glob(f"{RAW}/*.txt") + glob.glob("/home/leonardo/projects/LLM/bytebrain/data/agents/*.txt") + EXTRA
    seen = set(); kept = []; tot = 0; raw = 0
    for fp in files:
        try: txt = open(fp, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        raw += len(txt.encode())
        for para in re.split(r"\n\s*\n", txt):
            para = re.sub(r"[ \t]+", " ", para).strip()
            if good(para):
                h = hashlib.md5(para[:120].encode()).hexdigest()
                if h not in seen:
                    seen.add(h); kept.append(para); tot += len(para.encode())
    blob = "\n\n".join(kept)
    open(OUT, "w", encoding="utf-8").write(blob)
    print(f"bruto {raw/1e6:.1f}MB -> LIMPO {tot/1e6:.1f}MB ({len(kept)} paragrafos)")
    # verifica coesao (wtrans) do corpus limpo
    Wd = re.findall(r"[a-zàáâãéêíóôõúüç]+", blob.lower()); WUNI = Counter(Wd); WBI = Counter(zip(Wd, Wd[1:])); WV = len(WUNI)
    def wtrans(t):
        w = re.findall(r"[a-zàáâãéêíóôõúüç]+", t.lower())
        if len(w) < 3: return 12.0
        return float(sum(-math.log((WBI.get((w[i], w[i+1]), 0)+0.05)/(WUNI.get(w[i], 0)+0.05*WV)) for i in range(len(w)-1))/(len(w)-1))
    import random; random.seed(0)
    if len(blob) > 2000:
        samp = [blob[random.randint(0, len(blob)-400):][:300] for _ in range(20)]
        print(f"wtrans medio do corpus limpo: {sum(wtrans(s) for s in samp)/len(samp):.2f} (quanto menor, mais coeso; alvo bom <7)")
        print(f"amostra:\n{blob[:300]}")


if __name__ == "__main__":
    main()
