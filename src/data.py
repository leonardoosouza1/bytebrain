"""Portuguese-prose corpus filter & builder.

Garbage in, garbage out is brutal for a byte-level model: it will happily learn `MODEL:`, code
fences, and English boilerplate byte-for-byte. `is_portuguese_prose` is the strict gate that keeps
only natural Portuguese paragraphs; `build_corpus` applies it across many source files, dedups, and
returns one clean blob ready for training.
"""
import glob
import hashlib
import re

# Dense Portuguese function words — a paragraph of real PT prose hits many of these.
_PT_STOPWORDS = re.compile(
    r"\b(que|não|para|uma|com|de|do|da|em|os|as|ele|ela|mais|como|foi|são|seu|sua|por|isso|"
    r"este|esta|quando|porque|também|entre|sobre|ser|ter|sem)\b",
    re.I,
)
_EN_STOPWORDS = re.compile(
    r"\b(the|and|of|with|is|are|this|that|for|from|was|were|which|their|have|will)\b", re.I
)
_ACCENTS = set("ãõáéíóúâêôàçÃÕÁÉÍÓÚÂÊÔÀÇ")
# Markers of code, markup, chat templates, and other non-prose noise.
_NOISE = re.compile(
    r"[{}<>]|;\s|=>|::|def |function |import |return |```|##| \| |https?://|www\.|R\$|"
    r"MODEL:|USER:|\bself\b|\bconst\b|\bvoid\b"
)


def is_portuguese_prose(paragraph: str) -> bool:
    """True if `paragraph` looks like a real chunk of natural Portuguese prose."""
    p = paragraph.strip()
    if not (120 <= len(p) <= 3000):
        return False
    words = re.findall(r"\S+", p)
    if len(words) < 20 or _NOISE.search(p):
        return False
    pt = len(_PT_STOPWORDS.findall(p))
    en = len(_EN_STOPWORDS.findall(p))
    if pt < len(words) * 0.07 or en > pt:                       # dense PT, not English
        return False
    if sum(c in _ACCENTS for c in p) / len(p) < 0.004:          # has accents
        return False
    if sum(c.isdigit() or c in "|/\\[]{}*#=+`~_" for c in p) / len(p) > 0.12:  # few symbols/digits
        return False
    if p.count(". ") + p.count("? ") + p.count("! ") < 2:       # actual sentences
        return False
    return True


def build_corpus(globs, out_path: str = None) -> str:
    """Read every file matched by `globs`, keep deduplicated Portuguese-prose paragraphs, return
    them joined by blank lines. Optionally write the result to `out_path`."""
    if isinstance(globs, str):
        globs = [globs]
    files = [f for g in globs for f in glob.glob(g)]
    seen, kept = set(), []
    for fp in files:
        try:
            txt = open(fp, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for para in re.split(r"\n\s*\n", txt):
            para = re.sub(r"[ \t]+", " ", para).strip()
            if is_portuguese_prose(para):
                h = hashlib.md5(para[:120].encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    kept.append(para)
    blob = "\n\n".join(kept)
    if out_path:
        open(out_path, "w", encoding="utf-8").write(blob)
    return blob
