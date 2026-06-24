"""
ByteBrain — construtor de corpus PT-BR rico (nlm/NotebookLM + Wikipedia PT).
Volume rapido: Wikipedia PT random+extracts (~1MB/call). Qualidade: nlm queries nos notebooks.
Salva continuamente em data/pt_overnight.txt. Roda em background enquanto monto o treino.
"""
import os, json, time, re, subprocess, urllib.request, urllib.parse
OUT = "/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
TARGET_MB = 60
TIME_BUDGET = 2400  # 40 min teto

NB = {  # notebook -> prompts PT ricos no dominio dele
    "bdbaaaf6-476b-4c3c-95a9-287110192a54": [  # Byte Latent Transformers
        "Explique em portugues, detalhadamente, espacos latentes em deep learning, com exemplos.",
        "Descreva em portugues como funcionam transformers, atencao, embeddings e tokenizacao.",
        "Escreva em portugues sobre modelos de linguagem byte-level, robustez e multilinguismo.",
        "Explique em portugues quantizacao, compressao de modelos e eficiencia em inferencia."],
    "92474d26-41a7-496a-afe1-b65b969a1dcc": [  # Science News
        "Resuma em portugues as principais noticias e descobertas de ciencia, em varios paragrafos.",
        "Escreva um texto em portugues explicando avancos cientificos recentes de forma didatica."],
    "b9d11f2b-7845-48c7-810b-09706a3ba41a": [  # Tech & AI News
        "Escreva em portugues um panorama detalhado de tecnologia e inteligencia artificial.",
        "Explique em portugues as tendencias de IA, modelos e aplicacoes, em varios paragrafos."],
    "8c3622bc-af7f-47ba-bf58-8e0e3888cf02": [  # Games News
        "Escreva em portugues sobre jogos, a industria de games e lancamentos, detalhadamente."],
    "9cce9ffa-ee6a-4305-ae8b-d52499cc17e7": [  # Business & Startups
        "Escreva em portugues sobre negocios, startups e empreendedorismo, de forma rica e detalhada."],
}

CITE = re.compile(r"\[[\d,\s\-]+\]")


def write(f, txt):
    f.write(txt.strip() + "\n\n"); f.flush()


def nlm_query(nb, prompt):
    try:
        r = subprocess.run(["nlm", "notebook", "query", nb, prompt, "--json", "-t", "100"],
                           capture_output=True, text=True, timeout=130)
        d = json.loads(r.stdout)
        ans = d.get("answer", "")
        return CITE.sub("", ans).replace("**", "").replace("*", "")
    except Exception as e:
        print(f"  nlm falhou: {e}", flush=True); return ""


def wiki_batch(n=8):
    url = ("https://pt.wikipedia.org/w/api.php?action=query&format=json&generator=random"
           f"&grnnamespace=0&grnlimit={n}&prop=extracts&explaintext=1&exlimit={n}")
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "bytebrain-research/0.4"}), timeout=25))
        return [p.get("extract", "") for p in d.get("query", {}).get("pages", {}).values() if len(p.get("extract", "")) > 400]
    except Exception as e:
        print(f"  wiki falhou: {e}", flush=True); return []


def main():
    t0 = time.time(); total = 0
    f = open(OUT, "w", encoding="utf-8")
    # seed: PT existente
    for p in ["/home/leonardo/projects/LLM/bytebrain/data/multiscript/pt.txt"]:
        if os.path.exists(p):
            t = open(p, encoding="utf-8").read(); write(f, t); total += len(t.encode())
    print(f"seed PT: {total/1e6:.1f}MB", flush=True)
    # fase 1: nlm qualidade
    print("=== nlm: gerando PT rico (curado) ===", flush=True)
    for nb, prompts in NB.items():
        for pr in prompts:
            a = nlm_query(nb, pr)
            if len(a) > 300: write(f, a); total += len(a.encode()); print(f"  +nlm {len(a)} chars (total {total/1e6:.1f}MB)", flush=True)
            time.sleep(2)
    # fase 2: Wikipedia PT volume (random+extracts)
    print("=== wikipedia PT: volume ===", flush=True)
    calls = 0
    while total < TARGET_MB*1e6 and time.time()-t0 < TIME_BUDGET:
        for ext in wiki_batch(8):
            write(f, ext); total += len(ext.encode())
        calls += 1
        if calls % 5 == 0: print(f"  wiki {calls} calls | {total/1e6:.1f}MB | {time.time()-t0:.0f}s", flush=True)
        time.sleep(2)
    f.close()
    print(f"=== CORPUS PT PRONTO: {total/1e6:.1f}MB em {time.time()-t0:.0f}s -> {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
