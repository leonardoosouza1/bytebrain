"""worker nlm: gera PT rico e LIMPO via NotebookLM (muitos temas nos notebooks do Leonardo)."""
import subprocess, json, re, time
OUT = "/home/leonardo/projects/LLM/bytebrain/data/raw/nlm.txt"
CITE = re.compile(r"\[[\d,\s\-]+\]")
NB = {
    "bdbaaaf6-476b-4c3c-95a9-287110192a54": ["espacos latentes em deep learning", "transformers e atencao", "tokenizacao e modelos byte-level", "quantizacao e compressao de modelos", "redes neurais e aprendizado profundo", "embeddings e representacoes", "arquiteturas de IA modernas", "inferencia e eficiencia computacional"],
    "92474d26-41a7-496a-afe1-b65b969a1dcc": ["descobertas cientificas recentes", "fisica e o universo", "biologia e genetica", "avancos da medicina", "exploracao espacial"],
    "b9d11f2b-7845-48c7-810b-09706a3ba41a": ["tendencias de inteligencia artificial", "tecnologia e sociedade", "modelos de linguagem e aplicacoes", "futuro da computacao", "etica em IA"],
    "8c3622bc-af7f-47ba-bf58-8e0e3888cf02": ["a industria de jogos", "design de games", "historia dos videogames"],
    "9cce9ffa-ee6a-4305-ae8b-d52499cc17e7": ["startups e empreendedorismo", "modelos de negocio", "inovacao e mercado"],
}


def q(nb, topic):
    pr = f"Escreva em portugues do Brasil um texto rico, detalhado e bem articulado (varios paragrafos) sobre: {topic}. Use linguagem natural e fluida, sem listas nem topicos."
    try:
        r = subprocess.run(["nlm", "notebook", "query", nb, pr, "--json", "-t", "110"], capture_output=True, text=True, timeout=140)
        a = json.loads(r.stdout).get("answer", "")
        return CITE.sub("", a).replace("**", "").replace("*", "").replace("##", "")
    except Exception as e:
        print("nlm err", e, flush=True); return ""


def main():
    tot = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for nb, topics in NB.items():
            for tp in topics:
                a = q(nb, tp)
                if len(a) > 400:
                    f.write(a.strip()+"\n\n"); f.flush(); tot += len(a.encode())
                    print(f"[nlm] +{len(a)} ({tot/1e6:.2f}MB) {tp[:30]}", flush=True)
                time.sleep(2)
    print(f"[nlm] FIM {tot/1e6:.2f}MB", flush=True)


if __name__ == "__main__":
    main()
