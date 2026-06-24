"""fetcher nlm CONTINUO: gera PT rico a noite toda, varrendo muitos subtemas dos notebooks."""
import subprocess, json, re, time
OUT = "/home/leonardo/projects/LLM/bytebrain/data/raw/nlm_loop.txt"
CITE = re.compile(r"\[[\d,\s\-]+\]")
NB = {
    "bdbaaaf6-476b-4c3c-95a9-287110192a54": ["espacos latentes", "transformers e atencao", "tokenizacao", "modelos byte-level", "quantizacao de modelos", "embeddings", "redes neurais profundas", "mecanismo de atencao", "arquiteturas de rede", "inferencia eficiente", "compressao de modelos", "treinamento de modelos", "generalizacao em IA", "representacoes aprendidas"],
    "92474d26-41a7-496a-afe1-b65b969a1dcc": ["descobertas cientificas", "fisica moderna", "biologia molecular", "avancos da medicina", "exploracao espacial", "energia e materiais", "o cosmos", "pesquisa cientifica"],
    "b9d11f2b-7845-48c7-810b-09706a3ba41a": ["tendencias de IA", "tecnologia e sociedade", "aplicacoes de IA", "o futuro da computacao", "etica em inteligencia artificial", "transformacao digital", "automacao"],
    "8c3622bc-af7f-47ba-bf58-8e0e3888cf02": ["a industria de jogos", "design de games", "historia dos videogames", "narrativa em jogos"],
    "9cce9ffa-ee6a-4305-ae8b-d52499cc17e7": ["startups", "empreendedorismo", "modelos de negocio", "inovacao", "mercado de tecnologia"],
}


def q(nb, topic, variant):
    styles = ["um texto rico e detalhado", "uma explicacao didatica e aprofundada", "um ensaio articulado e fluido", "uma analise completa e bem escrita"]
    pr = f"Escreva em portugues do Brasil {styles[variant%4]} (varios paragrafos, prosa natural, sem listas) sobre: {topic}."
    try:
        r = subprocess.run(["nlm", "notebook", "query", nb, pr, "--json", "-t", "110"], capture_output=True, text=True, timeout=140)
        return CITE.sub("", json.loads(r.stdout).get("answer", "")).replace("**", "").replace("*", "").replace("##", "")
    except Exception as e:
        print("nlm err", str(e)[:60], flush=True); return ""


def main():
    t0 = time.time(); tot = 0; variant = 0
    with open(OUT, "w", encoding="utf-8") as f:
        while time.time()-t0 < 9.5*3600:        # ~noite toda
            for nb, topics in NB.items():
                for tp in topics:
                    if time.time()-t0 > 9.5*3600: break
                    a = q(nb, tp, variant)
                    if len(a) > 400:
                        f.write(a.strip()+"\n\n"); f.flush(); tot += len(a.encode())
                        print(f"[nlm-loop] +{len(a)} ({tot/1e6:.2f}MB) v{variant} {tp[:25]}", flush=True)
                    time.sleep(3)
            variant += 1
    print(f"[nlm-loop] FIM {tot/1e6:.2f}MB", flush=True)


if __name__ == "__main__":
    main()
