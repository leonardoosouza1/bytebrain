"""
ByteBrain — ideia do Leonardo: texto como SINAL de frequencia-por-letra. Analisa a FORMA da curva.
sinal[i] = log-frequencia do caractere i (comum=alto, raro=baixo). Hipotese:
  coeso = curva calma/suave | repetitivo = curva periodica | caotico = serrilhada.
Mede: suavidade (variacao total), achatamento espectral (FFT: ruidoso vs estruturado), periodicidade (autocorr).
Imprime medidas + exporta sinais p/ grafico. CPU.
"""
import re, math, json, random
from collections import Counter
import numpy as np
random.seed(0); np.random.seed(0)
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt"


def char_logfreq():
    t = open(CORPUS, encoding="utf-8").read()[:3_000_000].lower()
    c = Counter(t); tot = sum(c.values())
    return {ch: math.log((n+1)/tot) for ch, n in c.items()}, math.log(1/tot)


LF, FLOOR = char_logfreq()
def sig(text):
    return np.array([LF.get(ch, FLOOR) for ch in text.lower()])


def measures(s):
    if len(s) < 6:
        return {"suavidade_TV": round(float(np.mean(np.abs(np.diff(s)))), 2) if len(s) > 1 else 0, "achat_espectral": 0, "periodicidade": 0}
    tv = float(np.mean(np.abs(np.diff(s))))                       # variacao total (menor=suave)
    x = s - s.mean(); F = np.abs(np.fft.rfft(x))[1:] + 1e-9       # espectro
    flat = float(np.exp(np.mean(np.log(F))) / np.mean(F))         # achatamento (1=ruido/caotico; baixo=estruturado)
    ac = np.correlate(x, x, "full"); ac = ac[len(ac)//2:]; ac = ac/(ac[0]+1e-9)  # autocorr normalizada (ac[0]=1)
    period = float(np.max(ac[3:min(len(ac), 40)])) if len(ac) > 4 else 0          # pico em lag>=3 (repeticao->alto)
    return {"suavidade_TV": round(tv, 2), "achat_espectral": round(flat, 3), "periodicidade": round(period, 2)}


def main():
    rndtxt = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(120))
    words = [w for w in re.findall(r"[a-zàáâãéêíóôõúç]+", open(CORPUS, encoding="utf-8").read()[:500000].lower()) if len(w) > 2]
    salad = " ".join(random.choice(words) for _ in range(22))
    tests = {
        "COERENTE": "o gato subiu no telhado e dormiu ao sol da tarde enquanto as criancas brincavam no quintal da casa",
        "repeticao": "casa casa casa casa casa casa casa casa casa casa casa casa casa casa casa casa",
        "aleatorio": rndtxt,
        "salada_palavras": salad,
        "Ola_tudo_bem": "ola, tudo bem com voce hoje? espero que sim, tenha um otimo dia.",
        "NOSSO_modelo": "a historia registra de ignorancia nacional que havia um dos metodos historicoeles aceita de energia",
    }
    print("=== FORMA DA CURVA de frequencia-por-letra ===")
    print(f"{'tipo':<18}{'suavidade':>11}{'achat_esp':>11}{'periodic':>10}  interpretacao")
    out = {}
    for name, t in tests.items():
        m = measures(sig(t)); out[name] = list(np.round(sig(t), 2))
        interp = "calmo/coeso" if (m['periodicidade'] < 0.35 and m['suavidade_TV'] < 1.2) else \
                 ("PERIODICO (repeticao)" if m['periodicidade'] >= 0.35 else "CAOTICO")
        print(f"{name:<18}{m['suavidade_TV']:>11}{m['achat_espectral']:>11}{m['periodicidade']:>10}  {interp}")
    # frases curtas (a curva do 'oi')
    print("\n=== curvas curtas (a tua observacao do 'oi') ===")
    for w in ["oi", "ola", "casa", "xyzq"]:
        s = sig(w); print(f"  {w:<6} sinal={list(np.round(s, 2))}  (desce? {'sim' if len(s) > 1 and s[-1] < s[0] else 'nao'})")
    json.dump(out, open("/home/leonardo/projects/LLM/bytebrain/signal_curves.json", "w"))
    print("\ncurvas -> signal_curves.json")


if __name__ == "__main__":
    main()
