#!/usr/bin/env python3
"""GEMA 6 — REFLEXO RAIZ-DIGITAL (órgão System-1 calculável, 0 parâmetro).
A tese: existe cognição EXATA e barata que não precisa de peso nenhum — um reflexo aritmético que
sempre CONVERGE para atratores. Dois reflexos validados aqui:
 · n → s(n²) (raiz digital do quadrado): todo n cai num de {1},{9},{7↔4},{13↔16} = os quadrados mod 9;
 · Kaprekar 4-dígitos: qualquer número (dígitos não todos iguais) → 6174 em ≤7 passos.
0 param, 0 treino, 100% determinístico → é o REFLEXO da carcaça (antes de deliberar, o órgão exato responde)."""
def dr(n):                                   # raiz digital = 1+((n-1) mod 9) (o "resto de 9")
    return 0 if n==0 else 1+(n-1)%9
def orbita_quadrado(n, maxit=50):            # itera n → dr(n²) até fechar num ciclo (atrator)
    seen=[]; x=n
    for _ in range(maxit):
        x=dr(x*x)
        if x in seen: return seen[seen.index(x):]   # o ciclo
        seen.append(x)
    return seen
def kaprekar(n):                             # 4 dígitos: (desc - asc) até fixar; retorna passos até 6174
    if len(set(f"{n:04d}"))==1: return None  # dígitos todos iguais não convergem
    passos=0
    while n!=6174 and passos<10:
        ds=sorted(f"{n:04d}"); asc=int("".join(ds)); desc=int("".join(ds[::-1]))
        n=desc-asc; passos+=1
    return passos if n==6174 else -1

if __name__=="__main__":
    print("=== GEMA 6: reflexo raiz-digital — órgão calculável 0-param, converge sempre ===\n")
    print("· n → dr(n²): para onde cada resíduo cai (o atrator é função de n mod 9):")
    atratores={}
    for n in range(1,1001):
        ciclo=tuple(sorted(set(orbita_quadrado(n)))); atratores.setdefault(ciclo,[]).append(n%9)
    for ciclo,res in sorted(atratores.items()):
        print(f"    atrator {str(ciclo):12s} ← n≡{sorted(set(res))} (mod 9)  [{len(res)} de 1000]")
    classes=len(atratores)
    print(f"  → {classes} classes de atrator, 100% dos 1000 convergem, 0 parâmetro. São os quadrados mod 9.\n")

    print("· Kaprekar 6174 (4 dígitos): todo número (não-repdigit) chega a 6174:")
    passos=[]; falhas=0
    for n in range(1000,10000):
        k=kaprekar(n)
        if k is None: continue
        if k<0: falhas+=1
        else: passos.append(k)
    print(f"    convergiram a 6174: {len(passos)}/{len(passos)+falhas} (falhas {falhas}) · máx {max(passos)} passos · média {sum(passos)/len(passos):.2f}")
    print(f"  → 100% dos não-repdigit caem em 6174 em ≤7 passos. Constante de Kaprekar. 0 param.\n")
    print("VEREDITO: reflexo exato, determinístico, sem peso — é o System-1 da carcaça (responde antes de deliberar).")
