#!/usr/bin/env python3
"""TBH-TOOL — ferramenta de exploração da arquitetura do TaskbarHero (pro Claude e pro Leonardo).
Consulta o tbh_map.json (gerado pelo tbh_dump.py). Foca no código DO JOGO (Assembly-CSharp.dll).
Comandos:
  stats                      visão geral (assemblies, nº de classes do jogo)
  integrations               libs/serviços integrados (Steam, ES3, CodeStage, UniTask, Newtonsoft, TheBackend…)
  systems                    agrupa as classes do jogo por sistema (save/boss/skill/item/ui/net…)
  find <regex>               procura classe OU método por padrão
  class <Nome>               campos + métodos + assembly de uma classe
  ns <Namespace>             classes de um namespace
  methods <Nome>             só os métodos de uma classe
  refs <palavra>             classes/métodos que MENCIONAM a palavra (mapa de acoplamento)
  flow <Nome>                heurística de fluxo: métodos da classe + quem a referencia (candidatos a hook)
"""
import json, sys, re, os
MP=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tbh_map.json")
D=json.load(open(MP)); C=D["classes"]
GAME_ASM="Assembly-CSharp.dll"
game=[c for c in C if c["asm"]==GAME_ASM]
def full(c): return (c["ns"]+"." if c["ns"] else "")+c["name"]

def stats():
    from collections import Counter
    asm=Counter(c["asm"] for c in C)
    print(f"metadata v{D['version']} · {D['ntypes']} tipos · {D['nmethods']} métodos · {D['nfields']} campos")
    print(f"\nCÓDIGO DO JOGO ({GAME_ASM}): {len(game)} classes")
    print("\ntop assemblies por nº de classes:")
    for a,n in asm.most_common(12): print(f"  {n:6d}  {a}")

INTEG={"Steam":r"Steam","Easy Save 3 (save cripto)":r"\bES3|EasySave","CodeStage AntiCheat":r"CodeStage|AntiCheat|ObscuredI?nt|ObscuredFloat",
 "UniTask (async)":r"Cysharp|UniTask","Newtonsoft JSON":r"Newtonsoft|JsonConvert","TheBackend (BaaS)":r"BackEnd|TheBackend|BackndServer",
 "DOTween":r"DG\.Tweening|DOTween","Addressables":r"Addressable|ResourceManager","Firebase":r"Firebase","Google/GPGS":r"GooglePlay|GPGS",
 "Rewired/Input":r"Rewired|InputSystem","FMOD/Audio":r"FMOD","Analytics":r"Analytics|Amplitude|GameAnalytics"}
def integrations():
    allns={c["asm"] for c in C}|{c["ns"] for c in C if c["ns"]}
    txt="\n".join(sorted(allns))
    print("INTEGRAÇÕES DETECTADAS:")
    for nome,pat in INTEG.items():
        hits=sorted({n for n in allns if re.search(pat,n,re.I)})
        if hits: print(f"  ✓ {nome}: "+", ".join(hits[:6]))

SYS={"Save/Progressão":r"save|account|progress|profile|slot|serializ","Boss/Inimigo":r"boss|enemy|monster|undead|void|spawn",
 "Skill/Habilidade":r"skill|ability|abilit|passive|attribute","Item/Inventário":r"item|inventor|equip|loot|drop|gear",
 "Combate":r"attack|damage|combat|projectile|hit|weapon|dash","Herói/Player":r"player|hero|character|avatar",
 "UI":r"panel|button|ui|hud|popup|window|screen","Rede/Backend":r"backend|server|api|request|http|socket|net",
 "Economia/Loja":r"shop|store|gold|coin|currency|price|buy","Steam":r"steam","Áudio":r"audio|sound|music|sfx",
 "Anti-cheat":r"anticheat|obscured|hash|secret|verify|integrity","Localização":r"locale|language|localiz|i18n"}
def systems():
    print(f"SISTEMAS DO JOGO (classes de {GAME_ASM} agrupadas):")
    for nome,pat in SYS.items():
        hits=[c for c in game if re.search(pat,c["name"],re.I) or re.search(pat,c["ns"],re.I)]
        if hits:
            ex=", ".join(sorted(set(c["name"] for c in hits))[:8])
            print(f"  {nome:20s} {len(hits):4d}  ex: {ex}")

def find(pat):
    rx=re.compile(pat,re.I); shown=0
    for c in game:
        if rx.search(c["name"]): print(f"  [classe] {c['asm']} :: {full(c)}"); shown+=1
        ms=[m for m in c["methods"] if rx.search(m)]
        if ms: print(f"  [métodos de {c['name']}] "+", ".join(ms[:10])); shown+=1
        if shown>60: print("  …(corta)"); break

def klass(nome):
    cs=[c for c in C if c["name"]==nome] or [c for c in C if nome.lower() in c["name"].lower()]
    for c in cs[:6]:
        print(f"\n### {c['asm']} :: {full(c)}  ({c['nf']} campos, {c['nm']} métodos)")
        if c["fields"]: print("  campos:  "+", ".join(c["fields"][:30]))
        if c["methods"]: print("  métodos: "+", ".join(c["methods"][:40]))

def refs(w):
    rx=re.compile(re.escape(w),re.I); hit=[]
    for c in game:
        if rx.search(c["name"]) or any(rx.search(m) for m in c["methods"]) or any(rx.search(f) for f in c["fields"]):
            hit.append(c["name"])
    print(f"'{w}' aparece em {len(hit)} classes do jogo:")
    print("  "+", ".join(sorted(set(hit))[:40]))

def flow(nome):
    c=next((c for c in C if c["name"]==nome),None)
    if not c: print("não achei"); return
    print(f"### {full(c)} — métodos (candidatos a HOOK no MelonLoader):")
    for m in c["methods"][:30]: print(f"   {full(c)}::{m}")
    print(f"\n### quem REFERENCIA '{nome}' (acoplamento):")
    rx=re.compile(re.escape(nome))
    who=[c2["name"] for c2 in game if c2["name"]!=nome and (any(rx.search(m) for m in c2["methods"]) or any(rx.search(f) for f in c2["fields"]))]
    print("   "+", ".join(sorted(set(who))[:30]) or "   (nenhum direto no nome)")


def tree():
    # árvore dev reconstruída (namespaces do jogo → classes)
    from collections import defaultdict
    t=defaultdict(list)
    for c in game: t[c["ns"] or "(sem namespace)"].append(c["name"])
    for ns in sorted(t):
        print(f"{ns}/  ({len(t[ns])})")
        for n in sorted(t[ns])[:8]:
            if not n.startswith("<"): print("   ",n)

def hookgen(system):
    # gera um patch MelonLoader/Harmony que LOGA a entrada dos métodos do sistema escolhido (observar o FLUXO)
    pat=SYS.get(system) or system
    rx=re.compile(pat,re.I)
    sel=[c for c in game if (rx.search(c["name"]) or rx.search(c["ns"])) and not c["name"].startswith("<")]
    print("// gerado por tbh_tool hookgen — cole em TbhTrace/Hooks.g.cs")
    print("// alvos do sistema:",system,f"({len(sel)} classes)")
    print("using HarmonyLib; using MelonLoader;")
    n=0
    for c in sel[:40]:
        cls=(c["ns"]+"." if c["ns"] else "")+c["name"]
        for m in c["methods"][:6]:
            if m in (".ctor",".cctor") or m.startswith("get_") or m.startswith("set_"): continue
            print(f'// TARGET: {cls}::{m}')
            n+=1
            if n>=120: break
        if n>=120: break
    print(f"// {n} métodos-alvo — o TbhTrace.cs resolve e loga cada chamada")

cmd=sys.argv[1] if len(sys.argv)>1 else "stats"
arg=sys.argv[2] if len(sys.argv)>2 else ""
{"stats":stats,"integrations":integrations,"systems":systems,"find":lambda:find(arg),"class":lambda:klass(arg),
 "ns":lambda:[print(full(c)) for c in C if c["ns"]==arg][:50] and None,"methods":lambda:klass(arg),
 "refs":lambda:refs(arg),"flow":lambda:flow(arg),"tree":tree,"hookgen":lambda:hookgen(arg)}.get(cmd,stats)()
