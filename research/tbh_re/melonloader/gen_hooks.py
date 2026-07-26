#!/usr/bin/env python3
"""Gera tbh_hooks.txt (métodos p/ o TbhTrace logar) a partir do tbh_map.json, filtrando nomes legíveis.
Uso: python3 gen_hooks.py <sistema|regex> [max]   ex: python3 gen_hooks.py save 60"""
import json,re,sys,os
here=os.path.dirname(os.path.abspath(__file__))
D=json.load(open(os.path.join(here,"..","tbh_map.json")))
game=[c for c in D["classes"] if c["asm"]=="Assembly-CSharp.dll"]
pat=sys.argv[1] if len(sys.argv)>1 else "save"
mx=int(sys.argv[2]) if len(sys.argv)>2 else 60
rx=re.compile(pat,re.I)
legivel=lambda m: len(m)>=4 and not re.fullmatch(r"[a-z]{2,4}",m) and m not in(".ctor",".cctor")
out=[]
for c in game:
    if c["name"].startswith("<"): continue
    if rx.search(c["name"]) or rx.search(c["ns"]):
        cls=(c["ns"]+"." if c["ns"] else "")+c["name"]
        for m in c["methods"]:
            if legivel(m) and not m.startswith(("get_","set_")): out.append(f"{cls}::{m}")
out=out[:mx]
open(os.path.join(here,"tbh_hooks.txt"),"w").write("\n".join(out)+"\n")
print(f"{len(out)} hooks legíveis → tbh_hooks.txt (copie p/ <Jogo>/UserData/)")
for l in out[:12]: print("  ",l)
