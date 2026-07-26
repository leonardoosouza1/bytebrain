#!/usr/bin/env python3
"""Identifica os assemblies de FONTE PÚBLICA embutidos → 'revela' ~5.500 classes sem RE (é código conhecido)."""
import json,re,os
D=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tbh_map.json")))
from collections import Counter
asm=Counter(c["asm"] for c in D["classes"])
LIB={r"MoreMountains":("MoreMountains Tools/Feedbacks","github.com/reunono (Corgi/TopDown assets)","utilidades/juice/feedbacks de gameplay"),
 r"UniTask":("Cysharp UniTask","github.com/Cysharp/UniTask","async/await sem alocação (substitui coroutines)"),
 r"Newtonsoft":("Newtonsoft.Json (Json.NET)","github.com/JamesNK/Newtonsoft.Json","serialização JSON"),
 r"Sirenix|Odin":("Odin Inspector/Serializer","odininspector.com (comercial, API pública)","serialização + inspector no editor"),
 r"DOTween":("DOTween","dotween.demigiant.com","tweening/animação de UI e valores"),
 r"Backend|Backnd|TheBackend":("TheBackend (BaaS)","thebackend.io / SDK público","backend online: contas, inventário, ranking, mail"),
 r"Steamworks|Heathen":("Heathen Steamworks + Steamworks.NET","github.com/rlabrecque/Steamworks.NET","API Steam: conquistas, inventário, trading"),
 r"Slash":("Slash Framework DataBind","github.com/SlashGames/slash-framework","data-binding de UI (MVVM Unity)"),
 r"CodeStage|AntiCheat":("CodeStage Anti-Cheat Toolkit","assetstore.unity.com (API pública)","ObscuredTypes + detectores de tamper/speedhack"),
 r"CodeMonkey":("CodeMonkey Utils","youtube.com/CodeMonkeyUnity","utilidades de UI/grid"),
 r"EasySave|ES3":("Easy Save 3","moodkie.com/easysave (API pública)","save cifrado (AES) em disco"),
 r"Cinemachine":("Unity Cinemachine","docs.unity3d","câmera"),
 r"TextMeshPro|TMPro":("TextMeshPro","docs.unity3d","texto")}
print("MIDDLEWARE DE FONTE PÚBLICA (identificado → não precisa de RE, é código conhecido):\n")
revealed=0
for a,k in asm.most_common():
    for pat,(nome,src,papel) in LIB.items():
        if re.search(pat,a,re.I):
            print(f"  ✓ {k:5d} classes · {a}\n      = {nome}  ·  {papel}\n      fonte: {src}")
            revealed+=k; break
print(f"\n→ {revealed} classes 'reveladas' por identificação (código público conhecido).")
print(f"→ resta o proprietário: Assembly-CSharp TaskbarHero.* (563) — o Claude de-ofusca por contexto (tbh_annotate).")
