#!/usr/bin/env python3
"""TBH-DUMP — parser Python PURO do global-metadata.dat (IL2CPP v31) do TaskbarHero.
Reconstrói a ARQUITETURA REAL: assemblies → namespaces → classes → {campos, métodos}.
Sem .NET, sem dependências (só stdlib) → roda em qualquer máquina Linux CPU-only.
Uso: python3 tbh_dump.py [caminho_do_jogo]  → gera tbh_map.json"""
import struct, sys, json, os

GAME = sys.argv[1] if len(sys.argv)>1 else "/home/leonardo/.steam/debian-installation/steamapps/common/TaskbarHero"
META = os.path.join(GAME,"TaskBarHero_Data","il2cpp_data","Metadata","global-metadata.dat")
b = open(META,"rb").read()
H = struct.unpack_from("<66I", b, 0)
assert H[0]==0xFAB11BAF, "magic inválido"
VER=H[1]; print(f"[metadata v{VER} · {len(b)/1e6:.1f}MB]")
LAB=["stringLiteral","stringLiteralData","string","events","properties","methods","paramDefaults","fieldDefaults","fieldParamDefaultData","fieldMarshaledSizes","parameters","fields","genericParams","genericParamConstraints","genericContainers","nestedTypes","interfaces","vtableMethods","interfaceOffsets","typeDefinitions","images","assemblies"]
T={LAB[i]:(H[2+i*2],H[2+i*2+1]) for i in range(len(LAB))}

def cstr(off):                                   # string terminada em \0 a partir do offset da tabela 'string'
    base=T["string"][0]+off; e=b.index(b"\x00",base); return b[base:e].decode("utf-8","replace")

# ---- typeDefinitions (88 bytes) ----
tdo,tds=T["typeDefinitions"]; NT=tds//88
# ---- methods (36 bytes v31) ----
mo,ms=T["methods"]; NM=ms//36
# ---- fields (12 bytes) ----
fo,fs=T["fields"]; NF=fs//12
# ---- images/assemblies (mapeiam type range → assembly) ----
ao,asz=T["assemblies"]; io,isz=T["images"]
print(f"[{NT} tipos · {NM} métodos · {NF} campos]")

# pré-lê métodos e campos (só nome + declaringType) — leve
M_name=[0]*NM
for i in range(NM):
    M_name[i]=struct.unpack_from("<i",b,mo+i*36)[0]           # nameIndex
F_pair=[(0,0)]*NF
for i in range(NF):
    ni,ti=struct.unpack_from("<ii",b,fo+i*12)                # nameIndex,typeIndex
    F_pair[i]=(ni,ti)

# images: nameIndex, assemblyIndex, typeStart, typeCount, ... (Il2CppImageDefinition v31: 10 uint = 40 bytes? valida por divisão)
img_sz = isz // max(1,(isz//40 if isz%40==0 else isz//40))
# assemblies: aponta p/ imageIndex; em vez de mapear exato, uso images: typeStart/typeCount
# Il2CppImageDefinition: nameIndex(int), assemblyIndex(int), typeStart(int), typeCount(uint), exportedTypeStart(int), exportedTypeCount(uint), entryPointIndex(int), token(uint), customAttributeStart(int), customAttributeCount(int) = 40 bytes
NI = isz//40
images=[]
for i in range(NI):
    v=struct.unpack_from("<10i",b,io+i*40)
    images.append(dict(name=cstr(v[0]),typeStart=v[2],typeCount=v[3]))
# type -> assembly(image) por range
t2img={}
for im in images:
    for t in range(im["typeStart"],im["typeStart"]+im["typeCount"]):
        t2img[t]=im["name"]

# ---- reconstrói cada tipo ----
classes=[]
for i in range(NT):
    off=tdo+i*88
    v=struct.unpack_from("<16i8H2I",b,off)
    nameIndex,nsIndex=v[0],v[1]; fieldStart,methodStart=v[8],v[9]
    method_count,property_count,field_count=v[16],v[17],v[18]
    name=cstr(nameIndex); ns=cstr(nsIndex)
    methods=[]
    if 0<=methodStart and method_count<20000:
        for k in range(methodStart,methodStart+method_count):
            if 0<=k<NM: methods.append(cstr(M_name[k]))
    fields=[]
    if 0<=fieldStart and field_count<20000:
        for k in range(fieldStart,fieldStart+field_count):
            if 0<=k<NF: fields.append(cstr(F_pair[k][0]))
    classes.append(dict(ns=ns,name=name,asm=t2img.get(i,"?"),
                        fields=fields,methods=methods,nm=method_count,nf=field_count))

# ---- salva ----
out=dict(version=VER,ntypes=NT,nmethods=NM,nfields=NF,
         assemblies=sorted({c["asm"] for c in classes}),classes=classes)
outp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tbh_map.json")
json.dump(out,open(outp,"w"),ensure_ascii=False)
print(f"[salvo {outp} · {len(classes)} classes]")

# ---- validação: mostra uma classe-chave ----
for alvo in ("AccountSaveData","SaveSecret","ActBoss"):
    c=next((c for c in classes if c["name"]==alvo),None)
    if c:
        print(f"\n### {c['asm']} :: {c['ns']+'.' if c['ns'] else ''}{c['name']}  ({c['nf']} campos, {c['nm']} métodos)")
        print("  campos:",", ".join(c["fields"][:12]))
        print("  métodos:",", ".join(c["methods"][:14]))
