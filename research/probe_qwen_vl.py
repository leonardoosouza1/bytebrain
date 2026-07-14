#!/usr/bin/env python3
"""PROBE — Qwen2.5-VL-3B cabe e serve como córtex+olho unificado? (decide a frente B)
Checa: (1) VRAM, (2) submódulos p/ o grafo de pesos (embed_tokens/norm/down_proj),
(3) raciocínio só-texto, (4) leitura de imagem com texto/tabela."""
import torch,time,re
from transformers import AutoProcessor,AutoModelForImageTextToText,AutoTokenizer
from PIL import Image
ID="Qwen/Qwen2.5-VL-3B-Instruct"
t=time.time(); print(f"[carregando {ID} …]",flush=True)
proc=AutoProcessor.from_pretrained(ID)
m=AutoModelForImageTextToText.from_pretrained(ID,dtype=torch.float16).to("cuda").eval()
print(f"[carregado em {time.time()-t:.0f}s] classe={m.__class__.__name__}")
print("VRAM usada GB:",round(torch.cuda.memory_allocated()/1e9,2),"/ 12.9")

# (2) achar submódulos para o grafo de pesos
mods=dict(m.named_modules())
emb=[n for n in mods if n.endswith("embed_tokens")]
nrm=[n for n in mods if n.endswith(".norm") or n=="model.norm"]
dpr=[n for n in mods if n.endswith("mlp.down_proj")]
print("embed_tokens:",emb[:3]); print("norm final:",[n for n in nrm if "layers" not in n][:3]); print("down_proj (n):",len(dpr),"ex:",dpr[-1] if dpr else None)
try:
    E=m.get_input_embeddings().weight; print("embeddings shape:",tuple(E.shape))
except Exception as e: print("get_input_embeddings falhou:",e)

# (3) raciocínio só-texto
tok=AutoTokenizer.from_pretrained(ID)
msgs=[{"role":"user","content":[{"type":"text","text":"Qual a capital da França? Responda em 1 frase."}]}]
txt=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
ins=proc(text=[txt],return_tensors="pt").to("cuda")
out=m.generate(**ins,max_new_tokens=40,do_sample=False)
print("TEXTO →",proc.batch_decode(out[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].strip())

# (4) leitura de imagem com texto/tabela
img=Image.open("/tmp/tabela_teste.png").convert("RGB")
msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":"Leia em detalhe o título, as colunas e todos os números desta tabela."}]}]
txt=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
ins=proc(text=[txt],images=[img],return_tensors="pt").to("cuda")
out=m.generate(**ins,max_new_tokens=200,do_sample=False)
print("VISÃO →"," ".join(proc.batch_decode(out[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].split()))
print("VRAM pico GB:",round(torch.cuda.max_memory_allocated()/1e9,2))
