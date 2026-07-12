"""Loader do IARA-mini (modelo com intermediate_size POR CAMADA — cirurgia física).
Uso: from iara_mini_loader import load_iara_mini; model = load_iara_mini(dir, device)"""
import json, os, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file

def load_iara_mini(model_dir, device="cuda"):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    profile = cfg["iara_carve_profile"]                      # lista: intermediate por camada
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    for L, k in enumerate(profile):                          # redimensiona as MLPs ANTES de carregar
        mlp = model.model.layers[L].mlp
        h = config.hidden_size
        mlp.gate_proj = nn.Linear(h, k, bias=False, dtype=torch.float16)
        mlp.up_proj   = nn.Linear(h, k, bias=False, dtype=torch.float16)
        mlp.down_proj = nn.Linear(k, h, bias=False, dtype=torch.float16)
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if cfg.get("tie_word_embeddings"): model.tie_weights()
    assert not unexpected, f"pesos inesperados: {unexpected[:3]}"
    return model.to(device).eval()
