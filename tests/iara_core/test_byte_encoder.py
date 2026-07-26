import torch
from iara_core.byte_encoder import ByteEncoder

def test_shape():
    torch.manual_seed(0)
    e = ByteEncoder(d=64)
    assert e(["capital do brasil", "oi"]).shape == (2, 64)

def test_typo_fica_perto_do_original():
    """Byte-level: 1 typo muda pouco a representação (a tese da robustez)."""
    torch.manual_seed(0); e = ByteEncoder(d=64).eval()
    with torch.no_grad(): v = e(["capital do brasil","capitl do brasil","xyz qwerty zzz"])
    sim_typo  = torch.cosine_similarity(v[0:1], v[1:2]).item()
    sim_outro = torch.cosine_similarity(v[0:1], v[2:3]).item()
    assert sim_typo > sim_outro

def test_transformer_esta_ligado():
    """Guarda contra o encoder virar um saco-de-embeddings: a saída tem que diferir
    da média crua de (emb+pos). Pega o caso de alguém esquecer de chamar self.enc."""
    torch.manual_seed(0); e = ByteEncoder(d=64).eval()
    txt = ["capital do brasil"]
    with torch.no_grad():
        saida = e(txt)
        b = list(txt[0].encode("utf-8"))
        x = torch.tensor(b)[None]
        cru = (e.emb(x) + e.pos(torch.arange(len(b)))[None]).mean(1)
    assert (saida - cru).abs().mean() > 1e-3, "a saída é a média crua: o Transformer não está no caminho"
