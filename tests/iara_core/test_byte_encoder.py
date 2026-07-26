import torch
from iara_core.byte_encoder import ByteEncoder

def test_shape():
    e = ByteEncoder(d=64)
    assert e(["capital do brasil", "oi"]).shape == (2, 64)

def test_typo_fica_perto_do_original():
    """Byte-level: 1 typo muda pouco a representação (a tese da robustez)."""
    torch.manual_seed(0); e = ByteEncoder(d=64).eval()
    with torch.no_grad(): v = e(["capital do brasil","capitl do brasil","xyz qwerty zzz"])
    sim_typo  = torch.cosine_similarity(v[0:1], v[1:2]).item()
    sim_outro = torch.cosine_similarity(v[0:1], v[2:3]).item()
    assert sim_typo > sim_outro
