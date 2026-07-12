
# BATERIA 6 — reversal cure + raciocínio — (re)início Sun Jul  5 05:57:01 AM -03 2026


========== RUN bytebrain/research/marco133_reversal_cure.py (05:57:01) ==========
Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]Loading weights:   0%|          | 1/338 [00:00<03:02,  1.85it/s]Loading weights:  51%|█████     | 171/338 [00:00<00:00, 342.71it/s]Loading weights:  76%|███████▌  | 256/338 [00:00<00:00, 349.75it/s]Loading weights:  94%|█████████▍| 319/338 [00:01<00:00, 295.19it/s]Loading weights: 100%|██████████| 338/338 [00:01<00:00, 280.35it/s]
[    14s] M133 START | Math-1.5B
[   966s] === reversa INÉDITA: só-forward 0/7 (0.0) vs bidirecional 3/7 (0.43) ===
[   966s] (BIDIRECIONAL CURA o reversal curse)
[   966s] DONE M133 (966s)
>>> bytebrain/research/marco133_reversal_cure.py OK (06:13:11)


========== RUN bytebrain/research/marco134_reasoning_seed.py (06:13:11) ==========
Loading weights:   0%|          | 0/338 [00:00<?, ?it/s]Loading weights:  22%|██▏       | 75/338 [00:00<00:00, 729.99it/s]Loading weights:  53%|█████▎    | 179/338 [00:00<00:00, 905.64it/s]Loading weights:  80%|███████▉  | 270/338 [00:00<00:00, 804.57it/s]Loading weights: 100%|██████████| 338/338 [00:00<00:00, 858.52it/s]
[    13s] M134 START | Math-1.5B
[   331s]   K=4: treino 30/30 | generaliza silogismo INÉDITO 20/20 (1.0) | baseline 4
[   690s]   K=8: treino 30/30 | generaliza silogismo INÉDITO 20/20 (1.0) | baseline 4
[  1108s]   K=16: treino 30/30 | generaliza silogismo INÉDITO 20/20 (1.0) | baseline 4
[  1108s] === VEREDITO: baseline 4/20 → melhor semente 1.0 (SEMENTE CARREGA RACIOCÍNIO) ===
[  1108s] DONE M134 (1108s)
>>> bytebrain/research/marco134_reasoning_seed.py OK (06:31:43)

# FIM BATERIA 6 — Sun Jul  5 06:31:43 AM -03 2026
