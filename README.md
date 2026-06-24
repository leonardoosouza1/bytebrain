# ByteBrain

**A tokenizer-free, byte-level language model — trained from scratch on a consumer AMD GPU.**

ByteBrain has no tokenizer and no BPE vocabulary. Its entire vocabulary is the **256 possible
byte values**. Every string — in any language or script — is already a sequence of UTF-8 bytes,
so the model needs no preprocessing, has **zero out-of-vocabulary tokens**, and learns the
structure of text directly from the rawest possible representation.

This repository contains the model, the self-validating training loop, the data pipeline, and the
evaluation tools used to validate the idea on Portuguese — running entirely on a single
**AMD Radeon RX 6750 XT** (RDNA2, ROCm), a desktop gaming GPU.

> Built for the **AMD Developer Hackathon — Act II** (Track 3). The companion routing project
> lives in [`../iara-router`](../iara-router).

---

## Why bytes?

| | BPE / token models | ByteBrain (bytes) |
|---|---|---|
| Vocabulary size | 32k–150k+ | **256** |
| Embedding table | huge | **~600× smaller** |
| Out-of-vocabulary | yes (rare scripts, typos) | **0% — impossible by construction** |
| New language | often needs retraining | **works unchanged** |
| Robustness to noise/typos | brittle | **+16 pp under noise** (measured) |

The trade-off is sequence length — bytes are finer-grained than tokens, which makes long-range
coherence harder. That trade-off is exactly what this project measures and pushes on.

---

## Results

Every number below is **measured on the hardware listed, not extrapolated**.

### Training (8M-param model, validated run)
- Trained **from scratch** on a growing, cleaned Portuguese corpus on the RX 6750 XT.
- 100 training cycles / ~6 h wall-clock, **zero overfitting** (see *Self-validating loop*).
- Validation loss: **1.84 bits/byte**.
- The model writes **grammatically coherent Portuguese** — correct agreement, dates, place names —
  then degrades into repetition after ~2–3 sentences. That degradation is a capacity limit at 8M,
  not a knowledge gap: the *prefix* of every sample is real Portuguese.

A **26M-param model is training at the time of writing** to extend the coherent span.

### Generation quality (nucleus + repetition-penalty sampling)
Replacing plain temperature sampling with **nucleus (top-p) sampling plus a recent-byte
repetition penalty** drops the word-transition surprisal (`wtrans`, see below) from **9.0 → 7.9**
and roughly doubles the coherent span. It directly kills the degenerate `"e e e"` collapse that
small byte-level models fall into.

Sample (8M model, `temperature=0.6, top_p=0.85`):

> *"O ex-mandioca, os artigos de Almeida pela Nicarágua. Em 1956, foi inaugurado a 18 de setembro
> de 2017. Em 1953, foi convencido para a Costa da Segunda-feira."*

### Structural findings (the internal structure of a byte)
- **Bit 5 of each ASCII byte encodes case** (lower ↔ UPPER) — confirmed on-device with
  KL ≈ 0.003. First numerical evidence that the 8 bits of a byte carry learnable structure.
- A single byte-level model represents **8 different scripts** at ~1.8 bits/byte, with 0% OOV.

---

## Coherence as a measurable signal — `wtrans`

A guiding question of this project: *can text coherence be measured the way image quality can?*

ByteBrain ships **`wtrans` (word-transition surprisal)** — the mean
`-log P(wordᵢ | wordᵢ₋₁)` under a word-bigram model fit on a reference corpus. Real Portuguese
scores ~6; gibberish ~10–11. It is robust to capitalization and catches the semantic *word-salad*
band that character-level perplexity misses — so it is the metric the training loop uses to
**validate its own generations automatically**, every cycle.

---

## Self-validating training loop

`overnight_loop.py` trains unattended for hours and **guards its own health** so a long run can
never silently collapse:

- **Self-cleaning data.** Each cycle rebuilds a *cleaned* corpus as new data arrives. A strict
  Portuguese-prose filter (`src/data.py`) rejects code, markup, English, and fragments.
- **Regularized, bounded cycles.** Dropout 0.15, weight decay 0.1, and a hard cap of 2 epochs
  per cycle — the failure mode that originally caused catastrophic memorization was ~80 epochs
  on a small corpus.
- **Automatic overfit correction.** After every cycle it measures validation loss and `wtrans`.
  If validation loss **rises** (the signature of overfitting) for two consecutive cycles, it
  **reverts to the best checkpoint and lowers the learning rate** — with no human in the loop.
- **Best-by-generalization checkpointing.** It always keeps the best checkpoint by validation,
  never just the latest.

In the validated run this kept **100 consecutive cycles overfit-free**.

---

## Architecture

A compact decoder-only transformer over raw bytes (`src/model.py`):

- Input/output vocabulary: **256** (one logit per byte value).
- Token + learned positional embeddings; **256 bytes** of context.
- Pre-norm transformer blocks, multi-head self-attention via fused SDPA (causal), GELU MLP.
- Dropout on attention, MLP, and residual paths.
- Default `dim=512, layers=8, heads=8` ≈ **26M** parameters; the validated run used `dim=384`-class
  configs in the 8–11M range.

Generation uses nucleus sampling with a repetition penalty over the last 48 bytes (`src/sample.py`).

---

## Hardware & reproducibility

- **GPU:** AMD Radeon RX 6750 XT — 12 GB, RDNA2, `gfx1032`.
- **ROCm 7.x.** The card needs `HSA_OVERRIDE_GFX_VERSION=10.3.0` to be recognized.
- **Wave size 32** (RDNA2; wave64 does not work here).
- **fp32 on device.** RDNA2 has no bf16 matrix units — bf16 is dramatically slower on this GPU,
  so training is fp32 without activation checkpointing.

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 MIOPEN_FIND_MODE=2
```

---

## Quickstart

```bash
pip install -r requirements.txt   # torch (ROCm build) + numpy

# Sample from a trained checkpoint
python -m examples.sample --checkpoint overnight_ck/loop_best.pt --prompt "O "

# Train (self-validating loop)
python overnight_loop.py
```

See [`examples/sample_outputs.md`](examples/sample_outputs.md) for real generations.

---

## Repository layout

```
src/
  model.py        # ByteGPT — the byte-level transformer (+ legacy checkpoint loader)
  sample.py       # nucleus + repetition-penalty generation
  coherence.py    # wtrans — word-transition surprisal coherence metric
  data.py         # Portuguese-prose corpus filter & builder
overnight_loop.py # self-validating, self-correcting training loop
examples/         # runnable sampling demo + saved outputs
```

Research/experimental scripts (robustness, byte-vs-token, coherence batteries, fetchers) are kept
at the repository root and will be consolidated under `research/` once the current training run
completes.

---

## Roadmap

- **Scale up** the model on AMD MI300X (AMD Developer Cloud) to extend coherence past a few sentences.
- **Byte-MoE:** per-language experts behind a byte-level router (ties into the IARA-Router project).
- Greedy/beam decoding and a small instruction-tuned variant.

---

## License

MIT — see [LICENSE](LICENSE).
