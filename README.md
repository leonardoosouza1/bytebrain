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

> 📊 **Full results, every number, and all ~10 documented dead ends:**
> [`FINDINGS.md`](FINDINGS.md) (results + failures + changelog) ·
> [`research/architecture_battery/README.md`](research/architecture_battery/README.md) (experiment methodology) ·
> [`overnight_journal.md`](overnight_journal.md) (live training log).

### Training (40M-param model, validated run)
- Trained **from scratch** on **1.33 GB** of clean Portuguese Wikipedia on the RX 6750 XT.
- ~16 h wall-clock, validation **1.47 → 1.288 bits/byte**, **zero overfitting** (val stays *below*
  train — the model never runs out of fresh data).
- It writes **fluent, grammatically correct Portuguese** — agreement, dates, sentence structure —
  but still rambles and invents facts over long spans. That gap is **global** coherence: a capacity
  ceiling, not a local one (see the battery below).

An **86M-param model is training as of this writing** (gradient checkpointing + fp16) to push the
ceiling further.

### Generation quality — coherence-guided decoding
**Coherence-guided decoding** (draft K candidate words, keep the one maximising model fluency minus
its word-transition surprisal `wtrans`) roughly **doubles the coherent span, 16 → 32 words**, on the
same weights with no retraining.

Sample (40M model):

> *"O Brasil em uma tentativa de população... a Assembleia Nacional de Relações Extraordinárias, que
> se formou na cidade de São Paulo, em 2013."*

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

### 1 · Clone & install

```bash
git clone https://github.com/<your-user>/bytebrain.git
cd bytebrain
pip install -r requirements.txt        # numpy + torch (use a ROCm build on AMD)
```

### 2 · Build the dataset

The corpus is **not** shipped (it is ~1.3 GB). Regenerate it from the public Wikipedia dump — the
pipeline downloads, stream-parses the bz2 XML, strips wiki markup, and keeps only clean Portuguese
prose via the `is_portuguese_prose` filter:

```bash
mkdir -p data/dumps
wget -c https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2 \
     -O data/dumps/ptwiki-latest-pages-articles.xml.bz2   # ~2.6 GB

python build_wiki_corpus.py            # -> data/pt_big.txt  (~1.3 GB clean prose)
```

For another language, point the dump URL at that wiki and swap the prose filter — the byte model
itself is language-agnostic.

### 3 · Train (power-loss safe, resumable)

```bash
# AMD RDNA2 (RX 6700/6750 XT) env — force the gfx override:
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 MIOPEN_FIND_MODE=2

python train.py --corpus data/pt_big.txt --ckpt-dir ckpt \
    --dim 640 --layers 8 --heads 8 --ctx 512 \
    --batch 24 --accum 4 --amp --decay-steps 50000
```

`--amp` uses fp16 (RDNA2 runs fp16 at ~2× fp32). Re-run the **exact same command** to resume from
the last atomic checkpoint after a crash or power cut. To train a larger model on 12 GB, add
`--checkpoint` (gradient checkpointing) and bump `--dim/--layers` (e.g. `--dim 768 --layers 12`).

### 4 · Generate

```bash
python -m examples.sample --checkpoint ckpt/ckpt_best.pt \
    --dim 640 --layers 8 --heads 8 --prompt "O "
```

Pass the same `--dim/--layers/--heads` you trained with. See
[`examples/sample_outputs.md`](examples/sample_outputs.md) for real generations.

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
research/         # exploratory scripts behind the results (see research/README.md)
  coherence/      #   the search for a measurable coherence signal
  robustness/     #   byte-vs-token robustness experiment
  data_pipeline/  #   corpus construction, cleaning, polite fetchers
  model_variants/ #   architecture & scale experiments
results/          # training metrics from runs (evidence)
```

The exploratory scripts that produced the results above live under
[`research/`](research/README.md), grouped by theme. The clean, importable package is
[`src/`](src); the canonical trainer is `overnight_loop.py`.

---

## Roadmap

- **Scale up** the model on AMD MI300X (AMD Developer Cloud) to extend coherence past a few sentences.
- **Byte-MoE:** per-language experts behind a byte-level router (ties into the IARA-Router project).
- Greedy/beam decoding and a small instruction-tuned variant.

---

## License

MIT — see [LICENSE](LICENSE).
