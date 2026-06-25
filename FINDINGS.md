# Findings & changelog — ByteBrain

An honest, numbers-first log of this PoC: what worked, what didn't, and every measurement behind it.
Dead ends are listed on purpose — in research they *are* the result. Methodology behind each
experiment is in [`research/architecture_battery/README.md`](research/architecture_battery/README.md);
the live training journal is in [`overnight_journal.md`](overnight_journal.md).

---

## ✅ What worked

| Finding | Measured |
|---|---|
| Tokenizer-free, byte-level model | vocab **256** (vs ~150k BPE) · **0% out-of-vocabulary**, any script |
| 40M model trained from scratch (RX 6750 XT) | val **1.47 → 1.288 bits/byte**, ~16 h, **zero overfitting** |
| Coherence-guided decoding | coherent span **16 → 32 words**, same weights, no retraining |
| More data lowers loss (no overfit) | held-out bpb **2.43 → 2.29** (5 MB → 843 MB); corpus↔bpb corr **−0.90** |
| Bit 5 of each byte encodes case | KL ≈ **0.003** — first numerical evidence the thesis holds |
| `wtrans` coherence metric | real PT ≈ **6**, gibberish ≈ **10–11** (separates the bands) |
| fp16 packed math on RDNA2 | **~1.7× faster** than fp32 (8.2k → 13.9k tok/s) |
| One byte-model over 8 scripts | ~1.8 bpb each, 0% OOV |

## ❌ What didn't (dead ends — all documented)

| Attempt | Outcome | Why |
|---|---|---|
| bf16 mixed precision | **80× slower** | RDNA2 has no bf16 units — it emulates in software |
| Hierarchy (MEGABYTE-lite) | lost to flat (**val 3.1 vs 2.2**) | local-decoder bottleneck at this scale |
| Unlikelihood training | no gain → reverted | byte window too short for phrase-level loops |
| Entropy-gated decoding | **worse** (span 11 vs 16) | backwards signal — the collapse is *low* entropy |
| Real-word trie constraint | no net gain | kills byte-garbage, not word-salad |
| Topic / PMI verifier | no gain | co-occurrence signal too sparse |
| True rollback decoding | thrashed | brittle threshold + bad retry policy |
| Discriminator as critic | can't critique | model is locally **indistinguishable** from real PT |
| First training run | overfit: **train 0.22 / val 5.0** | memorized a tiny corpus → built a self-validating loop |
| wikiextractor | crashed on Python 3.12 | wrote a dependency-free bz2 streaming parser |

**Tally:** ~8 wins, ~10 documented dead ends. The recurring lesson: local tricks plateau; the lever is scale.

---

## 🗓️ Timeline

1. **Thesis validation** — bytes give 0% OOV, +16 pp robustness under noise vs token-level, and bit 5 = case (KL 0.003).
2. **First runs** — 8M overnight model reaches val ~1.84; hits overfitting → self-validating training loop.
3. **Architecture battery (A–G)** — 7 experiments against the coherence ceiling; coherence-guided decoding wins, the rest are marginal/negative.
4. **Dataset** — custom bz2 parser builds **1.33 GB** of clean Portuguese Wikipedia.
5. **40M run** — val **1.47 → 1.288**, ~16 h, zero overfit; writes fluent (still rambling) PT.
6. **Hardware push** — fp16 packed math (1.7×) + gradient checkpointing → **86M model scaling up** to chase the ceiling.

> The frontier now is global coherence — a capacity/scale limit, not the idea. Next: AMD MI300X.
