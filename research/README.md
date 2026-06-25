# Research scripts

Exploratory scripts behind ByteBrain. These are research notebooks-as-scripts — they are how the
results in the top-level README were found. They are kept for reproducibility and are not part of
the clean importable package in [`../src`](../src). Most are standalone (`python <script>.py`).

### `coherence/`
The search for a measurable coherence signal for text — the experiments that led to `wtrans`
(now in [`../src/coherence.py`](../src/coherence.py)).
- `coherence_wild.py` — wide sweep of candidate measures; where `wtrans` was discovered.
- `coherence_signal.py` — treating coherence as a signal (periodicity, autocorrelation, total variation).
- `coherence_battery.py`, `coheref.py` — composite coherence score and its calibration.
- `coherence_model.py` — a small learned byte-CNN coherence classifier.
- `coherence_cotrain.py` — adversarial generator/discriminator co-training (an honest negative result).
- `battery_validators.py` — validating that `wtrans` tracks real corpus quality.

### `robustness/`
- `byte_vs_token_robustness.py` — the experiment showing byte-level inputs degrade far less than
  token-level under noise/typos (+16 pp), and reach 0% out-of-vocabulary on unseen scripts.

### `data_pipeline/`
Corpus construction and cleaning for Portuguese.
- `clean_and_filter.py`, `clean_train_test.py` — the prose filter and the clean-corpus training test.
- `build_pt_corpus.py`, `build_pt_local.py` — corpus assembly.
- `fetch_mediawiki.py`, `fetch_nlm.py`, `fetch_nlm_loop.py` — polite fetchers for Wikipedia /
  Wikisource / NotebookLM that grow the corpus during training.

### `model_variants/`
Architecture and scale experiments: `bytebrain_big.py`, `bytebrain_deep.py`, `bytebrain_scale.py`,
`bytebrain_multiscript.py` (one model over 8 scripts), `bytebrain_unified_lm.py`,
`bytebrain_overnight.py`, `bytebrain_best_gen.py`, `bytebrain_coscientist.py`.

> **Note:** these are the actual exploration scripts, kept for transparency and reproducibility — not polished CLIs. Several contain machine-specific absolute paths. The runnable, clean entry points are at the repo root (`train.py`, `build_wiki_corpus.py`, `examples/`).
