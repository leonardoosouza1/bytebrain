# Architecture battery — solving the coherence collapse

A flat byte-level LM writes good local Portuguese but **degrades after ~30 words** into repetition
and word-salad. This battery asks: *can we fix that by being smarter, not just bigger?* Every
experiment is a fair, measured comparison on the **same trained 26M weights** (or matched budgets),
scored by `wtrans` (word-transition surprisal) and **coherent span** (how many words before a
sliding coherence window first breaks).

## Results

| # | Experiment | Idea | Result |
|---|-----------|------|--------|
| A | `exp_a_hier_vs_flat.py` | Hierarchical / MEGABYTE-lite (transformer over byte *patches*) | ❌ Lost to flat twice, even at equal wall-clock. Hierarchy needs scale; at ~14M / 5MB the local-decoder bottleneck dominates. |
| D | `exp_d_coherence_guided_decode.py` | Draft K next-words, keep the one with best fluency − word-transition surprisal | ✅ **Winner. Coherent span 16 → 32 words, wtrans 9.6 → 8.0, same weights, no retraining.** |
| D2 | `exp_d2_true_rollback.py` | Delete recent words and retry when coherence breaks | ❌ Thrashed — brittle word-bigram threshold + raising temperature on retry. |
| E | `exp_e_entropy_gated.py` | Use the model's own entropy as the drift detector | ❌ Backwards: the repetition collapse is **low** entropy (confident garbage); real creativity is **high** entropy. Confidence ≠ coherence. |
| F | `exp_f_constrained_words.py` | Trie-constrain generation to real corpus words | ❌ Kills byte-garbage but not word-salad — the bottleneck was order, not garbage. |
| G | `exp_g_topic_coherence.py` | Long-range topic verifier (PMI with last ~12 content words) | ❌ Signal too sparse to beat the local bigram; can't reconstruct coherence the model lacks. |

## Conclusion

**Coherence-guided decoding (D) is the robust win** — it roughly doubles the coherent span for
free and is now the production decoder (`src/sample.coherence_guided_generate`). Everything fancier
(rollback, entropy, constraints, topic models) either thrashed or failed to beat it.

The deeper lesson: once the model's own next-byte distribution turns to garbage, **no external
verifier can rebuild coherence the model doesn't have** — it can only pick the least-bad of K bad
drafts. So past ~32 words the ceiling is the **model**, not the decoder. The two validated levers
to raise it are (1) **more data** (corpus size vs val-bpb correlate −0.90) and (2) **scale**
(more params / longer context — the AMD MI300X path). Smart decoding and a bigger model compound;
they are not substitutes.
