"""wtrans — word-transition surprisal, a measurable proxy for text coherence.

Motivation: image codecs have measurable quality (PSNR/SSIM). Can text coherence be scored the
same way? `wtrans` is the cleanest answer we found. Fit a word-bigram model on a reference corpus,
then score a candidate text by its mean transition surprisal:

    wtrans(text) = mean_i [ -log P(word_{i+1} | word_i) ]

Real Portuguese prose scores ~6; word-salad / gibberish scores ~10-11. Unlike character-level
perplexity it is robust to capitalization and catches *semantic* incoherence (a sentence of real
words in an impossible order). The training loop uses it to validate its own samples each cycle.
"""
import math
import re
from collections import Counter

import numpy as np

# Latin letters incl. the accented characters used in Portuguese.
_WORD = re.compile(r"[a-zàáâãéêíóôõúüç]+")


class WordTransition:
    def __init__(self, reference_text: str, alpha: float = 0.05):
        words = _WORD.findall(reference_text.lower())
        if len(words) < 2:
            raise ValueError("reference_text is too short to fit a bigram model")
        self.unigram = Counter(words)
        self.bigram = Counter(zip(words, words[1:]))
        self.vocab = len(self.unigram)
        self.alpha = alpha  # additive smoothing

    def score(self, text: str) -> float:
        """Mean transition surprisal in nats. Lower = more coherent. Returns 12.0 for <3 words."""
        w = _WORD.findall(text.lower())
        if len(w) < 3:
            return 12.0
        a, v = self.alpha, self.vocab
        surprisals = [
            -math.log((self.bigram.get((w[i], w[i + 1]), 0) + a) / (self.unigram.get(w[i], 0) + a * v))
            for i in range(len(w) - 1)
        ]
        return float(np.mean(surprisals))

    def transition_surprisal(self, prev_word: str, word: str) -> float:
        """-log P(word | prev_word): the coherence cost of following prev_word with word.
        Lower = more coherent. Used by the coherence-guided decoder to rank candidate words."""
        if not prev_word or not word:
            return 6.0
        a, v = self.alpha, self.vocab
        return -math.log((self.bigram.get((prev_word, word), 0) + a) / (self.unigram.get(prev_word, 0) + a * v))
