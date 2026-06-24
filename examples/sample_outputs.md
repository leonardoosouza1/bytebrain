# Sample outputs

Real, unedited generations from ByteBrain checkpoints trained from scratch on an AMD RX 6750 XT.
The model is byte-level and tokenizer-free; the only prompt is the seed `"O "`.

Note the pattern across every sample: the **prefix is coherent Portuguese** — correct grammar,
agreement, dates, place names — and then it degrades into repetition. That degradation is the
capacity ceiling of a small model on byte-level sequences, not a lack of learned language. Better
sampling (nucleus + repetition penalty) and a larger model both push the coherent span further.

---

### 8M model — nucleus sampling (`temperature=0.6, top_p=0.85, rep_penalty=1.4`)
`wtrans` ≈ 7.9

> O ex-mandioca, os artigos de Almeida pela Nicarágua. Em 1956, foi inaugurado a 18 de setembro de
> 2017. Em 1953, foi convencido para a Costa da Segunda-feira. *[...degrades...]*

### 8M model — temperature sampling (`temperature=0.8`)
> O excesso de emissões de gases fundamentais da energia nuclear. As formas de energia cinéticas
> revelam a razão coesa sobre os filamentos, particularmente as *[...degrades...]*

### 8M model — temperature sampling (`temperature=0.9`)
> O dos Padrões Unidos em 1954 e a Previsão do Brasil em São Paulo, com vendas áreas divididas entre
> as partes de Versalhy, entre 1944 e 1914 e 1995. *[...degrades...]*

---

A 26M-parameter model is training to extend the coherent span; outputs will be added here as it
converges.
