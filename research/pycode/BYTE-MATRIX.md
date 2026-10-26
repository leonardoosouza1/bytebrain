# Code as a byte matrix (ledger #6, #9) — 2026-10-26

His idea (10-22): "converte o código em matrizes de bytes... cada matriz é uma linha ou coluna, ou ambas, ou tratar
como matriz inteira de X linhas e Y colunas, se eu pegar linha 1, pega x0 e multiplicar com os outros, o que daria?
o que iria indicar que uma identação nasce? uma quebra de linha?" and #9: use the forest / water image models as a
lens on code.

Script: `byte_matrix_probe.py` (numpy, CPU, 4 threads, ~1 min). Command:

```bash
OMP_NUM_THREADS=4 nice -n 5 .venv-rocm/bin/python -u byte_matrix_probe.py --n 4000 \
  --out /media/llm-data/universe-runs/bytebrain-pycode-20261022/byte_matrix
```

Output: `clean.json`, `raw.json`, `example_{clean,raw}.png` in that directory.

Data:
- 4000 functions per corpus, from `clean_train.txt` (ast.unparse, no comments) and `raw_train.txt` (as written).
- Functions are split on a blank line before `@`, `def ` or `async def `.
- Functions are kept only if they have 3–40 rows, at most 100 bytes per row and no tabs. This kept 4000 of 8556
  clean and 4000 of 6249 raw functions examined. The filter favours short functions; declared.
- Grid G[row, column] = byte value, padded with spaces to 100 columns. Ink map = G != space.

Pair statistics use each non-blank row and the next non-blank row: 29,948 pairs clean, 43,121 raw.

## A. What says an indent is born / dies (row r → next row)

Mutual information, in bits, between one feature of row r and the event:

| feature of row r | born, clean | born, raw | dies, clean | dies, raw |
|---|---|---|---|---|
| H(event) (the ceiling) | 0.977 | 0.889 | 0.740 | 0.659 |
| x0 (first byte, his "x0") | 0.193 | 0.195 | 0.049 | 0.029 |
| first non-space byte | 0.649 | 0.392 | 0.238 | 0.116 |
| **last non-space byte** | **0.977** | 0.722 | 0.199 | 0.105 |
| first word | 0.977 | 0.666 | 0.477 | 0.298 |

- **Birth is one byte.** Rule "row ends with `:`" → next row is indented deeper:
  - clean: precision 1.000, recall 1.000 (12,321 / 12,321);
  - raw: precision 0.980, recall 0.797.
  - The raw misses are continuation lines: the row ends with `(` (1485), `,` (236) or `[` (100), i.e. a hanging
    indent inside brackets.
- **Death is not local.** The block terminators `return / raise / pass / break / continue` force a dedent (precision
  1.000 clean, 0.780 raw), but they cover only 45% (clean) / 25% (raw) of dedents. The other dedents follow an
  ordinary statement: closing a block is a decision of the plan (macro), not a property of the current line (micro).
- **x0** carries 0.19 bits, not 0. A non-space x0 means the row starts at column 0, which is the `def` header (or a
  decorator), and a header always opens a block. Multiplying the row by x0 only rescales it, so it adds nothing
  beyond "header or not". The useful column information is a position: the first non-space column, i.e. the indent
  depth (argmax, not a product).

Predictions:

| prediction | result | verdict |
|---|---|---|
| A1 clean colon rule ≥ 0.99 / 0.99 | 1.0 / 1.0 | PASS |
| A1 raw recall < 0.95 | 0.797 | PASS |
| A2 best single byte < 0.3 × H(dies) | clean 0.32 × H | FAIL (narrow) |
| A2 same, raw | raw 0.18 × H | PASS |
| A3 MI(x0; born) < 0.05 | 0.19 | FAIL (mechanism: x0 marks the header) |

## B. Line break

P(newline right after byte), clean, bytes seen ≥ 200 times: `:` 0.81, `)` 0.49, `]` 0.32, `}` 0.22, digits ~0.1.

The best one-byte rule ("newline after `:`") finds only 36% of line ends (raw 23%) at precision 0.81 (raw 0.64).
A line ends when the statement is complete: that is parse state (bracket depth, expression closed), which is the
job of the GPU syntax checker / byte-LM, not of one byte.

## C. The whole matrix as the .dna "two matrices" (truncated SVD, G ≈ U_k S_k V_k)

| rank k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| bytes exact after rounding, clean | 3.8% | 11.2% | 43.2% | 75.8% | 94.8% |
| non-space bytes exact, clean | 1.2% | 6.4% | 39.2% | 73.4% | 93.7% |
| storage (fp16 factors / raw bytes), clean | 0.37× | 0.74× | **1.37×** | 1.86× | 2.11× |
| ink map pixel accuracy, clean | 90.8% | **96.0%** | 98.9% | 99.9% | 100% |
| ink map: indent of every row right, clean | 46.7% | 73.3% | **95.3%** | 99.7% | 100% |

(At k = 16 many functions are already full rank, since they have fewer than 16 rows.)

- On byte **values** the two matrices do not compress code. 90% of the bytes needs 0.97 (clean) / 0.90 (raw) of full
  rank. At rank 4 the factors already cost 1.37× the bytes and recover 43%. Byte codes are symbols, not magnitudes:
  'a' (97) and 'b' (98) are neighbours as numbers and unrelated as code.
- The **layout** (where there is text) is low-rank. The ink map at rank 2 is 96% right; at rank 4 the indent of 95%
  of rows is right.
- Reading: the shape of code is cheap and smooth, the content is not. This matches the belt: the planner owns
  layout/structure, the writer owns content.

Predictions:

| prediction | result | verdict |
|---|---|---|
| C1 rank 4 recovers < 30% of non-space bytes | clean 39%, raw 28% | FAIL clean, PASS raw |
| C1 90% of bytes needs k ≥ 0.5 min(R, C) | 0.97 / 0.90 | PASS |
| C2 ink rank 2 ≥ 0.85 | 0.960 / 0.956 | PASS |

## D. Forest lens (#9): at which scale does the indentation live?

Multi-scale pyramid of the ink map: average blocks of rows × columns, upsample, threshold 0.5.

| block (rows × cols) | 1×2 | 1×4 | 1×8 | 2×4 | 2×8 | 4×8 |
|---|---|---|---|---|---|---|
| pixels right, clean | 96.8% | 96.4% | 93.4% | 85.7% | 84.4% | 82.5% |
| indent of every row right, clean | 97.3% | **97.5%** | 39.6% | 55.0% | 35.8% | 23.3% |
| indent of every row right, raw | 91.4% | 95.0% | 38.2% | 57.1% | 33.9% | 22.2% |

- Indentation lives at exactly one scale: one row × one indent unit (4 columns).
  - Going coarser in columns (8) mixes two indent levels.
  - Going coarser in rows (2) mixes two lines with different depths.
  - Pixel accuracy hides this: 4×8 still gets 82% of pixels right while losing the indent of 77% of rows.
- The natural "address" of a code position is therefore (line, indent level, offset in the line). This is his
  10-25 "endereço, tanto coluna e linha" idea, with the column quantized by the indent unit for structure.

Predictions: D1 1×4 ≥ 95% → 97.5% PASS; 4×8 < 60% → 23% PASS.

## Not done (declared)

- His "e se o próprio modelo ajusta matrizes?": a learned factorization of code (the model fits U, V itself, e.g. on
  one-hot or embedded bytes instead of raw byte values) was not run; only fixed SVD of byte values and of the ink map.
- Water lens (atoms eating the residual): not run.
- No generation experiment uses these laws yet. The direct use would be:
  - indent birth as a hard rule in the GPU checker (it is already implied by the parser);
  - indent death as a planner output.
